"""Reciprocal pair proposals and single-owner conservative merge commits."""
import warp as wp
from .state import ParticleArrays
from .topology import compatible_faces


@wp.kernel
def propose(d:ParticleArrays,indices:wp.array(dtype=int,ndim=2),counts:wp.array(dtype=int),
            threshold:float,max_radius:float,partners:wp.array(dtype=int)):
    i=wp.tid(); choice=int(-1); best=float(1.e20)
    if d.active[i]==1:
        radius=wp.pow(d.volume[i]*0.238732414637843,1./3.)
        for k in range(counts[i]):
            if k<counts[i]:
                j=indices[i,k]
                if d.active[j]==0 or d.state[i]!=d.state[j]: continue
                other=wp.pow(d.volume[j]*0.238732414637843,1./3.)
                merged=wp.pow((d.volume[i]+d.volume[j])*0.238732414637843,1./3.)
                distance=wp.length(d.position[j]-d.position[i])
                if distance<threshold*(radius+other) and merged<=max_radius:
                    if choice<0 or distance<best or (distance==best and d.ids[j]<d.ids[choice]):
                        choice=j; best=distance
    partners[i]=choice


@wp.kernel
def commit(d:ParticleArrays,partners:wp.array(dtype=int),island_meshes:wp.array(dtype=wp.uint64),
           mapping:wp.array(dtype=int),offsets:wp.array(dtype=int),
           support:wp.array(dtype=int,ndim=2),capture:float,merges:wp.array(dtype=int)):
    i=wp.tid(); j=partners[i]
    if j<0 or partners[j]!=i or d.ids[i]>=d.ids[j]: return
    total=d.volume[i]+d.volume[j]
    p=(d.position[i]*d.volume[i]+d.position[j]*d.volume[j])/total
    velocity=(d.velocity[i]*d.volume[i]+d.velocity[j]*d.volume[j])/total
    if d.state[i]==0:
        handle=island_meshes[d.island[i]]
        query=wp.mesh_query_point_no_sign(handle,p,capture)
        if not query.result: return
        face=mapping[offsets[d.island[i]]+query.face]
        if not compatible_faces(support,d.face[i],face) or not compatible_faces(support,d.face[j],face): return
        p=wp.mesh_eval_position(handle,query.face,query.u,query.v)
        normal=wp.mesh_eval_face_normal(handle,query.face)
        # Velocity is left volume-weighted: the next attached integrator applies
        # the external surface constraint, separately from merge momentum.
        d.normal[i]=normal; d.face[i]=face; d.bary[i]=wp.vec2(query.u,query.v)
    d.position[i]=p; d.velocity[i]=velocity; d.volume[i]=total
    d.age[i]=wp.max(d.age[i],d.age[j])
    d.path[i]+=wp.int64(1)
    d.active[j]=0
    wp.atomic_add(merges,0,1)


def merge_pairs(pool,source,topology,config,neighbors)->None:
    if config.merge_distance_scale<=0: return
    wp.launch(propose,pool.capacity,inputs=[pool.data,neighbors.indices,neighbors.counts,
        config.merge_distance_scale,config.radius*config.maximum_merged_radius_scale,
        pool.merge_partners],device=pool.device)
    wp.launch(commit,pool.capacity,inputs=[pool.data,pool.merge_partners,source.island_handles,
        source.local_to_global,source.island_offsets,topology.face_support,
        config.capture_distance,pool.merge_count],device=pool.device)
