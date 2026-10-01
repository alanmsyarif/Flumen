"""Bounded deterministic cell aggregation and conservative marker resampling."""
import numpy as np
import warp as wp
from warp.utils import radix_sort_pairs, array_scan
from .state import ParticleArrays


class AggregateBuffers:
    def __init__(self, capacity, device):
        self.keys=wp.empty(capacity*2,dtype=wp.int64,device=device)
        self.values=wp.empty(capacity*2,dtype=int,device=device)
        self.volume_keys=wp.empty(capacity*2,dtype=float,device=device)
        self.cells=wp.empty(capacity,dtype=wp.vec3i,device=device)
        self.partners=wp.full(capacity,-1,dtype=int,device=device)
        self.free_prefix=wp.empty(capacity,dtype=int,device=device)
        self.free_slots=wp.empty(capacity,dtype=int,device=device)
        self.count=wp.zeros(1,dtype=int,device=device)
        self.merged=wp.zeros(1,dtype=wp.int64,device=device)
        self.resampled=wp.zeros(1,dtype=wp.int64,device=device)
        self.domain_limited=wp.zeros(1,dtype=int,device=device)

    def close(self):
        for name in tuple(self.__dict__): setattr(self,name,None)


@wp.kernel
def id_keys(d: ParticleArrays, keys: wp.array(dtype=wp.int64), values: wp.array(dtype=int)):
    i=wp.tid(); keys[i]=d.ids[i]; values[i]=i


@wp.func
def particle_group(d: ParticleArrays, i: int, faces: int):
    if d.state[i]==0: return d.face[i]
    return faces+d.island[i]+1


@wp.kernel
def cell_keys(d: ParticleArrays, values: wp.array(dtype=int), keys: wp.array(dtype=wp.int64),
              cells: wp.array(dtype=wp.vec3i), origin: wp.vec3, pitch: float, faces: int,
              limited: wp.array(dtype=int)):
    rank=wp.tid(); i=values[rank]
    cell=wp.vec3i(int(wp.floor((d.position[i][0]-origin[0])/pitch)),
                 int(wp.floor((d.position[i][1]-origin[1])/pitch)),
                 int(wp.floor((d.position[i][2]-origin[2])/pitch)))
    cells[i]=cell
    # A hash collision can reduce candidate coverage, never authorize a merge.
    hashed=(wp.int64(cell[0])*wp.int64(73856093))^(wp.int64(cell[1])*wp.int64(19349663))^(wp.int64(cell[2])*wp.int64(83492791))
    hashed=hashed & wp.int64(4294967295)
    group=particle_group(d,i,faces)
    keys[rank]=wp.int64(group)*wp.int64(4294967296)+hashed
    if d.active[i]==0: keys[rank]=wp.int64(9223372036854775807)
    if wp.abs(cell[0])>1073741823 or wp.abs(cell[1])>1073741823 or wp.abs(cell[2])>1073741823:
        wp.atomic_add(limited,0,1)
        keys[rank]=wp.int64(9223372036854775807)


@wp.kernel
def propose_pairs(d: ParticleArrays, keys: wp.array(dtype=wp.int64), values: wp.array(dtype=int),
                  cells: wp.array(dtype=wp.vec3i), partners: wp.array(dtype=int), mesh: wp.uint64,
                  scale: float, max_volume: float):
    rank=wp.tid(); i=values[rank]; partners[i]=-1
    if d.active[i]!=1: return
    # Prefer consecutive sorted IDs, with at most seven alternative local probes.
    for probe in range(8):
        candidate=rank^1
        if probe>0:
            offset=(probe+1)//2
            if probe%2==0: offset=-offset
            candidate=rank+offset
        if candidate<0 or candidate>=values.shape[0]//2 or candidate==rank: continue
        j=values[candidate]
        if keys[rank]!=keys[candidate] or d.active[j]!=1: continue
        a=cells[i]; b=cells[j]
        if a[0]!=b[0] or a[1]!=b[1] or a[2]!=b[2] or d.state[i]!=d.state[j] or d.island[i]!=d.island[j]: continue
        if d.state[i]==0 and (d.face[i]!=d.face[j] or wp.dot(d.normal[i],d.normal[j])<.5): continue
        if d.volume[i]+d.volume[j]>max_volume: continue
        distance=wp.length(d.position[i]-d.position[j])
        reach=scale*(wp.pow(d.volume[i]*.238732414637843,1./3.)+wp.pow(d.volume[j]*.238732414637843,1./3.))
        if distance>reach: continue
        if d.state[i]==1 and distance>1.e-6:
            hit=wp.mesh_query_ray(mesh,d.position[i],(d.position[j]-d.position[i])/distance,wp.max(0.,distance-1.e-6))
            if hit.result: continue
        partners[i]=j; break


@wp.kernel
def apply_pairs(d: ParticleArrays, partners: wp.array(dtype=int), merged: wp.array(dtype=wp.int64)):
    i=wp.tid(); j=partners[i]
    if j<0 or partners[j]!=i or d.ids[i]>=d.ids[j]: return
    a=d.volume[i]; b=d.volume[j]; total=a+b
    if total<=0.: return
    d.position[i]=(d.position[i]*a+d.position[j]*b)/total
    d.velocity[i]=(d.velocity[i]*a+d.velocity[j]*b)/total
    d.bary[i]=(d.bary[i]*a+d.bary[j]*b)/total
    d.normal[i]=wp.normalize(d.normal[i]*a+d.normal[j]*b)
    d.age[i]=wp.max(d.age[i],d.age[j])
    d.volume[i]=total; d.volume[j]=0.; d.active[j]=0
    wp.atomic_add(merged,0,wp.int64(1))


@wp.kernel
def free_mask(d: ParticleArrays, mask: wp.array(dtype=int)):
    i=wp.tid(); mask[i]=1-d.active[i]


@wp.kernel
def free_list(mask: wp.array(dtype=int), prefix: wp.array(dtype=int), slots: wp.array(dtype=int),
              count: wp.array(dtype=int)):
    i=wp.tid()
    if mask[i]==1: slots[prefix[i]]=i
    if i==0:
        last=mask.shape[0]-1; count[0]=mask[last]+prefix[last]


@wp.kernel
def volume_keys(d: ParticleArrays, values: wp.array(dtype=int), keys: wp.array(dtype=float)):
    rank=wp.tid(); i=values[rank]; keys[rank]=1.e20
    if d.active[i]==1: keys[rank]=-d.volume[i]


@wp.kernel
def split_particles(d: ParticleArrays, donors: wp.array(dtype=int), free_slots: wp.array(dtype=int),
                    first_id: wp.int64):
    rank=wp.tid(); donor=donors[rank]; child=free_slots[rank]
    before=d.volume[donor]
    retained=before*.99
    d.volume[donor]=retained; d.volume[child]=before-retained
    d.position[child]=d.position[donor]; d.velocity[child]=d.velocity[donor]
    d.normal[child]=d.normal[donor]; d.age[child]=d.age[donor]
    d.state[child]=d.state[donor]; d.island[child]=d.island[donor]
    d.face[child]=d.face[donor]; d.bary[child]=d.bary[donor]
    d.path[child]=d.path[donor]; d.limited[child]=d.limited[donor]
    d.ids[child]=first_id+wp.int64(rank); d.active[child]=1


def _scratch(pool):
    if getattr(pool,'field_aggregate',None) is None:
        pool.field_aggregate=AggregateBuffers(pool.capacity,pool.device)
    return pool.field_aggregate


def aggregate_field_particles(pool, prepared, config) -> None:
    if config.merge_distance_scale<=0: return
    buffers=_scratch(pool); n=pool.capacity
    wp.launch(id_keys,n,inputs=[pool.data,buffers.keys,buffers.values],device=pool.device)
    radix_sort_pairs(buffers.keys,buffers.values,n)
    buffers.domain_limited.zero_()
    origin=prepared.source.vertices_cpu.min(axis=0)
    wp.launch(cell_keys,n,inputs=[pool.data,buffers.values,buffers.keys,buffers.cells,
        wp.vec3(*origin),2*config.radius*config.maximum_merged_radius_scale,
        len(prepared.source.triangles_cpu),buffers.domain_limited],device=pool.device)
    radix_sort_pairs(buffers.keys,buffers.values,n)
    max_volume=4*np.pi/3*(config.radius*config.maximum_merged_radius_scale)**3
    wp.launch(propose_pairs,n,inputs=[pool.data,buffers.keys,buffers.values,buffers.cells,buffers.partners,
        prepared.source.mesh.id,config.merge_distance_scale,max_volume],device=pool.device)
    wp.launch(apply_pairs,n,inputs=[pool.data,buffers.partners,buffers.merged],device=pool.device)


def resample_particles(pool, prepared, target: int) -> None:
    if isinstance(target,bool) or not isinstance(target,int) or not 0<=target<=pool.capacity:
        raise ValueError('Resampling target must be within capacity')
    if target==0: return
    buffers=_scratch(pool); n=pool.capacity
    for _ in range(20):
        wp.launch(free_mask,n,inputs=[pool.data,pool.mask],device=pool.device)
        array_scan(pool.mask,buffers.free_prefix,inclusive=False)
        wp.launch(free_list,n,inputs=[pool.mask,buffers.free_prefix,buffers.free_slots,buffers.count],device=pool.device)
        live=n-int(buffers.count.numpy()[0]); missing=target-live
        if missing<=0 or live==0: return
        count=min(live,missing)
        if pool.next_id>2**63-1-count: raise RuntimeError('Particle ID range exhausted')
        wp.launch(id_keys,n,inputs=[pool.data,buffers.keys,buffers.values],device=pool.device)
        radix_sort_pairs(buffers.keys,buffers.values,n)
        wp.launch(volume_keys,n,inputs=[pool.data,buffers.values,buffers.volume_keys],device=pool.device)
        radix_sort_pairs(buffers.volume_keys,buffers.values,n)
        wp.launch(split_particles,count,inputs=[pool.data,buffers.values,buffers.free_slots,wp.int64(pool.next_id)],device=pool.device)
        pool.next_id+=count
        buffers.resampled.fill_(int(buffers.resampled.numpy()[0])+count)
