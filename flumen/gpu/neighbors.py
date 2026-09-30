"""Deterministic capped CUDA neighborhoods with surface-locality checks."""
import warp as wp
from .state import ParticleArrays
from .topology import compatible_faces


class NeighborBuffers:
    def __init__(self,capacity,device):
        self.device=device
        self.grid=wp.HashGrid(64,64,64,device=device)
        self.indices=wp.full((capacity,64),-1,dtype=int,device=device)
        self.distances=wp.zeros((capacity,64),dtype=float,device=device)
        self.counts=wp.zeros(capacity,dtype=int,device=device)
        self.overflow=wp.zeros(capacity,dtype=int,device=device)

    def close(self):
        self.grid=self.indices=self.distances=self.counts=self.overflow=None


@wp.kernel
def neighbors_kernel(d:ParticleArrays,grid:wp.uint64,mesh:wp.uint64,
                     support:wp.array(dtype=int,ndim=2),radius:float,
                     indices:wp.array(dtype=int,ndim=2),distances:wp.array(dtype=float,ndim=2),
                     counts:wp.array(dtype=int),overflow:wp.array(dtype=int)):
    i=wp.tid()
    count=int(0); eligible=int(0)
    for k in range(64): indices[i,k]=-1
    if d.active[i]==1:
        query=wp.hash_grid_query(grid,d.position[i],radius)
        for j in query:
            if j==i or d.active[j]==0 or d.state[i]!=d.state[j] or d.island[i]!=d.island[j]: continue
            delta=d.position[j]-d.position[i]
            distance=wp.length(delta)
            if distance>radius: continue
            if d.state[i]==0:
                if wp.dot(d.normal[i],d.normal[j])<.5 or not compatible_faces(support,d.face[i],d.face[j]): continue
            elif distance>1.e-7:
                hit=wp.mesh_query_ray(mesh,d.position[i],delta/distance,wp.max(0.,distance-1.e-6))
                if hit.result: continue
            eligible+=1
            insert=count
            for k in range(64):
                if k<count:
                    other=indices[i,k]
                    if insert==count and (distance<distances[i,k] or (distance==distances[i,k] and d.ids[j]<d.ids[other])):
                        insert=k
            if insert<64:
                for reverse in range(63):
                    k=63-reverse
                    if k>insert and k<=count:
                        indices[i,k]=indices[i,k-1]; distances[i,k]=distances[i,k-1]
                indices[i,insert]=j; distances[i,insert]=distance
                count=wp.min(64,count+1)
    counts[i]=count
    overflow[i]=wp.max(0,eligible-64)


def build_neighbors(pool,source,topology,config,buffers)->None:
    radius=config.radius*config.interaction_radius_scale
    buffers.grid.build(pool.data.position,radius)
    wp.launch(neighbors_kernel,pool.capacity,inputs=[pool.data,buffers.grid.id,source.mesh.id,
        topology.face_support,radius,buffers.indices,buffers.distances,buffers.counts,buffers.overflow],device=pool.device)
