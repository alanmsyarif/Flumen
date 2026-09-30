"""Volume-normalized live thickness and an independent physical-time wetness field."""
from dataclasses import dataclass
import numpy as np
import warp as wp
from .state import ParticleArrays
from .topology import compatible_faces


@dataclass
class SurfaceBatch:
    thickness: np.ndarray
    wetness: np.ndarray
    represented_volume: float
    unrepresented_volume: float
    support_overflow: int = 0


class SurfaceState:
    def __init__(self,topology,config,device):
        self.topology=topology; self.device=device
        count=len(topology.vertices)
        self.thickness=wp.zeros(count,dtype=float,device=device)
        self.wetness=wp.zeros(count,dtype=float,device=device)
        self.indices=wp.full((config.capacity,64),-1,dtype=int,device=device)
        self.weights=wp.zeros((config.capacity,64),dtype=float,device=device)
        self.summary=wp.zeros(3,dtype=wp.float64,device=device)
        self.host_thickness=wp.zeros(count,dtype=float,device='cpu',pinned=True)
        self.host_wetness=wp.zeros(count,dtype=float,device='cpu',pinned=True)

    def reset(self):
        self.thickness.zero_(); self.wetness.zero_(); self.summary.zero_()

    def close(self):
        self.thickness=self.wetness=self.indices=self.weights=self.summary=None
        self.host_thickness=self.host_wetness=None


@wp.kernel
def footprints(d:ParticleArrays,grid:wp.uint64,points:wp.array(dtype=wp.vec3),
               normals:wp.array(dtype=wp.vec3),areas:wp.array(dtype=float),
               faces:wp.array(dtype=int),support:wp.array(dtype=int,ndim=2),radius:float,
               indices:wp.array(dtype=int,ndim=2),weights:wp.array(dtype=float,ndim=2),
               summary:wp.array(dtype=wp.float64)):
    i=wp.tid(); count=int(0); eligible=int(0)
    for k in range(64): indices[i,k]=-1; weights[i,k]=0.
    if d.active[i]==0 or d.state[i]!=0: return
    query=wp.hash_grid_query(grid,d.position[i],radius)
    for j in query:
        distance=wp.length(points[j]-d.position[i])
        if distance>=radius or wp.dot(normals[j],d.normal[i])<.5: continue
        if not compatible_faces(support,d.face[i],faces[j]): continue
        eligible+=1
        insert=count
        for k in range(64):
            if k<count:
                other=indices[i,k]
                old_distance=wp.length(points[other]-d.position[i])
                if insert==count and (distance<old_distance or (distance==old_distance and j<other)): insert=k
        if insert<64:
            for reverse in range(63):
                k=63-reverse
                if k>insert and k<=count: indices[i,k]=indices[i,k-1]
            indices[i,insert]=j; count=wp.min(64,count+1)
    normalization=float(0.)
    for k in range(64):
        if k<count:
            j=indices[i,k]
            q=wp.length_sq(points[j]-d.position[i])/(radius*radius)
            weight=(1.-q)*(1.-q)
            weights[i,k]=weight; normalization+=weight*areas[j]
    if normalization>1.e-20:
        for k in range(64):
            if k<count: weights[i,k]*=d.volume[i]/normalization
        wp.atomic_add(summary,0,wp.float64(d.volume[i]))
    else:
        wp.atomic_add(summary,1,wp.float64(d.volume[i]))
        for k in range(64): indices[i,k]=-1
    wp.atomic_add(summary,2,wp.float64(wp.max(0,eligible-64)))


@wp.kernel
def deposit(indices:wp.array(dtype=int,ndim=2),weights:wp.array(dtype=float,ndim=2),
            thickness:wp.array(dtype=float)):
    i,k=wp.tid()
    j=indices[i,k]
    if j>=0: wp.atomic_add(thickness,j,weights[i,k])


@wp.kernel
def wetness_step(thickness:wp.array(dtype=float),wetness:wp.array(dtype=float),
                 dt:float,deposit_rate:float,drying:float):
    i=wp.tid(); value=wetness[i]
    if thickness[i]>1.e-5:
        value=1.-(1.-value)*wp.exp(-deposit_rate*dt)
    else:
        value*=wp.exp(-drying*dt)
    wetness[i]=wp.clamp(value,0.,1.)


def build_surface(topology,config,device)->SurfaceState:
    return SurfaceState(topology,config,getattr(device,'alias',device))


def advance_wetness(surface,config,dt:float)->None:
    if dt>0:
        wp.launch(wetness_step,len(surface.topology.vertices),inputs=[surface.thickness,surface.wetness,
            dt,config.wetness_deposit_rate,config.wetness_drying_rate],device=surface.device)


def update_surface(surface,pool,source,config,dt:float)->None:
    t=surface.topology
    surface.thickness.zero_(); surface.summary.zero_()
    wp.launch(footprints,pool.capacity,inputs=[pool.data,t.proxy_grid.id,t.points_gpu,t.normals_gpu,
        t.areas_gpu,t.vertex_faces_gpu,t.face_support,config.radius*config.interaction_radius_scale,
        surface.indices,surface.weights,surface.summary],device=surface.device)
    wp.launch(deposit,(pool.capacity,64),inputs=[surface.indices,surface.weights,surface.thickness],device=surface.device)
    advance_wetness(surface,config,dt)


def snapshot_surface(surface)->SurfaceBatch:
    wp.copy(surface.host_thickness,surface.thickness)
    wp.copy(surface.host_wetness,surface.wetness)
    wp.synchronize_device(surface.device)
    summary=surface.summary.numpy()
    return SurfaceBatch(surface.host_thickness.numpy().copy(),surface.host_wetness.numpy().copy(),
                        float(summary[0]),float(summary[1]),int(summary[2]))
