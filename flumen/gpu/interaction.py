"""Prior-state tangent attraction, spacing pressure and surface damping."""
import warp as wp
from .state import ParticleArrays
from .motion import tangent


class InteractionBuffers:
    def __init__(self,capacity,device):
        self.positions=wp.zeros(capacity,dtype=wp.vec3,device=device)
        self.velocities=wp.zeros(capacity,dtype=wp.vec3,device=device)
        self.forces=wp.zeros(capacity,dtype=wp.vec3,device=device)
        self.steps=wp.zeros(1,dtype=int,device=device)
        from .neighbors import NeighborBuffers
        self.neighbors=NeighborBuffers(capacity,device)

    def close(self):
        self.neighbors.close()
        self.positions=self.velocities=self.forces=self.steps=None


@wp.kernel
def forces_kernel(d:ParticleArrays,previous:wp.array(dtype=wp.vec3),
                  indices:wp.array(dtype=int,ndim=2),counts:wp.array(dtype=int),
                  range_radius:float,cohesion:float,repulsion:float,damping:float,
                  forces:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    attraction=wp.vec3(0.); pressure=wp.vec3(0.); viscous=wp.vec3(0.)
    if d.active[i]==1 and d.state[i]==0:
        radius=wp.pow(d.volume[i]*0.238732414637843,1./3.)
        for k in range(64):
            if k<counts[i]:
                j=indices[i,k]
                delta=tangent(d.position[j]-d.position[i],d.normal[i])
                length=wp.length(delta)
                if length>1.e-8:
                    direction=delta/length
                    pair_radius=radius+wp.pow(d.volume[j]*0.238732414637843,1./3.)
                    spacing=.5*pair_radius
                    if length<spacing:
                        pressure-=direction*(1.-length/wp.max(spacing,1.e-8))
                    else:
                        attraction+=direction*(1.-length/range_radius)
                viscous+=tangent(previous[j]-previous[i],d.normal[i])
        norm=wp.length(attraction)
        if norm>1.: attraction/=norm
        norm=wp.length(pressure)
        if norm>1.: pressure/=norm
        if counts[i]>0: viscous/=float(counts[i])
    forces[i]=attraction*cohesion+pressure*repulsion+viscous*damping


def compute_forces(pool,config,neighbors,scratch)->None:
    wp.copy(scratch.positions,pool.data.position)
    wp.copy(scratch.velocities,pool.data.velocity)
    wp.launch(forces_kernel,pool.capacity,inputs=[pool.data,scratch.velocities,neighbors.indices,
        neighbors.counts,config.radius*config.interaction_radius_scale,config.cohesion_acceleration,
        config.repulsion_acceleration,config.surface_damping,scratch.forces],device=pool.device)
