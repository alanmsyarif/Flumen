"""Independent attached/free particle integration on persistent CUDA BVHs."""
from math import cos, pi
import warp as wp
from .state import ParticleArrays


@wp.func
def tangent(v: wp.vec3, n: wp.vec3):
    return v-n*wp.dot(v,n)


@wp.kernel
def integrate(d: ParticleArrays, mesh: wp.uint64, island_meshes: wp.array(dtype=wp.uint64),
              face_islands: wp.array(dtype=int), local_to_global: wp.array(dtype=int),
              island_offsets: wp.array(dtype=int), gravity: wp.vec3, frame_dt: float,
              minimum: int, travel_limit: float, resistance: float, adhesion: float,
              capture_distance: float, capture_speed: float, turn_cos: float,
              lifetime: float, kill_height: float, ledger: wp.array(dtype=wp.float64),
              step_count: wp.array(dtype=int),fixed_steps:int):
    i = wp.tid()
    if d.active[i] == 0:
        return
    p = d.position[i]
    v = d.velocity[i]
    n = d.normal[i]
    state = d.state[i]
    radius = wp.pow(d.volume[i]*0.238732414637843,1.0/3.0)
    needed = int(wp.ceil((wp.length(v)+wp.length(gravity)*frame_dt)*frame_dt/travel_limit))
    steps = wp.min(64,wp.max(minimum,needed))
    if fixed_steps>0: steps=fixed_steps
    wp.atomic_max(step_count,0,steps)
    h = frame_dt/float(steps)
    if fixed_steps==0: d.limited[i] = 0
    if needed > 64:
        d.limited[i] = 1
    for substep in range(steps):
        previous_state = state
        displacement = wp.vec3(0.0)
        if state == 0:
            u = tangent(v,n)
            a = tangent(gravity,n)
            x = resistance*h
            decay = wp.exp(-x)
            A = h*(1.0-x*.5+x*x/6.0)
            B = h*h*(.5-x/6.0+x*x/24.0)
            if x >= .001:
                A = (1.0-decay)/resistance
                B = (h-A)/resistance
            displacement = u*A+a*B
            v = u*decay+a*A
        else:
            displacement = v*h+gravity*(.5*h*h)
            v += gravity*h
        travel = wp.length(displacement)
        if travel > travel_limit:
            factor = travel_limit/travel
            displacement *= factor
            v *= factor
            d.limited[i] = 1
        candidate = p+displacement
        if state == 0:
            local_mesh = island_meshes[d.island[i]]
            query = wp.mesh_query_point_no_sign(local_mesh,candidate,capture_distance+travel_limit)
            detach = wp.dot(gravity,n) > adhesion
            if query.result:
                q = wp.mesh_eval_position(local_mesh,query.face,query.u,query.v)
                next_n = wp.mesh_eval_face_normal(local_mesh,query.face)
                residual = wp.length(tangent(candidate-q,n))
                lost = wp.length(candidate-q) > capture_distance
                if travel > 1.e-5 and residual > .5*wp.length(displacement):
                    lost = True
                detach = detach or lost
                if not detach:
                    if wp.dot(n,next_n) > turn_cos:
                        p = q
                        n = next_n
                        v = tangent(v,n)
                        d.face[i] = local_to_global[island_offsets[d.island[i]]+query.face]
                        d.bary[i] = wp.vec2(query.u,query.v)
                    else:
                        v = wp.vec3(0.0)
            else:
                detach = True
            if detach:
                p = candidate+n*(radius+1.e-5)
                state = 1
        else:
            length = wp.length(displacement)
            ray = wp.mesh_query_ray(mesh,p,wp.normalize(displacement),length+radius)
            nearest = wp.mesh_query_point_no_sign(mesh,candidate,radius+1.e-5)
            hit = ray.result and length > 1.e-12
            if hit or nearest.result:
                face = nearest.face
                bu = nearest.u
                bv = nearest.v
                if hit:
                    face = ray.face
                    bu = ray.u
                    bv = ray.v
                q = wp.mesh_eval_position(mesh,face,bu,bv)
                canonical = wp.mesh_eval_face_normal(mesh,face)
                normal = canonical
                front = wp.dot(p-q,canonical) >= -1.e-8
                if not front:
                    normal = -canonical
                inward = wp.dot(v,normal)
                p = q+normal*(radius+1.e-5)
                v -= normal*wp.min(inward,0.0)
                if hit and front and inward < 0.0 and -inward <= capture_speed:
                    state = 0
                    p = q
                    n = canonical
                    v = tangent(v,n)
                    d.island[i] = face_islands[face]
                    d.face[i] = face
                    d.bary[i] = wp.vec2(bu,bv)
            else:
                p = candidate
        if state != previous_state:
            d.path[i] += wp.int64(1)
        d.age[i] += h
        if d.age[i] >= lifetime or p[2] < kill_height:
            d.active[i] = 0
            wp.atomic_add(ledger,1,wp.float64(d.volume[i]))
            break
    d.position[i] = p
    d.velocity[i] = v
    d.normal[i] = n
    d.state[i] = state


def advance(pool, source, config, dt: float) -> None:
    if dt <= 0:
        return
    pool.step_count.zero_()
    wp.launch(integrate,pool.capacity,inputs=[pool.data,source.mesh.id,source.island_handles,
        source.islands,source.local_to_global,source.island_offsets,
        wp.vec3(*config.gravity),dt,config.minimum_substeps,config.max_travel,
        config.resistance,config.adhesion,config.capture_distance,config.capture_speed,
        cos(config.normal_turn_limit*pi/180),config.lifetime,config.kill_height,pool.ledger,
        pool.step_count,0],device=pool.device)


@wp.kernel
def shared_requirement(d:ParticleArrays,forces:wp.array(dtype=wp.vec3),gravity:wp.vec3,
                       dt:float,travel:float,minimum:int,damping:float,steps:wp.array(dtype=int)):
    i=wp.tid()
    if d.active[i]==1:
        acceleration=wp.length(gravity)+wp.length(forces[i])
        needed=int(wp.ceil((wp.length(d.velocity[i])+acceleration*dt)*dt/travel))
        if d.state[i]==0: needed=wp.max(needed,int(wp.ceil(dt*damping*4.)))
        wp.atomic_max(steps,0,wp.max(minimum,needed))


@wp.kernel
def kick(d:ParticleArrays,forces:wp.array(dtype=wp.vec3),dt:float,first:int,over_budget:int,
         overflow:wp.array(dtype=int),overflow_count:wp.array(dtype=int)):
    i=wp.tid()
    if first==1: d.limited[i]=over_budget
    if d.active[i]==1:
        d.velocity[i]+=forces[i]*dt
        wp.atomic_add(overflow_count,0,overflow[i])


def advance_coupled(pool,source,topology,config,scratch,dt:float)->None:
    if dt<=0: return
    from .neighbors import build_neighbors
    from .interaction import compute_forces
    from .merge import merge_pairs
    build_neighbors(pool,source,topology,config,scratch.neighbors)
    compute_forces(pool,config,scratch.neighbors,scratch)
    scratch.steps.zero_()
    wp.launch(shared_requirement,pool.capacity,inputs=[pool.data,scratch.forces,wp.vec3(*config.gravity),
        dt,config.max_travel,config.minimum_substeps,config.surface_damping,scratch.steps],device=pool.device)
    needed=max(config.minimum_substeps,int(scratch.steps.numpy()[0]))
    steps=min(64,needed); interval=dt/steps
    pool.step_count.assign([steps]); pool.neighbor_overflow_count.zero_()
    for substep in range(steps):
        if substep>0:
            build_neighbors(pool,source,topology,config,scratch.neighbors)
            compute_forces(pool,config,scratch.neighbors,scratch)
        wp.launch(kick,pool.capacity,inputs=[pool.data,scratch.forces,interval,int(substep==0),
            int(needed>64),scratch.neighbors.overflow,pool.neighbor_overflow_count],device=pool.device)
        wp.launch(integrate,pool.capacity,inputs=[pool.data,source.mesh.id,source.island_handles,
            source.islands,source.local_to_global,source.island_offsets,wp.vec3(*config.gravity),interval,
            1,config.max_travel,config.resistance,config.adhesion,config.capture_distance,
            config.capture_speed,cos(config.normal_turn_limit*pi/180),config.lifetime,config.kill_height,
            pool.ledger,pool.step_count,1],device=pool.device)
        # Rebuild after motion: pair membership and distance must agree at commit.
        build_neighbors(pool,source,topology,config,scratch.neighbors)
        merge_pairs(pool,source,topology,config,scratch.neighbors)
