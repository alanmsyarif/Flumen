"""Transactional local-chart transport and bounded provenance-aware contacts."""
from math import cos, pi
from time import perf_counter
import warp as wp
import numpy as np
from warp.utils import array_scan
from .state import ParticleArrays
from .surface_chart import chart_anchor
from .motion import project_anchor, tangent
from .field_solver import deposit_attached, evolve_field

COPIED = ('position','velocity','normal','age','active','state','island','face','bary','limited')
SHARED = ('volume','ids','path')


class MotionBuffers:
    def __init__(self, pool, chart):
        self.proposed=ParticleArrays()
        for name in COPIED:
            array=getattr(pool.data,name)
            setattr(self.proposed,name,wp.empty(pool.capacity,dtype=array.dtype,device=pool.device))
        self.mask=wp.zeros(pool.capacity,dtype=int,device=pool.device)
        self.prefix=wp.zeros(pool.capacity,dtype=int,device=pool.device)
        self.queue=wp.empty(min(65536,pool.capacity),dtype=int,device=pool.device)
        self.count=wp.zeros(1,dtype=int,device=pool.device)
        self.unresolved=wp.zeros(1,dtype=int,device=pool.device)
        self.wet_backup=wp.empty(len(chart.vertices),dtype=float,device=pool.device)
        self.velocity_backup=wp.empty(len(chart.vertices),dtype=wp.vec3,device=pool.device)
        self.last_fallback_count=0

    def close(self):
        for name in tuple(self.__dict__): setattr(self,name,None)


@wp.func
def rotate_tangent(value: wp.vec3, old: wp.vec3, new: wp.vec3):
    axis=wp.cross(old,new)
    c=wp.clamp(wp.dot(old,new),-1.,1.)
    result=value
    if c > -.9999:
        result=value+wp.cross(axis,value)+wp.cross(axis,wp.cross(axis,value))/(1.+c)
    return tangent(result,new)


@wp.func
def walk_chart(mesh: wp.uint64, adjacency: wp.array(dtype=int,ndim=2), face: int,
               bary: wp.vec2, p: wp.vec3, displacement: wp.vec3, velocity: wp.vec3, turn: float):
    normal=wp.mesh_eval_face_normal(mesh,face)
    remaining=tangent(displacement,normal)
    state=int(0); limited=int(0); complete=bool(False)
    for step in range(8):
        candidate=p+remaining
        inside,q,next_bary=project_anchor(mesh,face,candidate)
        target=wp.vec3(next_bary[0],next_bary[1],1.-next_bary[0]-next_bary[1])
        if target[0]>=-1.e-6 and target[1]>=-1.e-6 and target[2]>=-1.e-6:
            target=wp.vec3(wp.max(0.,target[0]),wp.max(0.,target[1]),wp.max(0.,target[2]))
            target/=target[0]+target[1]+target[2]
            bary=wp.vec2(target[0],target[1]); p=q; complete=True; break
        current=wp.vec3(bary[0],bary[1],1.-bary[0]-bary[1])
        crossing=float(1.); corner=int(-1)
        for k in range(3):
            if target[k]<0. and current[k]-target[k]>1.e-12:
                fraction=wp.max(0.,current[k])/(current[k]-target[k])
                if fraction<crossing: crossing=fraction; corner=k
        if corner<0: break
        edge=p+remaining*crossing
        neighbor=adjacency[face,corner]
        if neighbor<0:
            p=candidate; state=1; complete=True; break
        next_normal=wp.mesh_eval_face_normal(mesh,neighbor)
        if wp.dot(normal,next_normal)<turn:
            p=edge; velocity=wp.vec3(0.); limited=1
            edge_valid,edge_point,bary=project_anchor(mesh,face,p)
            complete=True; break
        remaining=rotate_tangent(remaining*(1.-crossing),normal,next_normal)
        velocity=rotate_tangent(velocity,normal,next_normal)
        face=neighbor; normal=next_normal
        edge_valid,p,bary=project_anchor(mesh,face,edge)
    return complete,p,velocity,normal,face,bary,state,limited,remaining


@wp.func
def probe_contact(point: wp.vec3, low: wp.vec3, dims: wp.vec3i, spacing: float,
                  distances: wp.array(dtype=float), faces: wp.array(dtype=int),
                  ambiguity: wp.array(dtype=int)):
    high=low+wp.vec3(float(dims[0]-1),float(dims[1]-1),float(dims[2]-1))*spacing
    delta=wp.vec3(wp.max(low[0]-point[0],wp.max(0.,point[0]-high[0])),
                  wp.max(low[1]-point[1],wp.max(0.,point[1]-high[1])),
                  wp.max(low[2]-point[2],wp.max(0.,point[2]-high[2])))
    lower=wp.length(delta); face=int(-1); ambiguous=int(0)
    if lower>0.: return lower,face,ambiguous
    coordinate=(point-low)/spacing
    ix=wp.min(dims[0]-2,wp.max(0,int(wp.floor(coordinate[0]))))
    iy=wp.min(dims[1]-2,wp.max(0,int(wp.floor(coordinate[1]))))
    iz=wp.min(dims[2]-2,wp.max(0,int(wp.floor(coordinate[2]))))
    fraction=coordinate-wp.vec3(float(ix),float(iy),float(iz))
    interpolated=float(0.); closest=float(1.e10)
    for corner in range(8):
        x=corner%2; y=(corner//2)%2; z=corner//4
        index=(iz+z)*dims[0]*dims[1]+(iy+y)*dims[0]+ix+x
        weight=wp.where(x==0,1.-fraction[0],fraction[0])*wp.where(y==0,1.-fraction[1],fraction[1])*wp.where(z==0,1.-fraction[2],fraction[2])
        interpolated+=weight*distances[index]
        ambiguous=wp.max(ambiguous,ambiguity[index])
        if distances[index]<closest: closest=distances[index]; face=faces[index]
    return wp.max(0.,interpolated-1.7320508075688772*spacing),face,ambiguous


@wp.func
def triangle_contact(mesh: wp.uint64, face: int, start: wp.vec3, end: wp.vec3,
                     velocity: wp.vec3, radius: float):
    hit=bool(False); front=bool(True); p=end; v=velocity; bary=wp.vec2(0.); normal=wp.vec3(0.)
    if face>=0:
        canonical=wp.mesh_eval_face_normal(mesh,face)
        a=wp.mesh_eval_position(mesh,face,1.,0.)
        front=wp.dot(start-a,canonical)>=0.
        normal=wp.where(front,canonical,-canonical)
        old_h=wp.dot(start-a,normal); new_h=wp.dot(end-a,normal)
        if new_h<=radius and (old_h>=radius or old_h>=0.):
            fraction=float(0.)
            if old_h>radius and old_h-new_h>1.e-12:
                fraction=wp.clamp((old_h-radius)/(old_h-new_h),0.,1.)
            candidate=start+(end-start)*fraction-normal*radius
            valid,q,bary=project_anchor(mesh,face,candidate)
            if valid:
                p=q+normal*(radius+1.e-5)
                v-=normal*wp.min(0.,wp.dot(v,normal))
                hit=True
    return hit,front,p,v,normal,bary


@wp.kernel
def transport(d: ParticleArrays, proposed: ParticleArrays, mesh: wp.uint64,
              adjacency: wp.array(dtype=int,ndim=2), source_islands: wp.array(dtype=int),
              chart_triangles: wp.array(dtype=wp.vec3i), level: int, field_velocity: wp.array(dtype=wp.vec3),
              low: wp.vec3, dims: wp.vec3i, spacing: float, distances: wp.array(dtype=float),
              contact_faces: wp.array(dtype=int), ambiguity: wp.array(dtype=int),
              gravity: wp.vec3, adhesion: float, capture_speed: float, turn: float, dt: float,
              fallback: wp.array(dtype=int)):
    i=wp.tid(); fallback[i]=0
    if d.active[i]!=1: return
    p=d.position[i]; old_velocity=d.velocity[i]; velocity=old_velocity
    normal=d.normal[i]; face=d.face[i]; bary=d.bary[i]; state=d.state[i]
    radius=wp.pow(d.volume[i]*.238732414637843,1./3.)
    if state==0:
        if face<0 or face>=adjacency.shape[0]: fallback[i]=1; return
        child,weights=chart_anchor(face,bary,level)
        nodes=chart_triangles[child]
        velocity=tangent(field_velocity[nodes[0]]*weights[0]+field_velocity[nodes[1]]*weights[1]+field_velocity[nodes[2]]*weights[2],normal)
        displacement=(tangent(old_velocity,normal)+velocity)*(.5*dt)
        if wp.dot(gravity,normal)>adhesion:
            state=1; p+=displacement+normal*(radius+1.e-5)
        else:
            complete,p,velocity,normal,face,bary,state,limited,remaining=walk_chart(mesh,adjacency,face,bary,p,displacement,velocity,turn)
            proposed.limited[i]=limited
            if not complete:
                p+=remaining; fallback[i]=1
            if state==1: p+=normal*(radius+1.e-5)
    else:
        displacement=velocity*dt+gravity*(.5*dt*dt)
        velocity+=gravity*dt
        candidate=p+displacement
        travel=wp.length(displacement); progressed=float(0.); clear=bool(False); hit=bool(False)
        contact_front=bool(False); contact_bary=wp.vec2(0.); contact_normal=wp.vec3(0.)
        for iteration in range(8):
            point=p
            if travel>1.e-12: point+=displacement*(progressed/travel)
            distance,contact_face,ambiguous=probe_contact(point,low,dims,spacing,distances,contact_faces,ambiguity)
            if ambiguous!=0: break
            if distance>travel-progressed+radius: clear=True; break
            if contact_face>=0:
                hit,contact_front,candidate,velocity,contact_normal,contact_bary=triangle_contact(mesh,contact_face,p,candidate,velocity,radius)
                if hit:
                    face=contact_face; bary=contact_bary; normal=contact_normal
                    proposed.island[i]=source_islands[face]
                    if contact_front and wp.length(velocity)<=capture_speed:
                        state=0; candidate-=normal*(radius+1.e-5)
                    break
            advance=distance-radius
            if advance<=1.e-6: break
            progressed+=advance
            if progressed>=travel: clear=True; break
        if not clear and not hit: fallback[i]=1
        p=candidate
    proposed.position[i]=p; proposed.velocity[i]=velocity; proposed.normal[i]=normal
    proposed.face[i]=face; proposed.bary[i]=bary; proposed.state[i]=state
    proposed.age[i]=d.age[i]+dt


@wp.kernel
def queue_count(mask: wp.array(dtype=int), prefix: wp.array(dtype=int), count: wp.array(dtype=int)):
    last=mask.shape[0]-1; count[0]=mask[last]+prefix[last]


@wp.kernel
def compact_queue(mask: wp.array(dtype=int), prefix: wp.array(dtype=int), queue: wp.array(dtype=int)):
    i=wp.tid()
    if mask[i]==1: queue[prefix[i]]=i


@wp.kernel
def exact_contacts(d: ParticleArrays, proposed: ParticleArrays, queue: wp.array(dtype=int),
                   mesh: wp.uint64, adjacency: wp.array(dtype=int,ndim=2), islands: wp.array(dtype=int),
                   capture_distance: float, capture_speed: float, turn: float, dt: float,
                   unresolved: wp.array(dtype=int), diagnostics: wp.array(dtype=int)):
    index=wp.tid(); i=queue[index]
    p=d.position[i]; candidate=proposed.position[i]
    radius=wp.pow(d.volume[i]*.238732414637843,1./3.)
    diagnostics[i]=0
    if d.state[i]==0:
        face=proposed.face[i]; bary=proposed.bary[i]
        if face<0 or face>=adjacency.shape[0]:
            diagnostics[i]=1; wp.atomic_add(unresolved,0,1); return
        edge=wp.mesh_eval_position(mesh,face,bary[0],bary[1])
        remaining=candidate-edge; velocity=proposed.velocity[i]
        normal=proposed.normal[i]; state=int(0); limited=int(0); complete=bool(False)
        for extension in range(4):
            complete,edge,velocity,normal,face,bary,state,limited,remaining=walk_chart(
                mesh,adjacency,face,bary,edge,remaining,velocity,turn)
            if complete: break
        if complete:
            if state==1: edge+=normal*(radius+1.e-5)
            proposed.position[i]=edge; proposed.velocity[i]=velocity
            proposed.face[i]=face; proposed.bary[i]=bary; proposed.normal[i]=normal
            proposed.state[i]=state; proposed.limited[i]=limited
            proposed.island[i]=islands[face]; proposed.age[i]=d.age[i]+dt
            return
        candidate=edge+remaining
        proposed.face[i]=face
        query=wp.mesh_query_point_no_sign(mesh,candidate,capture_distance)
        allowed=bool(False)
        face=proposed.face[i]
        if query.result and face>=0 and face<adjacency.shape[0]:
            allowed=query.face==face
            for k in range(3):
                if adjacency[face,k]==query.face: allowed=True
        if allowed:
            proposed.position[i]=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
            proposed.face[i]=query.face; proposed.bary[i]=wp.vec2(query.u,query.v)
            proposed.normal[i]=wp.mesh_eval_face_normal(mesh,query.face)
            proposed.velocity[i]=tangent(proposed.velocity[i],proposed.normal[i])
            proposed.island[i]=islands[query.face]
            proposed.age[i]=d.age[i]+dt
        else:
            diagnostics[i]=query.face+2
            wp.atomic_add(unresolved,0,1)
    else:
        delta=candidate-p; length=wp.length(delta)
        ray=wp.mesh_query_ray(mesh,p,wp.vec3(0.,0.,1.),0.)
        if length>1.e-12: ray=wp.mesh_query_ray(mesh,p,delta/length,length+radius)
        nearest=wp.mesh_query_point_no_sign(mesh,candidate,radius+1.e-5)
        if ray.result or nearest.result:
            face=nearest.face; u=nearest.u; v=nearest.v
            if ray.result: face=ray.face; u=ray.u; v=ray.v
            q=wp.mesh_eval_position(mesh,face,u,v)
            normal=wp.mesh_eval_face_normal(mesh,face)
            front=wp.dot(p-q,normal)>=-1.e-8
            if not front: normal=-normal
            velocity=proposed.velocity[i]
            velocity-=normal*wp.min(0.,wp.dot(velocity,normal))
            proposed.position[i]=q+normal*(radius+1.e-5)
            proposed.velocity[i]=velocity; proposed.normal[i]=normal
            proposed.face[i]=face; proposed.bary[i]=wp.vec2(u,v); proposed.island[i]=islands[face]
            if front and wp.length(velocity)<=capture_speed:
                proposed.state[i]=0; proposed.position[i]=q
        proposed.age[i]=d.age[i]+dt


def advance_field_particles(pool, prepared, buffers, config, dt: float, *, reuse_deposit=False):
    if getattr(pool,'field_motion',None) is None:
        pool.field_motion=MotionBuffers(pool,prepared.chart)
    scratch=pool.field_motion; chart=prepared.chart; contact=prepared.contact
    for name in COPIED: wp.copy(getattr(scratch.proposed,name),getattr(pool.data,name))
    for name in SHARED: setattr(scratch.proposed,name,getattr(pool.data,name))
    scratch.proposed.limited.zero_()
    wp.copy(scratch.wet_backup,buffers.wetness)
    if not reuse_deposit: deposit_attached(pool,prepared,buffers)
    wp.copy(scratch.velocity_backup,buffers.velocity)
    try:
        step=evolve_field(prepared,buffers,config,dt)
        start=perf_counter(); scratch.unresolved.zero_()
        wp.launch(transport,pool.capacity,inputs=[pool.data,scratch.proposed,prepared.source.mesh.id,
            chart.source_adjacency_gpu,prepared.source.islands,chart.triangles_gpu,chart.level,buffers.velocity,
            wp.vec3(*contact.low),wp.vec3i(*contact.dimensions),contact.effective_spacing,contact.distance,
            contact.faces,contact.ambiguity,wp.vec3(*config.gravity),config.adhesion,config.capture_speed,
            cos(config.normal_turn_limit*pi/180.),dt,scratch.mask],device=pool.device)
        array_scan(scratch.mask,scratch.prefix,inclusive=False)
        wp.launch(queue_count,1,inputs=[scratch.mask,scratch.prefix,scratch.count],device=pool.device)
        count=int(scratch.count.numpy()[0]); scratch.last_fallback_count=count
        if count>len(scratch.queue):
            raise RuntimeError(f'Contact fallback needs {count} particles, exceeding {len(scratch.queue)}; refine source/contact resolution')
        if count:
            wp.launch(compact_queue,pool.capacity,inputs=[scratch.mask,scratch.prefix,scratch.queue],device=pool.device)
            wp.launch(exact_contacts,count,inputs=[pool.data,scratch.proposed,scratch.queue,prepared.source.mesh.id,
                chart.source_adjacency_gpu,prepared.source.islands,config.capture_distance,config.capture_speed,
                cos(config.normal_turn_limit*pi/180.),dt,scratch.unresolved,scratch.mask],device=pool.device)
        unresolved=int(scratch.unresolved.numpy()[0])
        if unresolved:
            diagnostics=scratch.mask.numpy()
            details=[]
            for slot in np.flatnonzero(diagnostics)[:4]:
                slot=int(slot)
                details.append((int(slot),int(pool.data.face[slot:slot+1].numpy()[0]),
                    int(scratch.proposed.face[slot:slot+1].numpy()[0]),int(diagnostics[slot])-2))
            raise RuntimeError(f'{unresolved} source-local contacts unresolved (slot, old/walk/query faces: {details}); particle interval was not committed')
        for name in COPIED: wp.copy(getattr(pool.data,name),getattr(scratch.proposed,name))
        from .emission import retire
        retire(pool,config)
        step.contact_ms=(perf_counter()-start)*1000
        pool.substeps=step.substeps
        pool.step_count.fill_(step.substeps)
        return step
    except Exception:
        wp.copy(buffers.wetness,scratch.wet_backup)
        wp.copy(buffers.velocity,scratch.velocity_backup)
        raise
