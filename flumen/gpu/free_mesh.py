"""Bounded sparse-brick CUDA reconstruction of current free particles."""
import numpy as np
import warp as wp
from warp.utils import array_scan, radix_sort_pairs
from .state import ParticleArrays, MeshBatch
from .surface_mesh import GeometryBuffers, emit_triangle, total_count, mesh_volume

MAX_BRICKS=16777
MAX_KEYS=262144
AXIAL_MAX=4.**(2./3.)
ISO=.3460693359375


def anisotropy_scales(speeds,radius):
    axial=np.clip(1.+np.asarray(speeds)*.02/radius,1.,AXIAL_MAX)
    transverse=1./np.sqrt(axial)
    return np.stack((axial,transverse,transverse),axis=-1)


class FreeMeshBuffers:
    def __init__(self,capacity,device,output=None):
        self.device=device; self.owns_output=output is None
        self.output=output or GeometryBuffers(1,device)
        self.low=wp.zeros(capacity,dtype=wp.vec3i,device=device)
        self.high=wp.zeros(capacity,dtype=wp.vec3i,device=device)
        self.key_counts=wp.zeros(capacity,dtype=wp.int64,device=device)
        self.key_prefix=wp.zeros(capacity,dtype=wp.int64,device=device)
        self.key_total=wp.zeros(1,dtype=wp.int64,device=device)
        self.keys=wp.zeros(2*MAX_KEYS,dtype=wp.int64,device=device)
        self.values=wp.zeros(2*MAX_KEYS,dtype=int,device=device)
        self.markers=wp.zeros(MAX_KEYS,dtype=int,device=device)
        self.run_prefix=wp.zeros(MAX_KEYS,dtype=int,device=device)
        self.bricks=wp.zeros(MAX_BRICKS,dtype=wp.int64,device=device)
        self.total=wp.zeros(1,dtype=int,device=device)
        self.domain_error=wp.zeros(1,dtype=int,device=device)
        self.summary=wp.zeros(2,dtype=wp.float64,device=device)
        self.field=wp.zeros(MAX_BRICKS*125,dtype=float,device=device)
        self.gradient=wp.zeros(MAX_BRICKS*125,dtype=wp.vec3,device=device)
        self.counts=wp.zeros(MAX_BRICKS*64,dtype=int,device=device)
        self.prefix=wp.zeros(MAX_BRICKS*64,dtype=int,device=device)
        self.tets=wp.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]],dtype=int,device=device)
        self.grid=wp.HashGrid(64,64,64,device=device)

    def close(self):
        if self.owns_output and self.output is not None: self.output.close()
        for name in tuple(self.__dict__):
            if name not in ('device','owns_output'): setattr(self,name,None)


@wp.func
def axial_scale(velocity:wp.vec3,radius:float):
    return wp.clamp(1.+wp.length(velocity)*.02/radius,1.,2.5198420997897464)


@wp.func
def velocity_axis(velocity:wp.vec3):
    axis=wp.vec3(0.,0.,1.)
    if wp.length(velocity)>1.e-8: axis=wp.normalize(velocity)
    return axis


@wp.kernel
def brick_bounds(d:ParticleArrays,pitch:float,nominal:float,low:wp.array(dtype=wp.vec3i),
                 high:wp.array(dtype=wp.vec3i),counts:wp.array(dtype=wp.int64),
                 error:wp.array(dtype=int),summary:wp.array(dtype=wp.float64)):
    i=wp.tid(); counts[i]=wp.int64(0)
    if d.active[i]!=1 or d.state[i]!=1: return
    radius=wp.pow(d.volume[i]*.238732414637843,1./3.)
    wp.atomic_add(summary,0,wp.float64(d.volume[i]))
    axial=axial_scale(d.velocity[i],nominal); transverse=1./wp.sqrt(axial)
    axis=velocity_axis(d.velocity[i])
    lo=wp.vec3i(0); hi=wp.vec3i(0)
    for k in range(3):
        extent=2.*radius*wp.sqrt(transverse*transverse+(axial*axial-transverse*transverse)*axis[k]*axis[k])
        a=wp.floor((d.position[i][k]-extent)/(4.*pitch))
        b=wp.floor((d.position[i][k]+extent)/(4.*pitch))
        if not wp.isfinite(a) or not wp.isfinite(b) or a < -1048576. or b>=1048576.:
            wp.atomic_max(error,0,1); return
        lo[k]=int(a); hi[k]=int(b)
    low[i]=lo; high[i]=hi
    counts[i]=wp.int64(hi[0]-lo[0]+1)*wp.int64(hi[1]-lo[1]+1)*wp.int64(hi[2]-lo[2]+1)
    wp.atomic_max(summary,1,wp.float64(2.*radius*axial))


@wp.kernel
def count64(counts:wp.array(dtype=wp.int64),prefix:wp.array(dtype=wp.int64),total:wp.array(dtype=wp.int64)):
    last=counts.shape[0]-1; total[0]=counts[last]+prefix[last]


@wp.func
def encode_brick(p:wp.vec3i):
    return wp.int64(p[0]+1048576)*wp.int64(4398046511104)+wp.int64(p[1]+1048576)*wp.int64(2097152)+wp.int64(p[2]+1048576)


@wp.func
def decode_brick(key:wp.int64):
    x=int(key//wp.int64(4398046511104))-1048576
    y=int((key//wp.int64(2097152))%wp.int64(2097152))-1048576
    z=int(key%wp.int64(2097152))-1048576
    return wp.vec3i(x,y,z)


@wp.kernel
def write_keys(low:wp.array(dtype=wp.vec3i),high:wp.array(dtype=wp.vec3i),counts:wp.array(dtype=wp.int64),
               prefix:wp.array(dtype=wp.int64),keys:wp.array(dtype=wp.int64),values:wp.array(dtype=int)):
    i=wp.tid()
    if counts[i]==wp.int64(0): return
    start=int(prefix[i]); cursor=int(0)
    for x in range(low[i][0],high[i][0]+1):
        for y in range(low[i][1],high[i][1]+1):
            for z in range(low[i][2],high[i][2]+1):
                keys[start+cursor]=encode_brick(wp.vec3i(x,y,z)); values[start+cursor]=i; cursor+=1


@wp.kernel
def mark_runs(keys:wp.array(dtype=wp.int64),markers:wp.array(dtype=int)):
    i=wp.tid(); value=int(0)
    if i==0 or keys[i]!=keys[i-1]: value=1
    markers[i]=value


@wp.kernel
def compact_runs(keys:wp.array(dtype=wp.int64),markers:wp.array(dtype=int),prefix:wp.array(dtype=int),
                 bricks:wp.array(dtype=wp.int64)):
    i=wp.tid()
    if markers[i]==1 and prefix[i]<16777: bricks[prefix[i]]=keys[i]


@wp.kernel
def sample_field(d:ParticleArrays,grid:wp.uint64,mesh:wp.uint64,bricks:wp.array(dtype=wp.int64),
                 pitch:float,nominal:float,reach:float,field:wp.array(dtype=float),gradient:wp.array(dtype=wp.vec3)):
    i=wp.tid(); base=decode_brick(bricks[i//125]); corner=i%125
    point=wp.vec3(float(4*base[0]+corner%5),float(4*base[1]+(corner//5)%5),float(4*base[2]+corner//25))*pitch
    value=float(0.); grad=wp.vec3(0.)
    nearest=wp.mesh_query_point_no_sign(mesh,point,.55*pitch)
    if not nearest.result:
        query=wp.hash_grid_query(grid,point,reach)
        for j in query:
            if d.active[j]!=1 or d.state[j]!=1: continue
            delta=point-d.position[j]
            radius=wp.pow(d.volume[j]*.238732414637843,1./3.)
            support=2.*radius
            axial=axial_scale(d.velocity[j],nominal); transverse=1./wp.sqrt(axial)
            axis=velocity_axis(d.velocity[j]); along=wp.dot(delta,axis)
            perp=delta-along*axis
            q2=wp.dot(perp,perp)/(support*support*transverse*transverse)+along*along/(support*support*axial*axial)
            if q2>=1.: continue
            distance=wp.length(delta)
            if distance>1.e-7:
                hit=wp.mesh_query_ray(mesh,d.position[j],delta/distance,wp.max(0.,distance-1.e-6))
                if hit.result: continue
            weight=d.volume[j]*315./(64.*wp.pi*support*support*support)
            remainder=1.-q2
            value+=weight*remainder*remainder*remainder
            grad-=6.*weight*remainder*remainder*(perp/(support*support*transverse*transverse)+along*axis/(support*support*axial*axial))
    field[i]=value; gradient[i]=-grad


@wp.func
def sample_index(cell:int,corner:int):
    local=cell%64
    x=local%4+corner%2; y=(local//4)%4+(corner//2)%2; z=local//16+corner//4
    return (cell//64)*125+x+5*y+25*z


@wp.func
def sample_position(cell:int,corner:int,bricks:wp.array(dtype=wp.int64),pitch:float):
    base=decode_brick(bricks[cell//64]); local=cell%64
    return wp.vec3(float(4*base[0]+local%4+corner%2),float(4*base[1]+(local//4)%4+(corner//2)%2),
                   float(4*base[2]+local//16+corner//4))*pitch


@wp.kernel
def count_cells(field:wp.array(dtype=float),tets:wp.array(dtype=int,ndim=2),counts:wp.array(dtype=int)):
    cell=wp.tid(); count=int(0)
    for tetra in range(6):
        inside=int(0)
        for k in range(4):
            if field[sample_index(cell,tets[tetra,k])]>=.3460693359375: inside+=1
        if inside==1 or inside==3: count+=1
        elif inside==2: count+=2
    counts[cell]=count


@wp.func
def intersection(cell:int,a:int,b:int,field:wp.array(dtype=float),bricks:wp.array(dtype=wp.int64),pitch:float):
    fa=field[sample_index(cell,a)]; fb=field[sample_index(cell,b)]
    t=(.3460693359375-fa)/(fb-fa)
    pa=sample_position(cell,a,bricks,pitch); pb=sample_position(cell,b,bricks,pitch)
    return pa+t*(pb-pa)


@wp.func
def intersection_normal(cell:int,a:int,b:int,field:wp.array(dtype=float),gradient:wp.array(dtype=wp.vec3)):
    ia=sample_index(cell,a); ib=sample_index(cell,b)
    t=(.3460693359375-field[ia])/(field[ib]-field[ia])
    return wp.normalize(gradient[ia]+t*(gradient[ib]-gradient[ia]))


@wp.func
def oriented_triangle(output:int,cell:int,a:wp.vec3i,b:wp.vec3i,direction:wp.vec3,
                      field:wp.array(dtype=float),gradient:wp.array(dtype=wp.vec3),bricks:wp.array(dtype=wp.int64),
                      pitch:float,vertices:wp.array(dtype=wp.vec3),normals:wp.array(dtype=wp.vec3),triangles:wp.array(dtype=wp.vec3i)):
    p0=intersection(cell,a[0],b[0],field,bricks,pitch)
    p1=intersection(cell,a[1],b[1],field,bricks,pitch)
    p2=intersection(cell,a[2],b[2],field,bricks,pitch)
    n0=intersection_normal(cell,a[0],b[0],field,gradient)
    n1=intersection_normal(cell,a[1],b[1],field,gradient)
    n2=intersection_normal(cell,a[2],b[2],field,gradient)
    outward=wp.normalize(wp.cross(p1-p0,p2-p0))
    if wp.dot(outward,direction)<0.:
        emit_triangle(output,p0,p2,p1,n0,n2,n1,vertices,normals,triangles)
    else:
        emit_triangle(output,p0,p1,p2,n0,n1,n2,vertices,normals,triangles)


@wp.kernel
def write_cells(field:wp.array(dtype=float),gradient:wp.array(dtype=wp.vec3),bricks:wp.array(dtype=wp.int64),
                tets:wp.array(dtype=int,ndim=2),prefix:wp.array(dtype=int),pitch:float,
                vertices:wp.array(dtype=wp.vec3),normals:wp.array(dtype=wp.vec3),triangles:wp.array(dtype=wp.vec3i)):
    cell=wp.tid(); output=prefix[cell]
    for tetra in range(6):
        high=wp.vec4i(0); low=wp.vec4i(0); nh=int(0); nl=int(0)
        ph=wp.vec3(0.); pl=wp.vec3(0.)
        for k in range(4):
            corner=tets[tetra,k]
            if field[sample_index(cell,corner)]>=.3460693359375:
                high[nh]=corner; nh+=1; ph+=sample_position(cell,corner,bricks,pitch)
            else:
                low[nl]=corner; nl+=1; pl+=sample_position(cell,corner,bricks,pitch)
        if nh==0 or nh==4: continue
        direction=pl/float(nl)-ph/float(nh)
        if nh==1:
            oriented_triangle(output,cell,wp.vec3i(high[0]),wp.vec3i(low[0],low[1],low[2]),direction,
                field,gradient,bricks,pitch,vertices,normals,triangles); output+=1
        elif nh==3:
            oriented_triangle(output,cell,wp.vec3i(high[0],high[1],high[2]),wp.vec3i(low[0]),direction,
                field,gradient,bricks,pitch,vertices,normals,triangles); output+=1
        else:
            oriented_triangle(output,cell,wp.vec3i(high[0],high[0],high[1]),wp.vec3i(low[0],low[1],low[1]),direction,
                field,gradient,bricks,pitch,vertices,normals,triangles); output+=1
            oriented_triangle(output,cell,wp.vec3i(high[0],high[1],high[1]),wp.vec3i(low[0],low[1],low[0]),direction,
                field,gradient,bricks,pitch,vertices,normals,triangles); output+=1


def build_free_mesh(pool,source,config,available_vertex_budget,available_triangle_budget,buffers)->MeshBatch:
    b=buffers; output=b.output
    vertex_budget=min(available_vertex_budget,output.vertex_budget)
    triangle_budget=min(available_triangle_budget,output.triangle_budget)
    diagnostics=dict(volumetric_samples=0,coarsening_factor=1.,sample_pitch=.5*config.radius*config.reconstruction_scale)
    for attempt in range(4):
        pitch=.5*config.radius*config.reconstruction_scale*2**attempt
        diagnostics.update(sample_pitch=pitch,coarsening_factor=float(2**attempt))
        b.domain_error.zero_(); b.summary.zero_()
        wp.launch(brick_bounds,pool.capacity,inputs=[pool.data,pitch,config.radius,b.low,b.high,b.key_counts,b.domain_error,b.summary],device=b.device)
        array_scan(b.key_counts,b.key_prefix,inclusive=False)
        wp.launch(count64,1,inputs=[b.key_counts,b.key_prefix,b.key_total],device=b.device)
        if int(b.domain_error.numpy()[0]):
            volume=float(b.summary.numpy()[0])
            return MeshBatch.empty(error='Free liquid brick key domain exceeded',represented_volume=volume,unrepresented_volume=volume,**diagnostics)
        requested=int(b.key_total.numpy()[0])
        if not requested: return MeshBatch.empty(**diagnostics)
        if requested>MAX_KEYS:
            diagnostics.update(requested_keys=requested); continue
        wp.launch(write_keys,pool.capacity,inputs=[b.low,b.high,b.key_counts,b.key_prefix,b.keys,b.values],device=b.device)
        radix_sort_pairs(b.keys,b.values,requested)
        markers=b.markers[:requested]; prefix=b.run_prefix[:requested]
        wp.launch(mark_runs,requested,inputs=[b.keys,markers],device=b.device)
        array_scan(markers,prefix,inclusive=False)
        wp.launch(total_count,1,inputs=[markers,prefix,b.total],device=b.device)
        count=int(b.total.numpy()[0])
        diagnostics.update(requested_bricks=count,volumetric_samples=min(count,MAX_BRICKS)*125)
        if count>MAX_BRICKS: continue
        wp.launch(compact_runs,requested,inputs=[b.keys,markers,prefix,b.bricks],device=b.device)
        summary=b.summary.numpy()
        reach=max(float(summary[1]),config.radius)
        b.grid.build(pool.data.position,reach)
        wp.launch(sample_field,count*125,inputs=[pool.data,b.grid.id,source.mesh.id,b.bricks,pitch,config.radius,reach,b.field,b.gradient],device=b.device)
        cells=count*64; counts=b.counts[:cells]; scan=b.prefix[:cells]
        wp.launch(count_cells,cells,inputs=[b.field,b.tets,counts],device=b.device)
        array_scan(counts,scan,inclusive=False)
        wp.launch(total_count,1,inputs=[counts,scan,b.total],device=b.device)
        triangles=int(b.total.numpy()[0])
        diagnostics.update(requested_vertices=3*triangles,requested_triangles=triangles)
        if triangles>triangle_budget or 3*triangles>vertex_budget: continue
        if not triangles:
            return MeshBatch.empty(error='Free liquid is below sampling resolution',represented_volume=float(summary[0]),unrepresented_volume=float(summary[0]),**diagnostics)
        wp.launch(write_cells,cells,inputs=[b.field,b.gradient,b.bricks,b.tets,scan,pitch,output.vertices,output.normals,output.triangles],device=b.device)
        mesh=output.read(triangles,**diagnostics)
        volume=mesh_volume(mesh); live=float(summary[0])
        mesh.diagnostics.update(mesh_volume=volume,represented_volume=live,
            rendered_volume_error=abs(volume-live)/max(live,1.e-20))
        return mesh
    volume=float(b.summary.numpy()[0])
    return MeshBatch.empty(error='Free liquid exceeds sampling/output budget after four resolutions',represented_volume=volume,unrepresented_volume=volume,**diagnostics)
