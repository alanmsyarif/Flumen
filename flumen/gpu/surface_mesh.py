"""Count/scan/write of closed raised patches from the live thickness contour."""
import numpy as np
import warp as wp
from warp.utils import array_scan
from .state import MeshBatch


class GeometryBuffers:
    def __init__(self,face_capacity,device,vertex_budget=250000,triangle_budget=500000):
        if not 1<=vertex_budget<=250000 or not 1<=triangle_budget<=500000:
            raise ValueError('Invalid liquid output budget')
        self.device=device; self.vertex_budget=vertex_budget; self.triangle_budget=triangle_budget
        self.poly=wp.zeros((face_capacity,4),dtype=wp.vec3,device=device)
        self.normal=wp.zeros((face_capacity,4),dtype=wp.vec3,device=device)
        self.bary=wp.zeros((face_capacity,4),dtype=wp.vec3,device=device)
        self.height=wp.zeros((face_capacity,4),dtype=float,device=device)
        self.sides=wp.zeros((face_capacity,4),dtype=int,device=device)
        self.poly_count=wp.zeros(face_capacity,dtype=int,device=device)
        self.counts=wp.zeros(face_capacity,dtype=int,device=device)
        self.prefix=wp.zeros(face_capacity,dtype=int,device=device)
        self.total=wp.zeros(1,dtype=int,device=device)
        self.vertices=wp.zeros(vertex_budget,dtype=wp.vec3,device=device)
        self.normals=wp.zeros(vertex_budget,dtype=wp.vec3,device=device)
        self.triangles=wp.zeros(triangle_budget,dtype=wp.vec3i,device=device)
        self.host_vertices=wp.zeros(vertex_budget,dtype=wp.vec3,device='cpu',pinned=True)
        self.host_normals=wp.zeros(vertex_budget,dtype=wp.vec3,device='cpu',pinned=True)
        self.host_triangles=wp.zeros(triangle_budget,dtype=wp.vec3i,device='cpu',pinned=True)
        self.boundary=None; self.topology=None
        self.edge_vertices=self.face_edges=self.vertex_mask=self.vertex_prefix=self.vertex_total=None

    def prepare(self,topology):
        if self.topology is topology: return
        tri=topology.triangles
        edges=np.sort(np.stack((tri[:,[1,2]],tri[:,[2,0]],tri[:,[0,1]]),axis=1).reshape(-1,2),axis=1)
        unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
        self.boundary=wp.array((counts[inverse].reshape(-1,3)==1).astype(np.int32),dtype=int,device=self.device)
        self.edge_vertices=wp.array(unique.astype(np.int32),dtype=int,device=self.device)
        self.face_edges=wp.array(inverse.reshape(-1,3).astype(np.int32),dtype=int,device=self.device)
        self.vertex_mask=wp.zeros(len(topology.vertices)+len(unique),dtype=int,device=self.device)
        self.vertex_prefix=wp.zeros(self.vertex_mask.shape,dtype=int,device=self.device)
        self.vertex_total=wp.zeros(1,dtype=int,device=self.device)
        self.topology=topology

    def close(self):
        for name in ('poly','normal','bary','height','sides','poly_count','counts','prefix','total',
                     'vertices','normals','triangles','host_vertices','host_normals','host_triangles','boundary','topology',
                     'edge_vertices','face_edges','vertex_mask','vertex_prefix','vertex_total'):
            setattr(self,name,None)

    def read(self,count,vertex_count=None,**diagnostics):
        if not count: return MeshBatch.empty(**diagnostics)
        n=count*3 if vertex_count is None else vertex_count
        wp.copy(self.host_vertices,self.vertices,count=n)
        wp.copy(self.host_normals,self.normals,count=n)
        wp.copy(self.host_triangles,self.triangles,count=count)
        wp.synchronize_device(self.device)
        return MeshBatch(self.host_vertices.numpy()[:n].copy(),self.host_normals.numpy()[:n].copy(),
                         self.host_triangles.numpy()[:count].copy(),diagnostics)


@wp.kernel
def clip(points:wp.array(dtype=wp.vec3),normals:wp.array(dtype=wp.vec3),faces:wp.array(dtype=int),
         thickness:wp.array(dtype=float),boundary:wp.array(dtype=int,ndim=2),
         poly:wp.array(dtype=wp.vec3,ndim=2),normal:wp.array(dtype=wp.vec3,ndim=2),
         bary:wp.array(dtype=wp.vec3,ndim=2),height:wp.array(dtype=float,ndim=2),
         sides:wp.array(dtype=int,ndim=2),poly_count:wp.array(dtype=int),counts:wp.array(dtype=int)):
    i=wp.tid(); count=int(0)
    for edge in range(3):
        a=faces[3*i+edge]; b=faces[3*i+(edge+1)%3]
        ha=thickness[a]; hb=thickness[b]
        ba=wp.vec3(0.); bb=wp.vec3(0.)
        ba[edge]=1.; bb[(edge+1)%3]=1.
        if ha>=1.e-5:
            poly[i,count]=points[a]; normal[i,count]=normals[a]
            bary[i,count]=ba; height[i,count]=ha; count+=1
        if (ha>=1.e-5)!=(hb>=1.e-5):
            t=(1.e-5-ha)/(hb-ha)
            if t>1.e-7 and t<1.-1.e-7:
                poly[i,count]=points[a]+t*(points[b]-points[a])
                normal[i,count]=wp.normalize(normals[a]+t*(normals[b]-normals[a]))
                bary[i,count]=ba+t*(bb-ba); height[i,count]=1.e-5; count+=1
    poly_count[i]=count
    amount=int(0)
    if count>=3:
        amount=2*(count-2)
        for edge in range(4):
            side=int(0)
            if edge<count:
                next=(edge+1)%count
                if wp.abs(height[i,edge]-1.e-5)<1.e-9 and wp.abs(height[i,next]-1.e-5)<1.e-9:
                    side=1
                for corner in range(3):
                    if wp.abs(bary[i,edge][corner])<1.e-6 and wp.abs(bary[i,next][corner])<1.e-6 and boundary[i,corner]==1:
                        side=1
            sides[i,edge]=side
            amount+=2*side
    counts[i]=amount


@wp.kernel
def total_count(counts:wp.array(dtype=int),prefix:wp.array(dtype=int),total:wp.array(dtype=int)):
    n=counts.shape[0]-1
    total[0]=prefix[n]+counts[n]


@wp.func
def emit_triangle(index:int,p0:wp.vec3,p1:wp.vec3,p2:wp.vec3,
                  n0:wp.vec3,n1:wp.vec3,n2:wp.vec3,
                  vertices:wp.array(dtype=wp.vec3),normals:wp.array(dtype=wp.vec3),triangles:wp.array(dtype=wp.vec3i)):
    start=3*index
    vertices[start]=p0; vertices[start+1]=p1; vertices[start+2]=p2
    normals[start]=n0; normals[start+1]=n1; normals[start+2]=n2
    triangles[index]=wp.vec3i(start,start+1,start+2)


@wp.kernel
def mark_vertices(thickness:wp.array(dtype=float),edges:wp.array(dtype=int,ndim=2),mask:wp.array(dtype=int)):
    i=wp.tid(); n=thickness.shape[0]; active=int(0)
    if i<n:
        if thickness[i]>=1.e-5: active=1
    else:
        a=thickness[edges[i-n,0]]; b=thickness[edges[i-n,1]]
        if (a>1.e-5 and b<1.e-5) or (b>1.e-5 and a<1.e-5): active=1
    mask[i]=active


@wp.kernel
def write_vertices(points:wp.array(dtype=wp.vec3),source_normals:wp.array(dtype=wp.vec3),thickness:wp.array(dtype=float),
                   edges:wp.array(dtype=int,ndim=2),mask:wp.array(dtype=int),prefix:wp.array(dtype=int),
                   vertices:wp.array(dtype=wp.vec3),normals:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    if mask[i]==0: return
    n=thickness.shape[0]; point=wp.vec3(0.); normal=wp.vec3(0.); height=float(1.e-5)
    if i<n:
        point=points[i]; normal=source_normals[i]; height=thickness[i]
    else:
        a=edges[i-n,0]; b=edges[i-n,1]
        t=(1.e-5-thickness[a])/(thickness[b]-thickness[a])
        point=points[a]+t*(points[b]-points[a]); normal=wp.normalize(source_normals[a]+t*(source_normals[b]-source_normals[a]))
    output=2*prefix[i]
    vertices[output]=point; vertices[output+1]=point+normal*height
    normals[output]=-normal; normals[output+1]=normal


@wp.kernel
def write_shared_triangles(bary:wp.array(dtype=wp.vec3,ndim=2),sides:wp.array(dtype=int,ndim=2),
               faces:wp.array(dtype=int),edges:wp.array(dtype=int,ndim=2),n:int,
               poly_count:wp.array(dtype=int),face_prefix:wp.array(dtype=int),vertex_prefix:wp.array(dtype=int),
               triangles:wp.array(dtype=wp.vec3i)):
    i=wp.tid(); count=poly_count[i]; output=face_prefix[i]
    if count<3: return
    ids=wp.vec4i(0)
    for p in range(4):
        if p<count:
            candidate=int(-1)
            for k in range(3):
                if bary[i,p][k]==1.: candidate=faces[3*i+k]
            if candidate<0:
                for k in range(3):
                    if wp.abs(bary[i,p][k])<1.e-7: candidate=n+edges[i,k]
            ids[p]=2*vertex_prefix[candidate]
    for fan in range(2):
        if fan<count-2:
            a=ids[0]; b=ids[fan+1]; c=ids[fan+2]
            triangles[output]=wp.vec3i(a+1,b+1,c+1); output+=1
            triangles[output]=wp.vec3i(a,c,b); output+=1
    for edge in range(4):
        if edge<count and sides[i,edge]==1:
            a=ids[edge]; b=ids[(edge+1)%count]
            triangles[output]=wp.vec3i(a,b,b+1); output+=1
            triangles[output]=wp.vec3i(a,b+1,a+1); output+=1


def mesh_volume(mesh):
    if not len(mesh.triangles): return 0.
    p=mesh.vertices.astype(np.float64)[mesh.triangles]
    return float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6.)


def build_attached_mesh(topology,surface,config,buffers)->MeshBatch:
    buffers.prepare(topology)
    wp.launch(clip,len(topology.triangles),inputs=[topology.points_gpu,topology.normals_gpu,topology.triangles_gpu,
        surface.thickness,buffers.boundary,buffers.poly,buffers.normal,buffers.bary,buffers.height,
        buffers.sides,buffers.poly_count,buffers.counts],device=buffers.device)
    array_scan(buffers.counts,buffers.prefix,inclusive=False)
    wp.launch(total_count,1,inputs=[buffers.counts,buffers.prefix,buffers.total],device=buffers.device)
    count=int(buffers.total.numpy()[0])
    wp.launch(mark_vertices,len(buffers.vertex_mask),inputs=[surface.thickness,buffers.edge_vertices,buffers.vertex_mask],device=buffers.device)
    array_scan(buffers.vertex_mask,buffers.vertex_prefix,inclusive=False)
    wp.launch(total_count,1,inputs=[buffers.vertex_mask,buffers.vertex_prefix,buffers.vertex_total],device=buffers.device)
    vertex_count=2*int(buffers.vertex_total.numpy()[0])
    thickness_volume=float(np.dot(surface.thickness.numpy().astype(np.float64),topology.areas.astype(np.float64)))
    if count>buffers.triangle_budget or vertex_count>buffers.vertex_budget:
        return MeshBatch.empty(error='Attached liquid exceeds output budget',requested_vertices=vertex_count,requested_triangles=count,
            represented_volume=thickness_volume,unrepresented_volume=thickness_volume,rendered_volume_error=1.)
    wp.launch(write_vertices,len(buffers.vertex_mask),inputs=[topology.points_gpu,topology.normals_gpu,surface.thickness,
        buffers.edge_vertices,buffers.vertex_mask,buffers.vertex_prefix,buffers.vertices,buffers.normals],device=buffers.device)
    wp.launch(write_shared_triangles,len(topology.triangles),inputs=[buffers.bary,buffers.sides,topology.triangles_gpu,
        buffers.face_edges,len(topology.vertices),buffers.poly_count,buffers.prefix,buffers.vertex_prefix,buffers.triangles],device=buffers.device)
    mesh=buffers.read(count,vertex_count=vertex_count)
    volume=mesh_volume(mesh)
    mesh.diagnostics.update(represented_volume=thickness_volume,mesh_volume=volume,
        rendered_volume_error=abs(volume-thickness_volume)/max(thickness_volume,1.e-20),
        excluded_volume=max(0.,thickness_volume-volume),coarsening_factor=topology.coarsening_factor)
    return mesh
