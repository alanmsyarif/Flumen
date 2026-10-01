"""Bounded stationary surface proxy and topological support neighborhoods."""
from dataclasses import dataclass
import heapq
import numpy as np
import warp as wp


@dataclass
class SurfaceTopology:
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    vertex_faces: np.ndarray
    triangle_faces: np.ndarray
    coarsening_factor: float
    effective_edge: float
    face_support_cpu: np.ndarray
    face_support: object
    points_gpu: object
    normals_gpu: object
    areas_gpu: object
    vertex_faces_gpu: object
    triangles_gpu: object
    proxy_grid: object
    support_overflow: int

    def allows_faces(self,a,b):
        return (0<=a<len(self.face_support_cpu) and 0<=b<len(self.face_support_cpu)
                and b in self.face_support_cpu[a] and a in self.face_support_cpu[b])

    def close(self):
        self.face_support = self.points_gpu = self.normals_gpu = self.areas_gpu = None
        self.vertex_faces_gpu = self.triangles_gpu = self.proxy_grid = None


def _support(vertices,triangles,islands,radius):
    count=len(triangles)
    adjacency=[set() for _ in range(count)]
    edges={}
    for face,tri in enumerate(triangles):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edge=tuple(sorted((int(a),int(b))))
            others=edges.setdefault(edge,[])
            for other in others:
                if islands[other]==islands[face]:
                    adjacency[face].add(other); adjacency[other].add(face)
            others.append(face)
    points=vertices[triangles]
    centers=points.mean(axis=1)
    reach=np.linalg.norm(points-centers[:,None,:],axis=2).max(axis=1)
    table=np.full((count,64),-1,dtype=np.int32)
    overflow=0
    for face in range(count):
        distances={face:0.}
        queue=[(0.,face)]; accepted=[]
        cutoff=radius+2*reach[face]
        while queue and len(accepted)<64:
            distance,current=heapq.heappop(queue)
            if distance!=distances[current]: continue
            accepted.append(current)
            for other in sorted(adjacency[current]):
                candidate=distance+float(np.linalg.norm(centers[current]-centers[other]))
                if candidate<=cutoff and candidate<distances.get(other,float('inf')):
                    distances[other]=candidate; heapq.heappush(queue,(candidate,other))
        if queue: overflow+=1
        table[face,:len(accepted)]=accepted
    return table,overflow


def build_topology(source,support_radius:float,target_edge:float,vertex_budget=100000)->SurfaceTopology:
    if (not np.isfinite([support_radius,target_edge]).all() or support_radius<=0 or target_edge<=0
            or isinstance(vertex_budget,bool) or not 3<=vertex_budget<=100000):
        raise ValueError('Invalid surface proxy radius, resolution or vertex budget')
    original=source.triangles_cpu
    used=np.unique(original)
    if len(used)>vertex_budget:
        raise ValueError('Collision surface exceeds proxy vertex budget; simplify the source')
    vertices=source.vertices_cpu[used].copy()
    triangles=np.searchsorted(used,original).astype(np.int32)
    face_ids=np.arange(len(triangles),dtype=np.int32)
    for _ in range(12):
        edge_pairs=np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))
        edges,inverse=np.unique(np.sort(edge_pairs,axis=1),axis=0,return_inverse=True)
        edge=float(np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1).max())
        if edge<=target_edge or len(vertices)+len(edges)>vertex_budget: break
        midpoints=len(vertices)+inverse.reshape(3,-1).T
        a,b,c=triangles.T; ab,bc,ca=midpoints.T
        triangles=np.stack((np.stack((a,ab,ca),axis=1),np.stack((ab,b,bc),axis=1),
                            np.stack((ca,bc,c),axis=1),np.stack((ab,bc,ca),axis=1)),axis=1).reshape(-1,3).astype(np.int32)
        vertices=np.concatenate((vertices,(vertices[edges[:,0]]+vertices[edges[:,1]])*.5))
        face_ids=np.repeat(face_ids,4)
    p=vertices[triangles]
    cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    face_area=np.linalg.norm(cross,axis=1)*.5
    areas=np.zeros(len(vertices),dtype=np.float32)
    normals=np.zeros_like(vertices)
    vertex_faces=np.full(len(vertices),len(original),dtype=np.int32)
    for corner in range(3):
        np.add.at(areas,triangles[:,corner],face_area/3)
        np.add.at(normals,triangles[:,corner],cross)
        np.minimum.at(vertex_faces,triangles[:,corner],face_ids)
    length=np.linalg.norm(normals,axis=1)
    if (length<1e-12).any() or (areas<=0).any():
        raise ValueError('Surface proxy has ambiguous normals or zero area')
    normals/=length[:,None]
    support,overflow=_support(source.vertices_cpu,original,source.islands_cpu,support_radius)
    device=source.device
    points_gpu=wp.array(vertices,dtype=wp.vec3,device=device)
    grid=wp.HashGrid(64,64,64,device=device)
    grid.build(points_gpu,support_radius)
    return SurfaceTopology(vertices,triangles,normals,areas,vertex_faces,face_ids,
        max(1.,edge/target_edge),edge,support,wp.array(support,dtype=int,device=device),
        points_gpu,wp.array(normals,dtype=wp.vec3,device=device),wp.array(areas,dtype=float,device=device),
        wp.array(vertex_faces,dtype=int,device=device),wp.array(triangles.flatten(),dtype=int,device=device),grid,overflow)


@wp.func
def compatible_faces(support:wp.array(dtype=int,ndim=2),a:int,b:int):
    forward=bool(False); backward=bool(False)
    if a>=0 and b>=0 and a<support.shape[0] and b<support.shape[0]:
        for k in range(64):
            if support[a,k]==b: forward=True
            if support[b,k]==a: backward=True
            if forward and backward: return True
    return forward and backward
