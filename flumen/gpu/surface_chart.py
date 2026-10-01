"""Stationary source-local charts with exact barycentric refinement lookup."""
from dataclasses import dataclass
import numpy as np
import warp as wp


@wp.func
def chart_anchor(face: int, bary: wp.vec2, level: int):
    weights = wp.vec3(bary[0], bary[1], 1.0-bary[0]-bary[1])
    child = face
    for depth in range(level):
        corner = int(3)
        if weights[0] >= .5:
            corner = 0
            weights = wp.vec3(2.*weights[0]-1., 2.*weights[1], 2.*weights[2])
        elif weights[1] >= .5:
            corner = 1
            weights = wp.vec3(2.*weights[0], 2.*weights[1]-1., 2.*weights[2])
        elif weights[2] >= .5:
            corner = 2
            weights = wp.vec3(2.*weights[0], 2.*weights[1], 2.*weights[2]-1.)
        else:
            weights = wp.vec3(1.-2.*weights[2], 1.-2.*weights[0], 1.-2.*weights[1])
        child = child*4+corner
    weights = wp.vec3(wp.max(0.,weights[0]), wp.max(0.,weights[1]), wp.max(0.,weights[2]))
    weights /= weights[0]+weights[1]+weights[2]
    return child, weights


@dataclass
class SurfaceChart:
    vertices: np.ndarray
    triangles: np.ndarray
    areas: np.ndarray
    normals: np.ndarray
    original_faces: np.ndarray
    source_adjacency: np.ndarray
    level: int
    effective_spacing: float
    coarsening_factor: float
    barrier_count: int
    min_edge: float
    operator_spacing: float
    points_gpu: object
    triangles_gpu: object
    areas_gpu: object
    normals_gpu: object
    source_adjacency_gpu: object
    edges_gpu: object
    weights_gpu: object
    gradients_gpu: object
    triangle_areas_gpu: object
    offsets_gpu: object
    neighbors_gpu: object
    neighbor_weights_gpu: object

    def close(self):
        for name in tuple(self.__dict__):
            if name.endswith('_gpu'): setattr(self, name, None)


def build_chart(source, spacing: float, node_budget: int = 100000) -> SurfaceChart:
    if not np.isfinite(spacing) or spacing <= 0 or isinstance(node_budget,bool) or not 3 <= node_budget <= 100000:
        raise ValueError('Invalid field spacing or node budget')
    source_faces = source.triangles_cpu
    points = source.vertices_cpu[source_faces].astype(np.float64)
    crosses = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
    face_normals = crosses/np.linalg.norm(crosses,axis=1)[:,None]
    parents = np.arange(source_faces.size)
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]; index = parents[index]
        return int(index)
    adjacency = np.full((len(source_faces),3),-1,np.int32)
    edge_faces = {}
    for face, tri in enumerate(source_faces):
        for corner in range(3):
            edge = tuple(sorted((int(tri[(corner+1)%3]), int(tri[(corner+2)%3]))))
            edge_faces.setdefault(edge,[]).append((face,corner))
    barriers = 0
    for edge, incident in edge_faces.items():
        if len(incident) != 2:
            barriers += 1; continue
        (a,ka),(b,kb) = incident
        if source.islands_cpu[a] != source.islands_cpu[b] or face_normals[a]@face_normals[b] < .5:
            barriers += 1; continue
        adjacency[a,ka], adjacency[b,kb] = b,a
        for vertex in edge:
            ia = a*3+int(np.flatnonzero(source_faces[a]==vertex)[0])
            ib = b*3+int(np.flatnonzero(source_faces[b]==vertex)[0])
            parents[root(ib)] = root(ia)
    roots = np.array([root(i) for i in range(len(parents))])
    unique, inverse = np.unique(roots,return_inverse=True)
    vertices = source.vertices_cpu[source_faces.flatten()[unique]].copy()
    triangles = inverse.reshape(-1,3).astype(np.int32)
    if len(vertices)>node_budget:
        raise ValueError('Source chart exceeds node budget; simplify collision geometry')
    provenance = np.arange(len(triangles),dtype=np.int32)
    level = 0
    for _ in range(12):
        raw = np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))
        edges,inverse = np.unique(np.sort(raw,axis=1),axis=0,return_inverse=True)
        lengths = np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
        effective = float(lengths.max())
        if effective <= spacing or len(vertices)+len(edges)>node_budget: break
        ab,bc,ca = (len(vertices)+inverse.reshape(3,-1).T).T
        a,b,c = triangles.T
        triangles = np.stack((np.stack((a,ab,ca),1),np.stack((ab,b,bc),1),
                              np.stack((ca,bc,c),1),np.stack((ab,bc,ca),1)),1).reshape(-1,3).astype(np.int32)
        vertices = np.concatenate((vertices,(vertices[edges[:,0]]+vertices[edges[:,1]])*.5))
        provenance = np.repeat(provenance,4); level += 1
    p = vertices[triangles].astype(np.float64)
    cross = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    twice_area = np.linalg.norm(cross,axis=1)
    triangle_areas = twice_area*.5
    normals = np.zeros((len(vertices),3),np.float64); areas = np.zeros(len(vertices),np.float64)
    for corner in range(3):
        np.add.at(normals,triangles[:,corner],cross)
        np.add.at(areas,triangles[:,corner],triangle_areas/3.)
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    unit = cross/twice_area[:,None]
    gradients = np.stack((np.cross(unit,p[:,2]-p[:,1]),np.cross(unit,p[:,0]-p[:,2]),
                          np.cross(unit,p[:,1]-p[:,0])),axis=1)/twice_area[:,None,None]
    # Source triangles retain exact anchors; numerical derivatives need not resolve
    # source slivers below the requested field scale. One common triangle factor
    # preserves the zero gradient of a constant scalar.
    operator_spacing=max(float(lengths.min()),.5*spacing)
    gradient_scale=np.minimum(1.,2./(operator_spacing*np.linalg.norm(gradients,axis=2).max(axis=1)))
    gradients*=gradient_scale[:,None,None]
    raw = np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))
    edges,inverse = np.unique(np.sort(raw,axis=1),axis=0,return_inverse=True)
    length_sq = np.sum((vertices[edges[:,0]]-vertices[edges[:,1]])**2,axis=1)
    weights = np.zeros(len(edges),np.float64)
    np.add.at(weights,inverse,np.tile(triangle_areas/3,3)/np.maximum(length_sq[inverse],operator_spacing**2))
    rows = [[] for _ in vertices]
    for (a,b),weight in zip(edges,weights):
        rows[a].append((int(b),weight)); rows[b].append((int(a),weight))
    offsets = np.zeros(len(vertices)+1,np.int32)
    offsets[1:] = np.cumsum([len(row) for row in rows])
    neighbors = np.array([j for row in rows for j,w in sorted(row)],np.int32)
    neighbor_weights = np.array([w for row in rows for j,w in sorted(row)],np.float32)
    device = source.device
    return SurfaceChart(vertices,triangles,areas,normals.astype(np.float32),provenance,adjacency,
        level,effective,max(1.,effective/spacing),barriers,float(np.sqrt(length_sq.min())),operator_spacing,
        wp.array(vertices,dtype=wp.vec3,device=device),wp.array(triangles,dtype=wp.vec3i,device=device),
        wp.array(areas,dtype=wp.float64,device=device),wp.array(normals,dtype=wp.vec3,device=device),
        wp.array(adjacency,dtype=int,device=device),wp.array(edges,dtype=wp.vec2i,device=device),
        wp.array(weights,dtype=float,device=device),wp.array(gradients.reshape(-1,3),dtype=wp.vec3,device=device),
        wp.array(triangle_areas,dtype=float,device=device),wp.array(offsets,dtype=int,device=device),
        wp.array(neighbors,dtype=int,device=device),wp.array(neighbor_weights,dtype=float,device=device))
