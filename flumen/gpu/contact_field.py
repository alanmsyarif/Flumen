"""Bounded unsigned-distance contact samples with conservative error/provenance."""
from dataclasses import dataclass
import numpy as np
import warp as wp


@wp.kernel
def prepare_contacts(mesh: wp.uint64, triangles: wp.array(dtype=int,ndim=2),
                     islands: wp.array(dtype=int), low: wp.vec3, dimensions: wp.vec3i,
                     spacing: float, distances: wp.array(dtype=float), faces: wp.array(dtype=int),
                     normals: wp.array(dtype=wp.vec3), sample_islands: wp.array(dtype=int),
                     ambiguity: wp.array(dtype=int)):
    i = wp.tid()
    x = i % dimensions[0]
    y = (i//dimensions[0]) % dimensions[1]
    z = i//(dimensions[0]*dimensions[1])
    point = low+wp.vec3(float(x),float(y),float(z))*spacing
    q = wp.mesh_query_point_no_sign(mesh,point,1.e10)
    faces[i] = -1
    distances[i] = 1.e10
    if q.result:
        p = wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        distances[i] = wp.length(point-p)
        faces[i] = q.face
        normals[i] = wp.mesh_eval_face_normal(mesh,q.face)
        sample_islands[i] = islands[q.face]
        extent = wp.vec3(1.7320508075688772*spacing)
        query = wp.mesh_query_aabb(mesh,point-extent,point+extent)
        for other in query:
            if other != q.face:
                shared = bool(False)
                for a in range(3):
                    for b in range(3):
                        if triangles[q.face,a] == triangles[other,b]: shared = True
                if not shared or islands[other] != islands[q.face]:
                    ambiguity[i] = 1


@dataclass
class ContactField:
    low: np.ndarray
    dimensions: tuple
    effective_spacing: float
    coarsening_factor: float
    error_bound: float
    sample_count: int
    ambiguous_count: int
    distance: object
    faces: object
    normals: object
    islands: object
    ambiguity: object

    def close(self):
        self.distance=self.faces=self.normals=self.islands=self.ambiguity=None


def build_contact_field(source, spacing: float, sample_budget: int = 2097152) -> ContactField:
    if not np.isfinite(spacing) or spacing<=0 or isinstance(sample_budget,bool) or not 8 <= sample_budget <= 2097152:
        raise ValueError('Invalid contact spacing or sample budget')
    effective = spacing
    vertices = source.vertices_cpu.astype(np.float64)
    for _ in range(64):
        pad = max(effective*2,source.config.capture_distance,source.config.radius*source.config.maximum_merged_radius_scale*2)
        low = vertices.min(axis=0)-pad
        span = vertices.max(axis=0)+pad-low
        dimensions = np.maximum(2,np.ceil(span/effective).astype(np.int64)+1)
        count = int(np.prod(dimensions,dtype=np.int64))
        if count <= sample_budget: break
        effective *= max(1.05,(count/sample_budget)**(1/3))
    else:
        raise ValueError('Cannot fit contact domain within sample budget')
    device = source.device
    result = ContactField(low.astype(np.float32),tuple(int(v) for v in dimensions),effective,
                          effective/spacing,np.sqrt(3)*effective,count,0,None,None,None,None,None)
    try:
        result.distance = wp.empty(count,dtype=float,device=device)
        result.faces = wp.empty(count,dtype=int,device=device)
        result.normals = wp.zeros(count,dtype=wp.vec3,device=device)
        result.islands = wp.full(count,-1,dtype=int,device=device)
        result.ambiguity = wp.zeros(count,dtype=int,device=device)
        wp.launch(prepare_contacts,count,inputs=[source.mesh.id,
            wp.array(source.triangles_cpu,dtype=int,device=device),source.islands,
            wp.vec3(*result.low),wp.vec3i(*result.dimensions),effective,
            result.distance,result.faces,result.normals,result.islands,result.ambiguity],device=device)
        result.ambiguous_count = int(np.count_nonzero(result.ambiguity.numpy()))
        return result
    except Exception:
        result.close(); raise
