"""Persistent triangle BVHs and bounded, area-weighted CUDA source sampling."""
from dataclasses import dataclass
import numpy as np
import warp as wp


@wp.kernel
def sample_kernel(mesh: wp.uint64, cdf: wp.array(dtype=float), islands: wp.array(dtype=int),
                  ids: wp.array(dtype=wp.int64), seed: int, up: wp.vec3,
                  low: float, span: float, start: float, softness: float, attempts: int,
                  positions: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                  faces: wp.array(dtype=int), bary: wp.array(dtype=wp.vec2),
                  out_island: wp.array(dtype=int), valid: wp.array(dtype=int)):
    i = wp.tid()
    valid[i] = 0
    rng = wp.rand_init(seed, int(ids[i] % wp.int64(2147483647)))
    for attempt in range(attempts):
        x = wp.randf(rng)
        left = int(0)
        right = cdf.shape[0]-1
        while left < right:
            mid = (left+right)//2
            if x > cdf[mid]:
                left = mid+1
            else:
                right = mid
        face = left
        a = wp.sqrt(wp.randf(rng))
        u = 1.0-a
        v = a*wp.randf(rng)
        p = wp.mesh_eval_position(mesh, face, u, v)
        height = (wp.dot(p, up)-low)/wp.max(span, 1.e-8)
        mask = float(0.0)
        if softness < 1.e-8:
            if height > start:
                mask = 1.0
        else:
            t = wp.clamp((height-start+softness)/(2.0*softness),0.0,1.0)
            mask = t*t*t*(t*(t*6.0-15.0)+10.0)
        if span > 1.e-8 and wp.randf(rng) < mask:
            positions[i] = p
            normals[i] = wp.mesh_eval_face_normal(mesh, face)
            faces[i] = face
            bary[i] = wp.vec2(u,v)
            out_island[i] = islands[face]
            valid[i] = 1
            break


class Samples:
    def __init__(self, count, device):
        self.positions = wp.zeros(count, dtype=wp.vec3, device=device)
        self.normals = wp.zeros(count, dtype=wp.vec3, device=device)
        self.faces = wp.zeros(count, dtype=int, device=device)
        self.bary = wp.zeros(count, dtype=wp.vec2, device=device)
        self.islands = wp.zeros(count, dtype=int, device=device)
        self.valid = wp.zeros(count, dtype=int, device=device)


@dataclass
class SourceMesh:
    mesh: object
    island_meshes: list
    island_handles: object
    islands: object
    cdf: object
    up: tuple
    low: float
    span: float
    config: object
    device: str
    vertices_cpu: object
    triangles_cpu: object
    islands_cpu: object
    local_to_global: object
    island_offsets: object
    evaluated_triangle_ids: object

    def refresh_sampling(self, config):
        """Refresh physical source controls while retaining collision arrays/BVHs."""
        config.validate()
        up = -np.asarray(config.gravity, dtype=np.float64)
        up /= np.linalg.norm(up)
        heights = self.vertices_cpu @ up
        self.up, self.low, self.span = tuple(up), float(heights.min()), float(np.ptp(heights))
        self.config = config

    def close(self):
        self.mesh = None
        self.island_meshes.clear()
        self.island_handles = self.islands = self.cdf = None
        self.vertices_cpu = self.triangles_cpu = self.islands_cpu = None
        self.local_to_global = self.island_offsets = None
        self.evaluated_triangle_ids = None


def build_source(vertices, triangles, island_ids, config, device) -> SourceMesh:
    config.validate()
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1,3)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1,3)
    islands = np.asarray(island_ids, dtype=np.int32)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError('Collision mesh must contain finite vertices and faces')
    if triangles.min() < 0 or triangles.max() >= len(vertices) or len(islands) != len(triangles):
        raise ValueError('Collision mesh indices/islands are invalid')
    p = vertices[triangles]
    areas = np.linalg.norm(np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]),axis=1)*.5
    keep = areas > 1e-14
    triangles, areas, islands = triangles[keep], areas[keep], islands[keep]
    if not len(triangles):
        raise ValueError('Collision mesh has no nondegenerate faces')
    _, islands = np.unique(islands, return_inverse=True)
    alias = device.alias
    points = wp.array(vertices, dtype=wp.vec3, device=alias)
    mesh = wp.Mesh(points=points, indices=wp.array(triangles.flatten(),dtype=int,device=alias))
    island_meshes = []
    mappings=[]; offsets=[]
    for index in range(int(islands.max())+1):
        offsets.append(len(mappings))
        mappings.extend(np.flatnonzero(islands==index).tolist())
        subset = triangles[islands == index]
        island_meshes.append(mesh if len(subset)==len(triangles) else wp.Mesh(
            points=points, indices=wp.array(subset.flatten(),dtype=int,device=alias)))
    up = -np.array(config.gravity,dtype=np.float64)
    up /= np.linalg.norm(up)
    heights = vertices @ up
    cdf = np.cumsum(areas.astype(np.float64)); cdf /= cdf[-1]
    return SourceMesh(mesh, island_meshes,
        wp.array([m.id for m in island_meshes],dtype=wp.uint64,device=alias),
        wp.array(islands,dtype=int,device=alias), wp.array(cdf,dtype=float,device=alias),
        tuple(up), float(heights.min()), float(np.ptp(heights)), config, alias,
        vertices.copy(),triangles.copy(),islands.astype(np.int32),
        wp.array(mappings,dtype=int,device=alias),wp.array(offsets,dtype=int,device=alias),
        np.flatnonzero(keep).astype(np.int32))


def sample_source(source, seed, frame, candidate_ids, max_attempts=16, output=None, count=None):
    count = len(candidate_ids) if count is None else count
    if not 0 <= count <= min(len(candidate_ids),source.config.capacity) or not 1 <= max_attempts <= 16:
        raise ValueError('Sampling must stay within capacity and 16 attempts')
    out = output if output is not None else Samples(count, source.device)
    if count:
        cfg = source.config
        wp.launch(sample_kernel, count, inputs=[source.mesh.id, source.cdf, source.islands,
            candidate_ids, (seed+frame*1000003) % 2147483647, wp.vec3(*source.up), source.low,
            source.span, cfg.source_start, cfg.source_softness, max_attempts, out.positions,
            out.normals, out.faces, out.bary, out.islands, out.valid],device=source.device)
    return out
