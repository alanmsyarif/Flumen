"""CPU/NumPy offline meshing of immutable particle caches. Needs no CUDA, Warp or Blender.

Attached film: source triangles are refined into a shared lattice at the mesh spacing;
attached particles deposit their exact volume onto lattice nodes through their cached
(face, bary) anchors; a closed film shell (base, offset top, cut and boundary walls) is
built over nodes thicker than `min_thickness`. Separate sheets share no nodes, so films
never bridge between them.

Free drops: an anisotropic poly6 field (the live CUDA mesher's math) is sampled on a
global grid in bounded tiles of `tile_cells`^3 owned cells plus a one-cell halo, then
contoured with marching tetrahedra. Vertices are keyed by global grid edges, so tile
seams share vertices and leave no cracks. Grid samples within 0.55 pitch of the source
surface are cleared, so drops do not bridge through a contact sheet (the live mesher
also ray-tests each particle-sample pair; this CPU path does not).
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
from io import BytesIO
from math import ceil, isfinite, log2
from pathlib import Path
from time import perf_counter
import json
import os
import numpy as np
from .water_types import MeshBatch, WaterGeometry

ISO = .3460693359375
AXIAL_MAX = 4.**(2./3.)
TETS = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
CORNERS = np.array([[q % 2, (q//2) % 2, q//4] for q in range(8)])


@dataclass(frozen=True)
class MeshOptions:
    spacing: float                 # free-drop grid pitch and film lattice edge target, meters
    film_spacing: float = 0.       # 0 uses `spacing`
    tile_cells: int = 32
    min_thickness: float = 1e-5
    max_triangles: int = 20_000_000
    film_smoothing: int = 0        # volume-conserving diffusion steps on deposited film thickness
    film_max_thickness: float = 0. # 0: off. Water above it at a node becomes a pendant drop (volume moves, not lost)
    film_sheen: float = 0.         # cosmetic: minimum film thickness x cached wetness where water has passed (reported)
    free_crop: tuple = None        # ((x, y, z) low, (x, y, z) high): free drops outside are skipped and reported
    drop_kernel: str = 'velocity'  # 'velocity': stretch along motion; 'pca': Yu & Turk neighbour anisotropy

    def __post_init__(self):
        for value in (self.spacing, self.film_spacing, self.min_thickness, self.film_max_thickness, self.film_sheen):
            if not isfinite(value) or value < 0: raise ValueError('Mesh distances must be finite and nonnegative')
        if self.spacing <= 0: raise ValueError('Mesh spacing must be positive')
        if not 1 <= self.tile_cells <= 32: raise ValueError('Tiles hold 1 to 32 cells per axis')
        if self.max_triangles < 1: raise ValueError('Triangle budget must be positive')
        if isinstance(self.film_smoothing, bool) or not isinstance(self.film_smoothing, int) or self.film_smoothing < 0:
            raise ValueError('Film smoothing must be a nonnegative integer')
        if self.drop_kernel not in ('velocity', 'pca'): raise ValueError("Drop kernel must be 'velocity' or 'pca'")
        if self.free_crop is not None:
            low, high = (np.asarray(corner, float) for corner in self.free_crop)
            if low.shape != (3,) or high.shape != (3,) or not (np.isfinite(low).all() and np.isfinite(high).all() and (low < high).all()):
                raise ValueError('Free crop needs finite low < high corners')


def chart_anchor(faces, bary, level):
    """Vectorized CPU copy of gpu.surface_chart.chart_anchor."""
    w = np.stack([bary[:, 0], bary[:, 1], 1.-bary[:, 0]-bary[:, 1]], 1).astype(np.float64)
    child = np.asarray(faces, np.int64).copy()
    for _ in range(level):
        c0 = w[:, 0] >= .5; c1 = ~c0 & (w[:, 1] >= .5); c2 = ~c0 & ~c1 & (w[:, 2] >= .5); c3 = ~(c0 | c1 | c2)
        corner = np.select([c0, c1, c2], [0, 1, 2], 3)
        new = 2.*w; new[c0, 0] -= 1.; new[c1, 1] -= 1.; new[c2, 2] -= 1.
        new[c3] = np.stack([1.-2.*w[c3, 2], 1.-2.*w[c3, 0], 1.-2.*w[c3, 1]], 1)
        w = new; child = child*4+corner
    w = np.maximum(w, 0.); w /= w.sum(1, keepdims=True)
    return child, w


def wet_corners(static, wetness):
    """Wetness at every source triangle corner through the cached chart's face/bary provenance."""
    count = len(static['source_triangles'])
    faces = np.repeat(np.arange(count), 3)
    bary = np.tile(np.array([[1., 0.], [0., 1.], [0., 0.]]), (count, 1))
    child, weights = chart_anchor(faces, bary, int(static['chart_level'][0]))
    nodes = static['chart_triangles'][child]
    return (weights*wetness[nodes]).sum(1).astype(np.float32).reshape(count, 3)


# ---------------------------------------------------------------- attached film

def _lattice(n):
    li, lj = np.array([(i, j) for i in range(n+1) for j in range(n+1-i)]).T
    local = -np.ones((n+1, n+1), np.int64); local[li, lj] = np.arange(len(li))
    up = [(local[i, j], local[i+1, j], local[i, j+1]) for i in range(n) for j in range(n-i)]
    down = [(local[i+1, j], local[i+1, j+1], local[i, j+1]) for i in range(n-1) for j in range(n-1-i)]
    return li, lj, local, np.array(up+down, np.int64)


def film_lattice(static, options):
    """Refined source lattice: depends only on static source data and film spacing, so a
    sequence builds it once and reuses it for every frame."""
    V = static['source_vertices'].astype(np.float64); T = static['source_triangles'].astype(np.int64); F = len(T)
    spacing = options.film_spacing or options.spacing
    lengths = np.linalg.norm(V[T]-V[np.roll(T, -1, axis=1)], axis=2)
    # One conforming subdivision for all faces (no T-junction cracks), sized by typical edges:
    # the longest edges get a coarser film, reported as film_max_spacing.
    n = int(min(256, max(1, ceil(float(np.percentile(lengths, 95))/spacing))))
    if 2*F*n*n > options.max_triangles:
        raise ValueError(f'Film lattice exceeds the triangle budget ({2*F*n*n:,} > {options.max_triangles:,})')
    li, lj, local, small = _lattice(n); m = len(li)
    face = np.repeat(np.arange(F), m); i = np.tile(li, F); j = np.tile(lj, F); k = n-i-j
    A, B, C = T[face, 0], T[face, 1], T[face, 2]
    # Global node keys: source vertex, source edge + step from the lower vertex id, or face interior.
    key = np.zeros((F*m, 4), np.int64)
    is_a, is_b, is_c = i == n, j == n, (i == 0) & (j == 0)
    corner = is_a | is_b | is_c
    key[corner, 0] = 0; key[corner, 1] = np.select([is_a, is_b], [A, B], C)[corner]
    for mask, va, vb, steps in ((~corner & (k == 0), A, B, j), (~corner & (i == 0), B, C, n-j), (~corner & (j == 0), C, A, i)):
        low = np.minimum(va, vb)
        key[mask] = np.stack([np.ones_like(low), low, np.maximum(va, vb), np.where(va == low, steps, n-steps)], 1)[mask]
        corner = corner | mask
    interior = ~corner
    key[interior] = np.stack([np.full(F*m, 2), face, i, j], 1)[interior]
    _, node = np.unique(key, axis=0, return_inverse=True); node = node.reshape(F, m)
    count = int(node.max())+1
    positions = np.zeros((count, 3))
    positions[node.ravel()] = (V[C]+(i/n)[:, None]*(V[A]-V[C])+(j/n)[:, None]*(V[B]-V[C]))
    # Each node's anchor in the cached chart, for sampling per-frame wetness.
    node_face = np.empty(count, np.int64); node_bary = np.empty((count, 2))
    node_face[node.ravel()] = face; node_bary[node.ravel()] = np.stack([i/n, j/n], 1)
    child, wet_weights = chart_anchor(node_face, node_bary, int(static['chart_level'][0]))
    wet_nodes = static['chart_triangles'][child]
    tris = node[:, small].reshape(-1, 3)
    cross = np.cross(V[T[:, 0]]-V[T[:, 2]], V[T[:, 1]]-V[T[:, 2]])
    small_area = np.repeat(np.linalg.norm(cross, axis=1)*.5/(n*n), len(small))
    normals = np.zeros((count, 3)); node_area = np.zeros(count)
    np.add.at(normals, tris.ravel(), np.repeat(np.repeat(cross, len(small), axis=0), 3, axis=0))
    np.add.at(node_area, tris.ravel(), np.repeat(small_area/3., 3))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    return dict(n=n, F=F, local=local, node=node, count=count, positions=positions, normals=normals, edges=edges,
                wet_nodes=wet_nodes, wet_weights=wet_weights,
                node_area=node_area, tris=tris, max_spacing=float(lengths.max())/n, film_spacing=spacing)


def _film(static, cached, options, lattice=None):
    lattice = lattice or film_lattice(static, options)
    if lattice['film_spacing'] != (options.film_spacing or options.spacing): raise ValueError('Lattice spacing mismatch')
    n, F, local, node, count = lattice['n'], lattice['F'], lattice['local'], lattice['node'], lattice['count']
    # Conservative deposit of attached particles onto lattice nodes.
    arrays = cached.arrays
    attached = (arrays['state'] == 0) & (arrays['face'] >= 0) & (arrays['face'] < F)
    volume = np.zeros(count)
    if attached.any():
        f = arrays['face'][attached].astype(np.int64); bary = arrays['bary'][attached].astype(np.float64)
        a, b = bary[:, 0]*n, bary[:, 1]*n
        i0 = np.clip(np.floor(a), 0, n-1).astype(np.int64); j0 = np.clip(np.floor(b), 0, n-1).astype(np.int64)
        j0 = np.minimum(j0, n-1-i0)
        fa, fb = a-i0, b-j0
        down = (fa+fb > 1.) & (i0+j0 <= n-2)
        w = np.where(down[:, None], np.stack([fa+fb-1., 1.-fa, 1.-fb], 1), np.stack([1.-fa-fb, fa, fb], 1))
        w = np.maximum(w, 0.); w /= w.sum(1, keepdims=True)
        corners = np.where(down[:, None], np.stack([local[i0+1, j0+1], local[i0, j0+1], local[i0+1, j0]], 1),
                           np.stack([local[i0, j0], local[i0+1, j0], local[i0, j0+1]], 1))
        # bincount adds in input order, exactly like np.add.at, but without its per-element overhead.
        volume += np.bincount(node[f[:, None], corners].ravel(), (w*arrays['volume'][attached][:, None]).ravel(), count)
    area = np.maximum(lattice['node_area'], 1e-30)
    volume = _smooth_film(volume, area, lattice['edges'], options.film_smoothing)
    pooled_positions = np.empty((0, 3)); pooled_volumes = np.empty(0)
    if options.film_max_thickness > 0.:
        # Tiny lattice areas can turn a hanging drop into a "film" metres thick along the normal.
        # A film cannot exceed the cap; the excess hangs as a round drop just off the surface.
        excess = np.maximum(volume-options.film_max_thickness*area, 0.)
        pooled = np.flatnonzero(excess > 0.)
        volume = volume-excess
        pooled_volumes = excess[pooled]
        drop_radius = np.cbrt(pooled_volumes*.238732414637843)
        pooled_positions = lattice['positions'][pooled]+lattice['normals'][pooled]*(options.film_max_thickness+drop_radius)[:, None]
    represented = float(volume.sum())
    sheen = np.zeros(count)
    if options.film_sheen > 0.:
        # Cosmetic coat: wetted surface keeps a thin continuous layer after the water drains.
        wet = (lattice['wet_weights']*cached.wetness[lattice['wet_nodes']]).sum(1)
        sheen = np.maximum(options.film_sheen*wet*area-volume, 0.)
        volume = volume+sheen
    thickness = volume/area
    mesh = _film_shell(lattice['positions'], lattice['normals'], thickness, lattice['tris'],
                       options.min_thickness, options.max_triangles)
    mesh_volume = _signed_volume(mesh.vertices, mesh.triangles)
    mesh.diagnostics.update(represented_volume=represented, mesh_volume=mesh_volume,
        excluded_volume=max(0., represented+float(sheen.sum())-mesh_volume), film_segments=n, film_nodes=count,
        dry_nodes=int((thickness < options.min_thickness).sum()),
        pooled_volume=float(pooled_volumes.sum()), pooled_positions=pooled_positions, pooled_volumes=pooled_volumes,
        sheen_volume=float(sheen.sum()),
        film_max_spacing=lattice['max_spacing'],
        unanchored_volume=float(arrays['volume'][(arrays['state'] == 0) & ~attached].astype(np.float64).sum()))
    return mesh


def _smooth_film(volume, area, edges, steps):
    """Diffuse film thickness along lattice edges by exchanging volume symmetrically: total volume
    is exact, and each step is a convex average (no negative or overshooting thickness).
    Thins sampling noise from sparse particles so the clip threshold does not punch holes."""
    if not steps or not len(edges): return volume
    a, b = edges[:, 0], edges[:, 1]
    degree = np.bincount(edges.ravel(), minlength=len(volume)).max()
    conductance = .5/degree*np.minimum(area[a], area[b])
    volume = volume.copy()
    for _ in range(steps):
        h = volume/area
        flux = conductance*(h[a]-h[b])
        volume += np.bincount(b, flux, len(volume))-np.bincount(a, flux, len(volume))
    return volume


def _film_shell(positions, normals, thickness, tris, hmin, budget):
    """Closed film shell over nodes with thickness >= hmin, built per clipping case without Python loops.

    Vertices come in (base, top) pairs: wet nodes, then wet/dry edge crossings at height hmin.
    Per small triangle, rotated so its polygon starts at a wet corner:
      3 wet: top + base; 1 wet (a): polygon a, x_ab, x_ca; 2 wet (a, b): polygon a, b, x_bc, x_ca.
    Walls close cuts (x -> x) and lattice boundary edges (used by one small triangle)."""
    count = len(positions)
    wet = thickness >= hmin
    vid = -np.ones(count, np.int64); vid[wet] = 2*np.arange(int(wet.sum()))
    edges = np.sort(np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], 1), axis=2)
    edge_keys = edges[..., 0]*count+edges[..., 1]
    unique, uses = np.unique(edge_keys, return_counts=True)
    boundary = np.isin(edge_keys, unique[uses == 1])            # (T, 3): edge k joins corners k, k+1
    cut = wet[edges[..., 0]] != wet[edges[..., 1]]
    cross_keys = np.unique(edge_keys[cut])
    lo, hi = cross_keys//count, cross_keys % count
    lo, hi = np.where(wet[lo], hi, lo), np.where(wet[lo], lo, hi)  # lo: dry end, hi: wet end
    s = ((hmin-thickness[lo])/(thickness[hi]-thickness[lo]))[:, None]
    cross_points = positions[lo]+s*(positions[hi]-positions[lo])
    cross_normals = normals[lo]+s*(normals[hi]-normals[lo])
    cross_normals /= np.maximum(np.linalg.norm(cross_normals, axis=1, keepdims=True), 1e-30)
    base = 2*int(wet.sum())
    points = np.empty((base+2*len(cross_keys), 3)); vnormals = np.empty_like(points)
    points[0:base:2] = positions[wet]; points[1:base:2] = positions[wet]+normals[wet]*thickness[wet][:, None]
    vnormals[0:base:2] = -normals[wet]; vnormals[1:base:2] = normals[wet]
    points[base::2] = cross_points; points[base+1::2] = cross_points+cross_normals*hmin
    vnormals[base::2] = -cross_normals; vnormals[base+1::2] = cross_normals

    def crossing(rows, k):   # vertex of the crossing on edge k of each row
        return base+2*np.searchsorted(cross_keys, edge_keys[rows, k])

    out = []

    def surface(poly):        # poly: list of (n,) base-vertex arrays in boundary order
        for f in range(len(poly)-2):
            out.append(np.stack([poly[0]+1, poly[f+1]+1, poly[f+2]+1], 1))
            out.append(np.stack([poly[0], poly[f+2], poly[f+1]], 1))

    def wall(a, b, keep):
        a, b = a[keep], b[keep]
        out.append(np.stack([a, b, b+1], 1)); out.append(np.stack([a, b+1, a+1], 1))

    wet_corner = wet[tris]; wet_count = wet_corner.sum(1)
    for case in (1, 2, 3):
        rows = np.flatnonzero(wet_count == case)
        if not len(rows): continue
        flags = wet_corner[rows]
        # Rotation r: corner r is the polygon start (the wet corner for 1, the corner after the dry one for 2).
        r = np.argmax(flags, 1) if case == 1 else ((np.argmin(flags, 1)+1) % 3 if case == 2 else np.zeros(len(rows), np.int64))
        k0, k1, k2 = r, (r+1) % 3, (r+2) % 3
        a, b, c = vid[tris[rows, k0]], vid[tris[rows, k1]], vid[tris[rows, k2]]
        e_ab, e_bc, e_ca = boundary[rows, k0], boundary[rows, k1], boundary[rows, k2]
        if case == 3:
            surface([a, b, c]); wall(a, b, e_ab); wall(b, c, e_bc); wall(c, a, e_ca)
        elif case == 1:
            x_ab, x_ca = crossing(rows, k0), crossing(rows, k2)
            surface([a, x_ab, x_ca]); wall(x_ab, x_ca, np.ones(len(rows), bool))
            wall(a, x_ab, e_ab); wall(x_ca, a, e_ca)
        else:
            x_bc, x_ca = crossing(rows, k1), crossing(rows, k2)
            surface([a, b, x_bc, x_ca]); wall(x_bc, x_ca, np.ones(len(rows), bool))
            wall(a, b, e_ab); wall(b, x_bc, e_bc); wall(x_ca, a, e_ca)
        if sum(len(o) for o in out) > budget: raise ValueError('Film exceeds the triangle budget')
    triangles = np.concatenate(out) if out else np.empty((0, 3), np.int64)
    return MeshBatch(points.astype(np.float32), vnormals.astype(np.float32), triangles.astype(np.int32), {})


# ---------------------------------------------------------------- free drops

def _point_triangle_distance(p, a, b, c):
    ab, ac, ap = b-a, c-a, p-a
    n = np.cross(ab, ac); nn = np.maximum((n*n).sum(1), 1e-300)
    plane = (ap*n).sum(1)/nn
    q = p-plane[:, None]*n
    v0, v1, v2 = ab, ac, q-a
    d00, d01, d11 = (v0*v0).sum(1), (v0*v1).sum(1), (v1*v1).sum(1)
    d20, d21 = (v2*v0).sum(1), (v2*v1).sum(1)
    den = np.maximum(d00*d11-d01*d01, 1e-300)
    v = (d11*d20-d01*d21)/den; w = (d00*d21-d01*d20)/den
    inside = (v >= 0) & (w >= 0) & (v+w <= 1)
    best = np.where(inside, np.abs(plane)*np.sqrt(nn), np.inf)
    for s, e in ((a, b), (b, c), (c, a)):
        d = e-s; t = np.clip(((p-s)*d).sum(1)/np.maximum((d*d).sum(1), 1e-300), 0, 1)
        best = np.minimum(best, np.linalg.norm(p-(s+t[:, None]*d), axis=1))
    return best


class _SurfaceBarrier:
    """Clears grid samples closer than `reach` to the source surface."""
    def __init__(self, vertices, triangles, reach):
        self.v = vertices.astype(np.float64); self.t = triangles.astype(np.int64); self.reach = reach
        p = self.v[self.t]; low = p.min(1)-reach; high = p.max(1)+reach
        self.cell = float((high-low).max())
        lo = np.floor(low/self.cell).astype(np.int64); hi = np.floor(high/self.cell).astype(np.int64)
        self.table = {}
        for index in range(len(self.t)):
            for x in range(lo[index, 0], hi[index, 0]+1):
                for y in range(lo[index, 1], hi[index, 1]+1):
                    for z in range(lo[index, 2], hi[index, 2]+1):
                        self.table.setdefault((x, y, z), []).append(index)

    def near(self, points):
        result = np.zeros(len(points), bool)
        cells = np.floor(points/self.cell).astype(np.int64)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        for u, cell in enumerate(map(tuple, unique)):
            candidates = self.table.get(cell)
            if not candidates: continue
            rows = np.flatnonzero(inverse.ravel() == u)
            tri = np.asarray(candidates)
            pp = np.repeat(points[rows], len(tri), axis=0); tt = np.tile(tri, len(rows))
            d = _point_triangle_distance(pp, *(self.v[self.t[tt, k]] for k in range(3))).reshape(len(rows), len(tri))
            result[rows] = (d < self.reach).any(1)
        return result


def _pca_kernels(p, support, min_neighbours=6, ratio=4., smoothing=.9):
    """Yu & Turk anisotropic kernels: weighted neighbour covariance per drop gives a volume-
    preserving ellipsoid (stretch factors multiply to one), so streams stay thin and connected
    while isolated drops (fewer than `min_neighbours`) stay round. Centres are Laplacian-smoothed.
    Returns smoothed centres, per-drop inverse-stretch matrices and maximum stretch."""
    n = len(p)
    # Bounded neighbourhoods: a few huge merged drops must not size the grid for everyone.
    cell = float(min((2.*support).max(), 4.*np.median(support))) if n else 1.
    reach = np.minimum(2.*support, cell)
    keys = np.floor(p/cell).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0])); sorted_keys = keys[order]
    packed = (sorted_keys[:, 0]*1_000_003+sorted_keys[:, 1])*1_000_003+sorted_keys[:, 2]
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    ranges = []
    for dx, dy, dz in offsets:
        target = ((keys[:, 0]+dx)*1_000_003+(keys[:, 1]+dy))*1_000_003+keys[:, 2]+dz
        ranges.append((np.searchsorted(packed, target, 'left'), np.searchsorted(packed, target, 'right')))
    candidates = sum(hi-lo for lo, hi in ranges)
    found_i, found_j, found_d = [], [], []
    bounds = np.searchsorted(np.cumsum(candidates), np.arange(0, int(candidates.sum())+1, 20_000_000)[1:])
    for begin, stop in zip(np.r_[0, bounds], np.r_[bounds, n]):   # chunks of ~20M candidate pairs
        rows = np.arange(begin, stop)
        if not len(rows): continue
        for lo, hi in ranges:
            count = (hi-lo)[rows]
            i = np.repeat(rows, count)
            j = order[np.repeat(lo[rows], count)+(np.arange(count.sum())-np.repeat(np.cumsum(count)-count, count))]
            d = np.linalg.norm(p[j]-p[i], axis=1); keep = d < reach[i]
            found_i.append(i[keep]); found_j.append(j[keep]); found_d.append(d[keep])
    i, j, d = np.concatenate(found_i), np.concatenate(found_j), np.concatenate(found_d)
    w = 1.-(d/reach[i])**3
    weight = np.bincount(i, w, n)
    mean = np.stack([np.bincount(i, w*p[j, k], n) for k in range(3)], 1)/weight[:, None]
    delta = p[j]-mean[i]
    cov = np.zeros((n, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = cov[:, b, a] = np.bincount(i, w*delta[:, a]*delta[:, b], n)/weight
    values, vectors = np.linalg.eigh(cov)                       # ascending eigenvalues
    isolated = (np.bincount(i, minlength=n) < min_neighbours) | (values[:, 2] <= 0.)
    values[isolated] = 1.
    values = np.maximum(values, values[:, 2:3]/ratio)
    stretch = np.sqrt(values)
    stretch /= np.cbrt(stretch.prod(1))[:, None]                # volume-preserving
    stretch = np.clip(stretch, 1./AXIAL_MAX, AXIAL_MAX)
    matrix = np.einsum('nij,nj,nkj->nik', vectors, 1./stretch, vectors)
    centres = np.where(isolated[:, None], p, (1.-smoothing)*p+smoothing*mean)
    return centres, matrix, stretch.max(1)


def _free_particles(cached, settings, crop=None, kernel='velocity', extra=None):
    a = cached.arrays; free = a['state'] == 1
    cropped = 0.
    if crop is not None:
        inside = ((a['position'] >= crop[0]) & (a['position'] <= crop[1])).all(1)
        cropped = float(a['volume'][free & ~inside].astype(np.float64).sum()); free &= inside
    p = a['position'][free].astype(np.float64); volume = a['volume'][free].astype(np.float64)
    velocity = a['velocity'][free].astype(np.float64)
    if extra is not None and len(extra[1]):   # pendant drops pooled from capped film, at rest
        p = np.concatenate([p, extra[0]]); volume = np.concatenate([volume, extra[1]])
        velocity = np.concatenate([velocity, np.zeros((len(extra[1]), 3))])
    radius = np.cbrt(volume*.238732414637843); support = 2.*radius
    nominal = settings.get('radius')
    speed = np.linalg.norm(velocity, axis=1)
    axial = np.clip(1.+speed*.02/nominal, 1., AXIAL_MAX) if nominal else np.ones(len(p))
    axis = np.where(speed[:, None] > 1e-9, velocity/np.maximum(speed, 1e-30)[:, None], [0., 0., 1.])
    if kernel == 'pca' and len(p):
        p, matrix, axial = _pca_kernels(p, support)
        return (p, volume, radius, support, axial, axis, matrix), cropped
    return (p, volume, radius, support, axial, axis, None), cropped


def _tile_field(tile, particles, members, origin, pitch, T):
    """Sum particle kernels on the tile's sample points (owned cells plus halo).

    Each particle is evaluated only on its own support box clipped to the tile. Particles are
    bucketed by clipped box size for vectorization, then contributions are sorted back to
    (particle order, lexicographic point order) so one bincount adds them in a fixed order:
    the same order as evaluating particles one by one."""
    p, volume, _, support, axial, axis, matrix = particles
    low = tile*T-1; size = T+3
    if not len(members): return np.zeros((size, size, size))
    extent = support[members]*axial[members]
    start = np.floor((p[members]-extent[:, None]-origin)/pitch).astype(np.int64)
    width = (np.ceil(2*extent/pitch)+2).astype(np.int64)
    lo = np.maximum(start, low); hi = np.minimum(start+width[:, None], low+size)
    span = (hi-lo).min(1) > 0
    side = (hi-lo).max(1)
    keys, indices, contributions = [], [], []
    for box in np.unique(side[span]):
        group = np.flatnonzero(span & (side == box))
        offsets = np.stack(np.meshgrid(*[np.arange(box)]*3, indexing='ij'), -1).reshape(-1, 3)
        chunk = max(1, 2_000_000//len(offsets))
        for begin in range(0, len(group), chunk):
            rank = group[begin:begin+chunk]; rows = members[rank]
            g = lo[rank][:, None, :]+offsets[None]
            valid = (g < hi[rank][:, None, :]).all(-1)
            delta = origin+g*pitch-p[rows][:, None, :]
            s2 = (support[rows]**2)[:, None]
            if matrix is None:
                along = (delta*axis[rows][:, None, :]).sum(-1)
                perp = delta-along[..., None]*axis[rows][:, None, :]
                tr2 = (1./axial[rows])[:, None]; ax2 = (axial[rows]**2)[:, None]
                q2 = (perp*perp).sum(-1)/(s2*tr2)+along*along/(s2*ax2)
            else:
                q = np.einsum('nij,nmj->nmi', matrix[rows], delta)
                q2 = (q*q).sum(-1)/s2
            valid &= q2 < 1.
            weight = (volume[rows]*315./(64.*np.pi*support[rows]**3))[:, None]
            value = weight*(1.-q2)**3
            local = g-low
            flat = (local[..., 0]*size+local[..., 1])*size+local[..., 2]
            keys.append((rank[:, None]*size**3+flat)[valid]); indices.append(flat[valid]); contributions.append(value[valid])
    if not keys: return np.zeros((size, size, size))
    order = np.argsort(np.concatenate(keys), kind='stable')
    return np.bincount(np.concatenate(indices)[order], np.concatenate(contributions)[order], size**3).reshape(size, size, size)


def _contour(field, tile, origin, pitch, T, dims):
    """Marching tetrahedra over the tile's owned cells; returns edge keys, positions, normals, triangles."""
    low = tile*T-1
    grad = np.stack(np.gradient(field, pitch), -1)  # central differences on owned points thanks to the halo
    hot = np.argwhere(field[1:T+2, 1:T+2, 1:T+2] >= ISO)+1
    if not len(hot): return None
    # Only cells with a corner at or above ISO can cross; keep their lexicographic order.
    low_cell, high_cell = np.maximum(hot.min(0)-1, 1), np.minimum(hot.max(0), T)
    cells = np.stack(np.meshgrid(*[np.arange(low_cell[a], high_cell[a]+1) for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
    values = np.stack([field[tuple((cells+CORNERS[q]).T)] for q in range(8)], 1)
    crossing = (values.min(1) < ISO) & (values.max(1) >= ISO)
    cells, values = cells[crossing], values[crossing]
    keys, positions, normals = [], [], []
    for tet in TETS:
        tv = values[:, tet]; inside = tv >= ISO; nh = inside.sum(1)
        order = np.argsort(~inside, axis=1, kind='stable'); c = tet[order]
        for count, pattern in ((1, [[(0, 1), (0, 2), (0, 3)]]), (3, [[(0, 3), (1, 3), (2, 3)]]),
                               (2, [[(0, 2), (0, 3), (1, 3)], [(0, 2), (1, 3), (1, 2)]])):
            rows = np.flatnonzero(nh == count)
            if not len(rows): continue
            base = cells[rows]
            pl = (base[:, None, :]+CORNERS[c[rows, count:]]).mean(1); ph = (base[:, None, :]+CORNERS[c[rows, :count]]).mean(1)
            direction = pl-ph
            for tri in pattern:
                corner_a = base[:, None, :]+CORNERS[c[rows][:, [e[0] for e in tri]]]
                corner_b = base[:, None, :]+CORNERS[c[rows][:, [e[1] for e in tri]]]
                ga, gb = corner_a+low, corner_b+low
                ida = (ga[..., 0]*dims[1]+ga[..., 1])*dims[2]+ga[..., 2]
                idb = (gb[..., 0]*dims[1]+gb[..., 1])*dims[2]+gb[..., 2]
                swap = ida > idb   # interpolate from the lower global id for tile-independent vertices
                la = np.where(swap[..., None], corner_b, corner_a); lb = np.where(swap[..., None], corner_a, corner_b)
                fa, fb = field[tuple(np.moveaxis(la, -1, 0))], field[tuple(np.moveaxis(lb, -1, 0))]
                t = ((ISO-fa)/(fb-fa))[..., None]
                pos = origin+((la+low)+t*(lb-la))*pitch
                g = grad[tuple(np.moveaxis(la, -1, 0))]+t*(grad[tuple(np.moveaxis(lb, -1, 0))]-grad[tuple(np.moveaxis(la, -1, 0))])
                nrm = -g/np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-30)
                cross = np.cross(pos[:, 1]-pos[:, 0], pos[:, 2]-pos[:, 0])
                flip = (cross*direction).sum(1) < 0
                pos[flip] = pos[flip][:, [0, 2, 1]]; nrm[flip] = nrm[flip][:, [0, 2, 1]]
                key = np.stack([np.minimum(ida, idb), np.maximum(ida, idb)], -1)
                key[flip] = key[flip][:, [0, 2, 1]]
                keys.append(key.reshape(-1, 2)); positions.append(pos.reshape(-1, 3)); normals.append(nrm.reshape(-1, 3))
    if not keys: return None
    keys = np.concatenate(keys); positions = np.concatenate(positions); normals = np.concatenate(normals)
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return unique, positions[first], normals[first], inverse.reshape(-1, 3)


def _free_tiles(static, cached, options, settings, stats, extra=None):
    particles, cropped = _free_particles(cached, settings, options.free_crop, options.drop_kernel, extra)
    p, volume, radius, support, axial, _, _ = particles
    stats.update(free_volume=float(volume.sum()), subresolution_volume=float(volume[radius < options.spacing].sum()),
                 cropped_volume=cropped, tiles=0, max_tile_cells=0)
    if not len(p): return
    pitch, T = options.spacing, options.tile_cells
    extent = (support*axial)[:, None]
    origin = (np.floor(((p-extent).min(0))/pitch)-2)*pitch
    gmin = np.floor((p-extent-origin)/pitch).astype(np.int64); gmax = np.ceil((p+extent-origin)/pitch).astype(np.int64)
    dims = gmax.max(0)+3
    # Each particle joins every tile whose sample range (owned cells plus halo) it can reach.
    kmin = -((-(gmin-T-1))//T); kmax = (gmax+1)//T
    span = (kmax-kmin+1).max(0)
    pairs = []
    for dx in range(span[0]):
        for dy in range(span[1]):
            for dz in range(span[2]):
                tile = kmin+[dx, dy, dz]; ok = (tile <= kmax).all(1)
                pairs.append(np.column_stack([tile[ok], np.flatnonzero(ok)]))
    pairs = np.concatenate(pairs)
    order = np.lexsort((pairs[:, 3], pairs[:, 2], pairs[:, 1], pairs[:, 0])); pairs = pairs[order]
    tiles, starts = np.unique(pairs[:, :3], axis=0, return_index=True)
    barrier = _SurfaceBarrier(static['source_vertices'], static['source_triangles'], .55*pitch)
    for index, tile in enumerate(tiles):
        end = starts[index+1] if index+1 < len(starts) else len(pairs)
        members = pairs[starts[index]:end, 3]
        field = _tile_field(tile, particles, members, origin, pitch, T)
        wet = np.flatnonzero(field.ravel() > 0)
        if len(wet):
            size = T+3
            local = np.stack(np.unravel_index(wet, (size,)*3), 1)
            near = barrier.near(origin+(local+tile*T-1)*pitch)
            field.ravel()[wet[near]] = 0.
        stats['tiles'] += 1; stats['max_tile_cells'] = max(stats['max_tile_cells'], (T+2)**3)
        contour = _contour(field, tile, origin, pitch, T, dims)
        if contour is None: continue
        keys, positions, normals, triangles = contour
        yield MeshBatch(positions.astype(np.float32), normals.astype(np.float32), triangles.astype(np.int32),
                        dict(vertex_keys=keys, tile=tuple(int(x) for x in tile)))


# ---------------------------------------------------------------- public API

def _signed_volume(vertices, triangles):
    if not len(triangles): return 0.
    q = vertices.astype(np.float64)[triangles]
    return float(np.einsum('ij,ij->i', q[:, 0], np.cross(q[:, 1], q[:, 2])).sum()/6.)


def _iter(static, cached, options, settings, stats, lattice=None):
    film = _film(static, cached, options, lattice)
    extra = (film.diagnostics.pop('pooled_positions'), film.diagnostics.pop('pooled_volumes'))
    yield WaterGeometry(film, MeshBatch.empty(), {'kind': 'attached'})
    for batch in _free_tiles(static, cached, options, settings, stats, extra):
        yield WaterGeometry(MeshBatch.empty(), batch, {'kind': 'free', 'tile': batch.diagnostics['tile']})


def iter_mesh_tiles(reader, frame: int, options: MeshOptions):
    """The attached film chunk first, then one chunk per free-drop tile with global vertex keys."""
    yield from _iter(reader.read_static(), reader.read(frame), options, reader.header.physical_settings, {})


def mesh_cached_frame(reader, frame: int, options: MeshOptions, destination: Path, cancel, *, static=None,
                      lattice=None) -> dict:
    """Mesh one cached frame to destination/frame_NNNNNNN.npz; returns diagnostics."""
    started = perf_counter()
    static = reader.read_static() if static is None else static
    cached = reader.read(frame); stats = {}
    film = None; keys, positions, normals, triangles = [], [], [], []; total = 0
    for chunk in _iter(static, cached, options, reader.header.physical_settings, stats, lattice):
        if cancel(): raise RuntimeError('Meshing was cancelled')
        if chunk.diagnostics['kind'] == 'attached':
            film = chunk.attached; total += len(film.triangles); continue
        batch = chunk.free
        keys.append(batch.diagnostics['vertex_keys']); positions.append(batch.vertices); normals.append(batch.normals)
        triangles.append(batch.triangles+sum(len(k) for k in keys[:-1]))
        total += len(batch.triangles)
        if total > options.max_triangles: raise ValueError('Mesh exceeds the triangle budget')
    if keys:
        unique, first, inverse = np.unique(np.concatenate(keys), axis=0, return_index=True, return_inverse=True)
        free_vertices = np.concatenate(positions)[first]; free_normals = np.concatenate(normals)[first]
        free_triangles = inverse.ravel()[np.concatenate(triangles)].astype(np.int32)
    else:
        free_vertices = free_normals = np.empty((0, 3), np.float32); free_triangles = np.empty((0, 3), np.int32)
    wet = wet_corners(static, cached.wetness)
    arrays = dict(attached_vertices=film.vertices, attached_normals=film.normals, attached_triangles=film.triangles,
                  free_vertices=free_vertices, free_normals=free_normals, free_triangles=free_triangles, wet_corner=wet)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO(); np.savez(buffer, **arrays); data = buffer.getvalue()
    name = f'frame_{frame:07d}.npz'; temporary = destination/(name+'.tmp')
    temporary.write_bytes(data); os.replace(temporary, destination/name)
    free_volume = _signed_volume(free_vertices, free_triangles)
    return dict(frame=frame, file=name, sha256=sha256(data).hexdigest(), bytes=len(data),
        spacing=options.spacing, film_segments=film.diagnostics['film_segments'],
        film_max_spacing=film.diagnostics['film_max_spacing'],
        attached_represented_volume=film.diagnostics['represented_volume'],
        attached_mesh_volume=film.diagnostics['mesh_volume'], attached_excluded_volume=film.diagnostics['excluded_volume'],
        unanchored_volume=film.diagnostics['unanchored_volume'], pooled_volume=film.diagnostics['pooled_volume'],
        sheen_volume=film.diagnostics['sheen_volume'],
        free_volume=stats.get('free_volume', 0.), free_mesh_volume=free_volume,
        subresolution_volume=stats.get('subresolution_volume', 0.), cropped_volume=stats.get('cropped_volume', 0.),
        tiles=stats.get('tiles', 0),
        max_tile_cells=stats.get('max_tile_cells', 0), attached_triangles=len(film.triangles),
        free_triangles=len(free_triangles), seconds=perf_counter()-started)


_WORKER = {}


def _worker_start(cache_path, options):
    from .particle_cache import CacheReader
    reader = CacheReader(cache_path); static = reader.read_static()
    _WORKER.update(reader=reader, static=static, options=options, lattice=film_lattice(static, options))


def _worker_frame(frame, destination):
    w = _WORKER
    return mesh_cached_frame(w['reader'], frame, w['options'], destination, lambda: False,
                             static=w['static'], lattice=w['lattice'])


def _worker_context():
    import multiprocessing, sys
    context = multiprocessing.get_context('spawn')
    # Inside Blender sys.executable is blender.exe; workers need the bundled Python instead.
    python = Path(sys.prefix)/'bin'/('python.exe' if os.name == 'nt' else 'python3')
    if Path(sys.executable).stem.lower().startswith('blender') and python.exists():
        context.set_executable(str(python))
    return context


def iter_mesh_cache(reader, options: MeshOptions, destination: Path, workers: int = 1):
    """Mesh cached frames in order, one result per step; `workers` > 1 meshes frames in parallel
    processes. Closing the generator early marks the sequence cancelled; the manifest is complete
    only after the last frame."""
    destination = Path(destination)
    if destination.exists(): raise FileExistsError(f'Mesh destination already exists: {destination}')
    if not isinstance(workers, int) or workers < 1: raise ValueError('Workers must be a positive integer')
    destination.mkdir(parents=True)
    manifest = dict(schema_version=1, status='writing', options=asdict(options),
                    source_fingerprint=reader.header.source_fingerprint, cache=str(reader.path), frames={})

    def save():
        temporary = destination/'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=1), encoding='utf8'); os.replace(temporary, destination/'manifest.json')
    save()
    frames = range(reader.header.start_frame, reader.header.end_frame+1)
    pool = None
    try:
        if workers == 1:
            static = reader.read_static(); lattice = film_lattice(static, options)
            results = (mesh_cached_frame(reader, f, options, destination, lambda: False, static=static, lattice=lattice)
                       for f in frames)
        else:
            import sys
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(workers, mp_context=_worker_context(), initializer=_worker_start,
                                       initargs=(str(reader.path), options))
            # Spawned workers re-run the parent's main script unless it is hidden; inside Blender that
            # script imports bpy, which the bundled Python cannot load. Workers only need this module.
            main = sys.modules.get('__main__'); hidden = getattr(main, '__file__', None)
            if hidden is not None: del main.__file__
            try: results = pool.map(_worker_frame, frames, [destination]*len(frames))
            finally:
                if hidden is not None: main.__file__ = hidden
        for result in results:
            manifest['frames'][str(result['frame'])] = result; save()
            yield result
    except BaseException:
        manifest['status'] = 'cancelled'; save(); raise
    finally:
        if pool is not None: pool.shutdown(wait=True, cancel_futures=True)
    manifest['status'] = 'complete'; save()


def mesh_cache_sequence(reader, options: MeshOptions, destination: Path, cancel, workers: int = 1) -> dict:
    """Mesh every cached frame, checking `cancel` before each one."""
    steps = iter_mesh_cache(reader, options, destination, workers); results = []
    try:
        while True:
            if cancel(): raise RuntimeError('Meshing was cancelled')
            try: results.append(next(steps))
            except StopIteration: break
    finally:
        steps.close()
    return dict(frames=len(results), seconds=sum(r['seconds'] for r in results))
