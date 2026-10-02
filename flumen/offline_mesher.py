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

    def __post_init__(self):
        for value in (self.spacing, self.film_spacing, self.min_thickness):
            if not isfinite(value) or value < 0: raise ValueError('Mesh distances must be finite and nonnegative')
        if self.spacing <= 0: raise ValueError('Mesh spacing must be positive')
        if not 1 <= self.tile_cells <= 32: raise ValueError('Tiles hold 1 to 32 cells per axis')
        if self.max_triangles < 1: raise ValueError('Triangle budget must be positive')


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


def _film(static, cached, options):
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
    tris = node[:, small].reshape(-1, 3)
    cross = np.cross(V[T[:, 0]]-V[T[:, 2]], V[T[:, 1]]-V[T[:, 2]])
    small_area = np.repeat(np.linalg.norm(cross, axis=1)*.5/(n*n), len(small))
    normals = np.zeros((count, 3)); node_area = np.zeros(count)
    np.add.at(normals, tris.ravel(), np.repeat(np.repeat(cross, len(small), axis=0), 3, axis=0))
    np.add.at(node_area, tris.ravel(), np.repeat(small_area/3., 3))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
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
        np.add.at(volume, node[f[:, None], corners].ravel(), (w*arrays['volume'][attached][:, None]).ravel())
    thickness = volume/np.maximum(node_area, 1e-30)
    represented = float(volume.sum())
    mesh = _film_shell(positions, normals, thickness, tris, options.min_thickness, options.max_triangles)
    mesh_volume = _signed_volume(mesh.vertices, mesh.triangles)
    mesh.diagnostics.update(represented_volume=represented, mesh_volume=mesh_volume,
        excluded_volume=max(0., represented-mesh_volume), film_segments=n, film_nodes=count,
        film_max_spacing=float(lengths.max())/n,
        unanchored_volume=float(arrays['volume'][(arrays['state'] == 0) & ~attached].astype(np.float64).sum()))
    return mesh


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


def _free_particles(cached, settings):
    a = cached.arrays; free = a['state'] == 1
    p = a['position'][free].astype(np.float64); volume = a['volume'][free].astype(np.float64)
    velocity = a['velocity'][free].astype(np.float64)
    radius = np.cbrt(volume*.238732414637843); support = 2.*radius
    nominal = settings.get('radius')
    speed = np.linalg.norm(velocity, axis=1)
    axial = np.clip(1.+speed*.02/nominal, 1., AXIAL_MAX) if nominal else np.ones(len(p))
    axis = np.where(speed[:, None] > 1e-9, velocity/np.maximum(speed, 1e-30)[:, None], [0., 0., 1.])
    return p, volume, radius, support, axial, axis


def _tile_field(tile, particles, members, origin, pitch, T):
    p, volume, _, support, axial, axis = particles
    low = tile*T-1; size = T+3
    field = np.zeros(size**3)
    if len(members):
        extent = support[members]*axial[members]
        width = int(np.ceil(2*extent.max()/pitch))+2
        offsets = np.stack(np.meshgrid(*[np.arange(width)]*3, indexing='ij'), -1).reshape(-1, 3)
        chunk = max(1, 2_000_000//len(offsets))
        for begin in range(0, len(members), chunk):
            rows = members[begin:begin+chunk]
            start = np.floor((p[rows]-(support[rows]*axial[rows])[:, None]-origin)/pitch).astype(np.int64)
            g = start[:, None, :]+offsets[None]
            local = g-low
            valid = ((local >= 0) & (local < size)).all(-1)
            delta = origin+g*pitch-p[rows][:, None, :]
            along = (delta*axis[rows][:, None, :]).sum(-1)
            perp = delta-along[..., None]*axis[rows][:, None, :]
            s2 = (support[rows]**2)[:, None]; tr2 = (1./axial[rows])[:, None]; ax2 = (axial[rows]**2)[:, None]
            q2 = (perp*perp).sum(-1)/(s2*tr2)+along*along/(s2*ax2)
            valid &= q2 < 1.
            weight = (volume[rows]*315./(64.*np.pi*support[rows]**3))[:, None]
            value = weight*(1.-q2)**3
            flat = (local[..., 0]*size+local[..., 1])*size+local[..., 2]
            np.add.at(field, flat[valid], value[valid])
    return field.reshape(size, size, size)


def _contour(field, tile, origin, pitch, T, dims):
    """Marching tetrahedra over the tile's owned cells; returns edge keys, positions, normals, triangles."""
    low = tile*T-1
    grad = np.stack(np.gradient(field, pitch), -1)  # central differences on owned points thanks to the halo
    cells = np.stack(np.meshgrid(*[np.arange(1, T+1)]*3, indexing='ij'), -1).reshape(-1, 3)
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


def _free_tiles(static, cached, options, settings, stats):
    particles = _free_particles(cached, settings)
    p, volume, radius, support, axial, _ = particles
    stats.update(free_volume=float(volume.sum()), subresolution_volume=float(volume[radius < options.spacing].sum()),
                 tiles=0, max_tile_cells=0)
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


def _iter(static, cached, options, settings, stats):
    film = _film(static, cached, options)
    yield WaterGeometry(film, MeshBatch.empty(), {'kind': 'attached'})
    for batch in _free_tiles(static, cached, options, settings, stats):
        yield WaterGeometry(MeshBatch.empty(), batch, {'kind': 'free', 'tile': batch.diagnostics['tile']})


def iter_mesh_tiles(reader, frame: int, options: MeshOptions):
    """The attached film chunk first, then one chunk per free-drop tile with global vertex keys."""
    yield from _iter(reader.read_static(), reader.read(frame), options, reader.header.physical_settings, {})


def mesh_cached_frame(reader, frame: int, options: MeshOptions, destination: Path, cancel, *, static=None) -> dict:
    """Mesh one cached frame to destination/frame_NNNNNNN.npz; returns diagnostics."""
    started = perf_counter()
    static = reader.read_static() if static is None else static
    cached = reader.read(frame); stats = {}
    film = None; keys, positions, normals, triangles = [], [], [], []; total = 0
    for chunk in _iter(static, cached, options, reader.header.physical_settings, stats):
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
        unanchored_volume=film.diagnostics['unanchored_volume'],
        free_volume=stats.get('free_volume', 0.), free_mesh_volume=free_volume,
        subresolution_volume=stats.get('subresolution_volume', 0.), tiles=stats.get('tiles', 0),
        max_tile_cells=stats.get('max_tile_cells', 0), attached_triangles=len(film.triangles),
        free_triangles=len(free_triangles), seconds=perf_counter()-started)


def iter_mesh_cache(reader, options: MeshOptions, destination: Path):
    """Mesh one cached frame per step. Closing the generator early marks the sequence cancelled;
    the manifest is complete only after the last frame."""
    destination = Path(destination)
    if destination.exists(): raise FileExistsError(f'Mesh destination already exists: {destination}')
    destination.mkdir(parents=True)
    manifest = dict(schema_version=1, status='writing', options=asdict(options),
                    source_fingerprint=reader.header.source_fingerprint, cache=str(reader.path), frames={})

    def save():
        temporary = destination/'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=1), encoding='utf8'); os.replace(temporary, destination/'manifest.json')
    save()
    try:
        static = reader.read_static()
        for frame in range(reader.header.start_frame, reader.header.end_frame+1):
            result = mesh_cached_frame(reader, frame, options, destination, lambda: False, static=static)
            manifest['frames'][str(frame)] = result; save()
            yield result
    except BaseException:
        manifest['status'] = 'cancelled'; save(); raise
    manifest['status'] = 'complete'; save()


def mesh_cache_sequence(reader, options: MeshOptions, destination: Path, cancel) -> dict:
    """Mesh every cached frame, checking `cancel` before each one."""
    steps = iter_mesh_cache(reader, options, destination); results = []
    try:
        while True:
            if cancel(): raise RuntimeError('Meshing was cancelled')
            try: results.append(next(steps))
            except StopIteration: break
    finally:
        steps.close()
    return dict(frames=len(results), seconds=sum(r['seconds'] for r in results))
