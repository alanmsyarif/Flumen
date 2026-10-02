import json
from hashlib import sha256
import numpy as np
import pytest
from flumen.particle_cache import CacheHeader, CachedFrame, CacheWriter, CacheReader, PARTICLE_FIELDS, SCHEMA_VERSION
from flumen.offline_mesher import MeshOptions, iter_mesh_tiles, mesh_cached_frame, mesh_cache_sequence

R = 2e-4
DROP = 4/3*np.pi*R**3


def write_cache(path, vertices, triangles, islands, particles, frames=1, wetness=None):
    """particles: list of (position, state, face, bary, volume) per frame (same for each frame)."""
    vertices = np.asarray(vertices, np.float32); triangles = np.asarray(triangles, np.int32)
    normals = np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]], vertices[triangles[:, 2]]-vertices[triangles[:, 0]])
    header = CacheHeader(SCHEMA_VERSION, 'src', {'g': 1}, 1/60, 1, frames, max(len(particles), 1), 'chart', len(vertices))
    writer = CacheWriter(path, header)
    writer.write_static(dict(source_vertices=vertices, source_triangles=triangles,
        source_islands=np.asarray(islands, np.int32), evaluated_triangle_ids=np.arange(len(triangles), dtype=np.int32),
        chart_vertices=vertices, chart_triangles=triangles, chart_areas=np.ones(len(vertices)),
        chart_normals=normals/np.linalg.norm(normals, axis=1, keepdims=True),
        chart_original_faces=np.arange(len(triangles), dtype=np.int32), chart_level=np.array([0], np.int32),
        corner_uv=np.zeros((len(triangles), 3, 2), np.float32), material_index=np.zeros(len(triangles), np.int32)))
    n = len(particles)
    for frame in range(1, frames+1):
        arrays = {name: np.zeros((n,) if width == 1 else (n, width), dtype) for name, (dtype, width) in PARTICLE_FIELDS.items()}
        for i, (position, state, face, bary, volume) in enumerate(particles):
            arrays['position'][i] = position; arrays['state'][i] = state; arrays['face'][i] = face
            arrays['bary'][i] = bary; arrays['volume'][i] = volume; arrays['ids'][i] = i; arrays['slot'][i] = i
        w = np.zeros(len(vertices), np.float32) if wetness is None else np.asarray(wetness, np.float32)
        writer.write(CachedFrame(frame, arrays, w, np.zeros(4, np.int64), np.zeros(2, np.float64), n))
    writer.finish()
    return CacheReader(path)


SQUARE = ([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [.01, .01, 0]], [[0, 1, 2], [1, 3, 2]])


def film_particles(fraction=1., count=4000, seed=1, total=2e-9):
    rng = np.random.default_rng(seed); out = []
    for k in range(count):
        face = k % 2; u, v = rng.random(2)
        if u+v > 1: u, v = 1-u, 1-v
        vertices = np.asarray(SQUARE[0]); tri = SQUARE[1][face]
        p = u*vertices[tri[0]]+v*vertices[tri[1]]+(1-u-v)*vertices[tri[2]]
        if p[0] <= .01*fraction: out.append((p, 0, face, (u, v), total/count))
    return out


def components(triangles):
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b, c in triangles:
        parent[find(b)] = find(a); parent[find(c)] = find(a)
    return len({find(v) for v in parent})


def closed(triangles):
    edges = {}
    for tri in triangles:
        for k in range(3):
            key = tuple(sorted((int(tri[k]), int(tri[(k+1) % 3]))))
            edges[key] = edges.get(key, 0)+1
    faces = {tuple(sorted(map(int, t))) for t in triangles}
    return all(count == 2 for count in edges.values()) and len(faces) == len(triangles)


def signed_volume(vertices, triangles):
    p = vertices.astype(np.float64)[triangles]
    return float(np.einsum('ij,ij->i', p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()/6)


def test_refined_film_volume_and_dry_holes(tmp_path):
    full = write_cache(tmp_path/'full', *SQUARE, [0, 0], film_particles())
    chunks = list(iter_mesh_tiles(full, 1, MeshOptions(spacing=.001)))
    film = chunks[0].attached
    represented = film.diagnostics['represented_volume']
    assert abs(represented-2e-9) <= 2e-9*1e-6
    assert abs(signed_volume(film.vertices, film.triangles)-represented) <= represented*.05
    assert closed(film.triangles)
    half = write_cache(tmp_path/'half', *SQUARE, [0, 0], film_particles(.5))
    errors = []
    for spacing in (.002, .0005):
        film = list(iter_mesh_tiles(half, 1, MeshOptions(spacing=spacing)))[0].attached
        rep = film.diagnostics['represented_volume']
        errors.append(abs(signed_volume(film.vertices, film.triangles)-rep)/rep)
        assert film.vertices[:, 0].max() <= .005+2*spacing  # dry half stays open
    assert errors[1] <= errors[0]+1e-6 and errors[1] <= .05


def test_free_neck_breakup_and_tile_seams(tmp_path):
    source = ([[0, 0, -.01], [.001, 0, -.01], [0, .001, -.01]], [[0, 1, 2]])
    touching = [((0, 0, 0), 1, 0, (.3, .3), DROP), ((2*R, 0, 0), 1, 0, (.3, .3), DROP)]
    apart = [((0, 0, 0), 1, 0, (.3, .3), DROP), ((5*R, 0, 0), 1, 0, (.3, .3), DROP)]
    options = MeshOptions(spacing=R/4)
    for name, particles, expected in (('touch', touching, 1), ('apart', apart, 2)):
        reader = write_cache(tmp_path/name, *source, [0], particles)
        result = mesh_cached_frame(reader, 1, options, tmp_path/(name+'_mesh'), lambda: False)
        mesh = np.load(tmp_path/(name+'_mesh')/result['file'])
        assert components(mesh['free_triangles']) == expected
        assert closed(mesh['free_triangles'])
    reader = write_cache(tmp_path/'seam', *source, [0], touching)
    small = mesh_cached_frame(reader, 1, MeshOptions(spacing=R/4, tile_cells=4), tmp_path/'small', lambda: False)
    large = mesh_cached_frame(reader, 1, MeshOptions(spacing=R/4, tile_cells=32), tmp_path/'large', lambda: False)
    a = np.load(tmp_path/'small'/small['file']); b = np.load(tmp_path/'large'/large['file'])
    assert small['tiles'] > large['tiles']
    assert closed(a['free_triangles'])
    assert len(a['free_triangles']) == len(b['free_triangles'])
    assert abs(signed_volume(a['free_vertices'], a['free_triangles'])-signed_volume(b['free_vertices'], b['free_triangles'])) < 1e-15
    assert small['max_tile_cells'] <= 34**3 and large['max_tile_cells'] <= 34**3
    assert small['free_mesh_volume'] > 0 and 'subresolution_volume' in small


def test_folded_surface_and_wetness_provenance(tmp_path):
    # Two separate parallel sheets 0.5 mm apart, both wet: films must never bridge.
    vertices = [[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .0005], [.01, 0, .0005], [0, .01, .0005]]
    triangles = [[0, 1, 2], [3, 5, 4]]
    particles = []
    rng = np.random.default_rng(3)
    for k in range(2000):
        face = k % 2; u, v = rng.random(2)
        if u+v > 1: u, v = 1-u, 1-v
        tri = np.asarray(vertices)[triangles[face]]
        particles.append((u*tri[0]+v*tri[1]+(1-u-v)*tri[2], 0, face, (u, v), 1e-12))
    wetness = [.1, .2, .3, .4, .5, .6]
    reader = write_cache(tmp_path/'sheets', vertices, triangles, [0, 1], particles, wetness=wetness)
    before = sha256((tmp_path/'sheets'/'frame_0000001.npz').read_bytes()).hexdigest()
    result = mesh_cached_frame(reader, 1, MeshOptions(spacing=.001), tmp_path/'m1', lambda: False)
    mesh_cached_frame(reader, 1, MeshOptions(spacing=.0005), tmp_path/'m2', lambda: False)
    assert sha256((tmp_path/'sheets'/'frame_0000001.npz').read_bytes()).hexdigest() == before
    mesh = np.load(tmp_path/'m1'/result['file'])
    z = mesh['attached_vertices'][:, 2][mesh['attached_triangles']]
    lower = (z < .00025).all(axis=1); upper = (z >= .00025).all(axis=1)
    assert (lower | upper).all()  # no triangle spans both sheets
    # Wet proxy corners carry cached wetness through global face/bary provenance.
    np.testing.assert_allclose(mesh['wet_corner'], np.asarray(wetness, np.float32)[triangles], rtol=1e-6)


def test_mesh_cancel_and_budget(tmp_path):
    reader = write_cache(tmp_path/'cache', *SQUARE, [0, 0], film_particles(count=500), frames=3)
    calls = {'n': 0}
    def cancel():
        calls['n'] += 1; return calls['n'] > 1
    with pytest.raises(RuntimeError, match='cancel'):
        mesh_cache_sequence(reader, MeshOptions(spacing=.001), tmp_path/'seq', cancel)
    manifest = json.loads((tmp_path/'seq'/'manifest.json').read_text(encoding='utf8'))
    assert manifest['status'] != 'complete'
    with pytest.raises(ValueError, match='budget'):
        mesh_cached_frame(reader, 1, MeshOptions(spacing=.0002, max_triangles=10), tmp_path/'tiny', lambda: False)
    done = mesh_cache_sequence(reader, MeshOptions(spacing=.001), tmp_path/'ok', lambda: False)
    manifest = json.loads((tmp_path/'ok'/'manifest.json').read_text(encoding='utf8'))
    assert manifest['status'] == 'complete' and sorted(manifest['frames']) == ['1', '2', '3']
    assert done['frames'] == 3
