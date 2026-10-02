import json
from hashlib import sha256
import numpy as np
import pytest
from flumen.particle_cache import CacheHeader, CachedFrame, CacheWriter, CacheReader, PARTICLE_FIELDS, SCHEMA_VERSION
from flumen.offline_mesher import MeshOptions, iter_mesh_tiles, mesh_cached_frame, mesh_cache_sequence, FILM_GAP

R = 2e-4
DROP = 4/3*np.pi*R**3


def write_cache(path, vertices, triangles, islands, particles, frames=1, wetness=None, velocity=(0., 0., 0.)):
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
            arrays['velocity'][i] = velocity
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


def test_parallel_sequence_matches_serial_bit_for_bit(tmp_path):
    particles = film_particles(count=600)+[((.004, .004, .002), 1, 0, (.3, .3), DROP), ((.004+2*R, .004, .002), 1, 0, (.3, .3), DROP)]
    reader = write_cache(tmp_path/'cache', *SQUARE, [0, 0], particles, frames=3)
    options = MeshOptions(spacing=R/4, film_spacing=.001)
    mesh_cache_sequence(reader, options, tmp_path/'serial', lambda: False)
    mesh_cache_sequence(reader, options, tmp_path/'parallel', lambda: False, workers=2)
    serial = json.loads((tmp_path/'serial'/'manifest.json').read_text(encoding='utf8'))
    parallel = json.loads((tmp_path/'parallel'/'manifest.json').read_text(encoding='utf8'))
    assert parallel['status'] == 'complete'
    assert {f: r['sha256'] for f, r in serial['frames'].items()} == {f: r['sha256'] for f, r in parallel['frames'].items()}
    assert all(r['free_triangles'] > 0 for r in serial['frames'].values())


def test_film_smoothing_closes_speckle_and_conserves_volume(tmp_path):
    # Sparse particles leave a noisy film near the clip threshold: holes appear inside the wet area.
    particles = film_particles(count=3000, total=1.5e-9, seed=7)  # ~15 um mean, ~2 particles per node
    reader = write_cache(tmp_path/'noisy', *SQUARE, [0, 0], particles)
    holes, volumes = [], []
    for iterations in (0, 6):
        film = list(iter_mesh_tiles(reader, 1, MeshOptions(spacing=.0004, film_smoothing=iterations)))[0].attached
        volumes.append(film.diagnostics['represented_volume'])
        holes.append(film.diagnostics['dry_nodes'])
        assert closed(film.triangles)
    assert abs(volumes[1]-volumes[0]) <= volumes[0]*1e-9
    assert holes[1] < holes[0]*.5
    with pytest.raises(ValueError): MeshOptions(spacing=.001, film_smoothing=-1)


def test_free_drop_crop_reports_excluded_volume(tmp_path):
    source = ([[0, 0, -.01], [.001, 0, -.01], [0, .001, -.01]], [[0, 1, 2]])
    inside, outside = ((0, 0, 0), 1, 0, (.3, .3), DROP), ((0, 0, -1.), 1, 0, (.3, .3), 2*DROP)
    reader = write_cache(tmp_path/'cache', *source, [0], [inside, outside])
    full = mesh_cached_frame(reader, 1, MeshOptions(spacing=R/4), tmp_path/'full', lambda: False)
    crop = mesh_cached_frame(reader, 1, MeshOptions(spacing=R/4, free_crop=((-.1, -.1, -.1), (.1, .1, .1))),
                             tmp_path/'crop', lambda: False)
    assert crop['cropped_volume'] == pytest.approx(2*DROP) and full['cropped_volume'] == 0
    alone = mesh_cached_frame(write_cache(tmp_path/'alone', *source, [0], [inside]), 1, MeshOptions(spacing=R/4),
                              tmp_path/'alone_mesh', lambda: False)
    assert crop['tiles'] < full['tiles'] and crop['free_triangles'] == alone['free_triangles'] < full['free_triangles']
    with pytest.raises(ValueError): MeshOptions(spacing=.001, free_crop=((0, 0, 0), (-1, 0, 0)))


def test_pca_kernel_keeps_isolated_drops_round_and_streams_connected(tmp_path, monkeypatch):
    from flumen import offline_mesher as M
    # An isolated fast drop: the velocity kernel stretches it into a needle, PCA keeps it round.
    reader = write_cache(tmp_path/'fast', [[0, 0, -.01], [.001, 0, -.01], [0, .001, -.01]], [[0, 1, 2]], [0],
                         [((0, 0, 0), 1, 0, (.3, .3), DROP)])
    frame = reader.read(1); frame.arrays['velocity'][:] = (0., 0., -3.)
    monkeypatch.setattr(reader, 'read', lambda f: frame)
    settings = {'radius': R}
    extents = {}
    for kernel in ('velocity', 'pca'):
        options = MeshOptions(spacing=R/4, drop_kernel=kernel)
        mesh = [c.free for c in M._iter(reader.read_static(), frame, options, settings, {}) if c.diagnostics['kind'] == 'free']
        v = np.concatenate([m.vertices for m in mesh])
        extents[kernel] = np.ptp(v, axis=0)
    assert extents['velocity'][2] > 1.5*extents['velocity'][0]          # stretched along velocity
    assert extents['pca'][2] < 1.25*extents['pca'][0]                     # round
    # A line of close drops meshes as one connected stream, thinner than isotropic blobs.
    line = [((k*R*.8, 0, 0), 1, 0, (.3, .3), DROP) for k in range(40)]
    reader = write_cache(tmp_path/'line', [[0, 0, -.01], [.001, 0, -.01], [0, .001, -.01]], [[0, 1, 2]], [0], line)
    result = mesh_cached_frame(reader, 1, MeshOptions(spacing=R/4, drop_kernel='pca'), tmp_path/'line_mesh', lambda: False)
    mesh = np.load(tmp_path/'line_mesh'/result['file'])
    assert components(mesh['free_triangles']) == 1 and closed(mesh['free_triangles'])
    span = np.ptp(mesh['free_vertices'], axis=0)
    assert span[0] > 20*R and span[1] < 3*R
    with pytest.raises(ValueError): MeshOptions(spacing=.001, drop_kernel='bogus')


def test_film_cap_turns_pooled_water_into_a_pendant_drop(tmp_path):
    # Lots of water anchored at one corner node: uncapped film shoots a long spike along the normal.
    particles = [((0., 0., 0.), 0, 0, (0.999, 0.0005), 5e-10) for _ in range(20)]
    reader = write_cache(tmp_path/'pool', *SQUARE, [0, 0], particles)
    spike = mesh_cached_frame(reader, 1, MeshOptions(spacing=.0002, film_spacing=.001), tmp_path/'spike', lambda: False)
    capped = mesh_cached_frame(reader, 1, MeshOptions(spacing=.0002, film_spacing=.001, film_max_thickness=.002),
                               tmp_path/'capped', lambda: False)
    a = np.load(tmp_path/'spike'/spike['file']); b = np.load(tmp_path/'capped'/capped['file'])
    assert a['attached_vertices'][:, 2].max() > .01                         # uncapped: >1 cm spike
    assert b['attached_vertices'][:, 2].max() <= .002+FILM_GAP+1e-6         # capped film (floated off the source)
    assert len(b['free_triangles']) > 0 and closed(b['free_triangles'])     # pooled water became a drop
    assert capped['pooled_volume'] > 0
    total = capped['attached_represented_volume']+capped['pooled_volume']
    assert abs(total-spike['attached_represented_volume']) <= 1e-9*total  # volume moved, not lost


def test_film_sheen_keeps_wetted_surface_coated(tmp_path):
    # No particles left, but the whole square was wetted: a sheen keeps a thin continuous coat.
    reader = write_cache(tmp_path/'wet', *SQUARE, [0, 0], [], wetness=[1., 1., 1., 1.])
    bare = mesh_cached_frame(reader, 1, MeshOptions(spacing=.001), tmp_path/'bare', lambda: False)
    sheen = mesh_cached_frame(reader, 1, MeshOptions(spacing=.001, film_sheen=2e-5), tmp_path/'sheen', lambda: False)
    assert bare['attached_triangles'] == 0 and bare['sheen_volume'] == 0
    mesh = np.load(tmp_path/'sheen'/sheen['file'])
    assert closed(mesh['attached_triangles']) and components(mesh['attached_triangles']) == 1
    assert sheen['sheen_volume'] == pytest.approx(1e-4*2e-5, rel=1e-3)          # area x sheen, all cosmetic
    assert sheen['attached_represented_volume'] == 0                            # no particle water claimed
    # Half-wet surface: only the wetted half gets a coat.
    half = write_cache(tmp_path/'half', *SQUARE, [0, 0], [], wetness=[1., 0., 1., 0.])
    part = mesh_cached_frame(half, 1, MeshOptions(spacing=.001, film_sheen=2e-5), tmp_path/'part', lambda: False)
    assert np.load(tmp_path/'part'/part['file'])['attached_vertices'][:, 0].max() < .01


def test_free_vertices_carry_drop_velocity_for_motion_blur(tmp_path):
    column = [((0, 0, k*3*R), 1, 0, (.3, .3), DROP) for k in range(3)]
    reader = write_cache(tmp_path/'fall', *SQUARE, [0], column, velocity=(.5, 0., -3.))
    result = mesh_cached_frame(reader, 1, MeshOptions(spacing=R/2), tmp_path/'mesh', lambda: False)
    mesh = np.load(tmp_path/'mesh'/result['file'])
    assert len(mesh['free_vertices']) and mesh['free_velocity'].shape == mesh['free_vertices'].shape
    assert np.allclose(mesh['free_velocity'], (.5, 0., -3.), atol=1e-5)


def test_film_never_shares_a_plane_with_the_source(tmp_path):
    # Coplanar film and source faces z-fight in path tracers (maze pattern).
    reader = write_cache(tmp_path/'film', *SQUARE, [0, 0], film_particles())
    result = mesh_cached_frame(reader, 1, MeshOptions(spacing=.001), tmp_path/'mesh', lambda: False)
    z = np.load(tmp_path/'mesh'/result['file'])['attached_vertices'][:, 2]
    assert len(z) and z.min() >= .99e-5
