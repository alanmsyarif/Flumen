import json
import subprocess
import sys
import numpy as np
import pytest
from flumen.particle_cache import (CacheHeader, CachedFrame, CacheWriter, CacheReader, PARTICLE_FIELDS,
                                   SCHEMA_VERSION, estimate_cache_bytes)

SETTINGS = {'gravity': [0., 0., -9.81], 'resistance': 60., 'solver_backend': 'FIELD'}


def header(**changes):
    values = dict(schema_version=SCHEMA_VERSION, source_fingerprint='source-a', physical_settings=SETTINGS,
                  physical_dt=1/60, start_frame=1, end_frame=3, capacity=16, chart_fingerprint='chart-a',
                  field_nodes=5)
    values.update(changes)
    return CacheHeader(**values)


def frame(number, count=7, seed=0):
    rng = np.random.default_rng(seed+number)
    arrays = {}
    for name, (dtype, width) in PARTICLE_FIELDS.items():
        shape = (count,) if width == 1 else (count, width)
        arrays[name] = (rng.random(shape)*100).astype(dtype)
    return CachedFrame(number, arrays, rng.random(5).astype(np.float32), np.arange(4, dtype=np.int64)+number,
                       np.array([1e-9*number, 1e-12], dtype=np.float64), 1000+number)


def bake(path, frames=(1, 2, 3), finish=True, **changes):
    writer = CacheWriter(path, header(**changes))
    writer.write_static({'vertices': np.zeros((4, 3), np.float32), 'triangles': np.array([[0, 1, 2]], np.int32)})
    written = [frame(n) for n in frames]
    for item in written: writer.write(item)
    if finish: writer.finish()
    return writer, written


def test_cache_roundtrip_and_size(tmp_path):
    path = tmp_path/'cache'
    _, written = bake(path)
    reader = CacheReader(path)
    reader.validate('source-a', SETTINGS)
    static = reader.read_static()
    np.testing.assert_array_equal(static['triangles'], [[0, 1, 2]])
    for expected in written:
        actual = reader.read(expected.frame)
        for name in PARTICLE_FIELDS:
            np.testing.assert_array_equal(actual.arrays[name], expected.arrays[name])
            assert actual.arrays[name].dtype == expected.arrays[name].dtype
        np.testing.assert_array_equal(actual.wetness, expected.wetness)
        np.testing.assert_array_equal(actual.counters, expected.counters)
        np.testing.assert_array_equal(actual.ledger, expected.ledger)
        assert actual.next_id == expected.next_id
    static_bytes = (path/'static.npz').stat().st_size
    actual_bytes = sum(p.stat().st_size for p in path.iterdir())
    assert actual_bytes <= estimate_cache_bytes(reader.header, static_bytes)
    assert estimate_cache_bytes(header(capacity=1_000_000, end_frame=1)) > 1_000_000*80


def test_writer_refuses_existing_and_object_or_nonfinite_data(tmp_path):
    (tmp_path/'taken').mkdir()
    with pytest.raises(FileExistsError): CacheWriter(tmp_path/'taken', header())
    writer = CacheWriter(tmp_path/'cache', header())
    with pytest.raises(ValueError): writer.write_static({'bad': np.array([object()], dtype=object)})
    bad = frame(1); bad.arrays['position'][0, 0] = np.nan
    with pytest.raises(ValueError): writer.write(bad)
    with pytest.raises(ValueError): writer.write(frame(2))  # frames must be sequential
    big = frame(1, count=17)
    with pytest.raises(ValueError): writer.write(big)  # exceeds capacity


@pytest.mark.parametrize('damage', ['truncate', 'delete', 'flip', 'unfinished', 'cancelled', 'manifest'])
def test_incomplete_corrupt_and_mismatch_rejected(tmp_path, damage):
    path = tmp_path/'cache'
    if damage == 'unfinished':
        bake(path, frames=(1, 2), finish=False)
    elif damage == 'cancelled':
        writer, _ = bake(path, frames=(1,), finish=False); writer.cancel()
    else:
        bake(path)
    target = path/'frame_0000002.npz'
    if damage == 'truncate': target.write_bytes(target.read_bytes()[:-10])
    if damage == 'delete': target.unlink()
    if damage == 'flip':
        data = bytearray(target.read_bytes()); data[len(data)//2] ^= 1; target.write_bytes(bytes(data))
    if damage == 'manifest': (path/'manifest.json').write_text('{not json', encoding='utf8')
    with pytest.raises(ValueError):
        reader = CacheReader(path)
        for number in (1, 2, 3): reader.read(number)


def test_source_and_settings_mismatch_rejected(tmp_path):
    bake(tmp_path/'cache')
    reader = CacheReader(tmp_path/'cache')
    with pytest.raises(ValueError): reader.validate('source-b', SETTINGS)
    with pytest.raises(ValueError): reader.validate('source-a', {**SETTINGS, 'resistance': 5.})
    with pytest.raises(ValueError): reader.read(4)


def test_no_pickle_and_no_cuda_required_to_read(tmp_path):
    bake(tmp_path/'cache')
    manifest = json.loads((tmp_path/'cache'/'manifest.json').read_text(encoding='utf8'))
    assert manifest['status'] == 'complete' and manifest['world_units'] == 'meters'
    assert all('O' not in dtype for record in manifest['frames'].values() for dtype, _ in record['arrays'].values())
    script = ('import sys; from flumen.particle_cache import CacheReader; '
              f'r=CacheReader(r"{tmp_path/"cache"}"); r.read(3); r.read_static(); '
              'assert "warp" not in sys.modules and "bpy" not in sys.modules; print("ok")')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.stdout.strip() == 'ok', result.stderr
