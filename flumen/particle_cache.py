"""Immutable, validated particle caches. Pure NumPy: reading needs no CUDA or Warp.

Layout of a cache directory:
  manifest.json        schema, header, per-file SHA256/byte/array records, status
  static.npz           world source/chart data, written once
  frame_0000001.npz    one file per integer frame, active particles only
Every file is written to a temporary name and atomically renamed. Only finish()
marks a cache complete, after re-hashing every file on disk.
"""
from dataclasses import dataclass, asdict, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import os
import numpy as np

SCHEMA_VERSION = 1
# name: (dtype, components per particle)
PARTICLE_FIELDS = {
    'position': ('float32', 3), 'velocity': ('float32', 3), 'normal': ('float32', 3),
    'age': ('float32', 1), 'volume': ('float32', 1), 'state': ('int32', 1), 'island': ('int32', 1),
    'face': ('int32', 1), 'bary': ('float32', 2), 'ids': ('int64', 1), 'path': ('int64', 1),
    'limited': ('int32', 1), 'slot': ('int32', 1),
}
PARTICLE_ROW_BYTES = sum(np.dtype(dtype).itemsize*width for dtype, width in PARTICLE_FIELDS.values())
_NPZ_OVERHEAD = 32*1024          # zip directory and .npy headers, generous
_MANIFEST_LIMIT = 64*1024*1024
_STATIC_LIMIT = 4*1024**3


@dataclass
class CacheHeader:
    schema_version: int
    source_fingerprint: str
    physical_settings: dict
    physical_dt: float
    start_frame: int
    end_frame: int
    capacity: int
    chart_fingerprint: str
    field_nodes: int = 0  # wetness values per frame; sizes the estimate and read bounds

    @property
    def frame_count(self):
        return self.end_frame-self.start_frame+1


@dataclass
class CachedFrame:
    frame: int
    arrays: dict
    wetness: np.ndarray
    counters: np.ndarray
    ledger: np.ndarray
    next_id: int


def _frame_bound(header):
    return header.capacity*PARTICLE_ROW_BYTES+header.field_nodes*4+6*8+_NPZ_OVERHEAD


def estimate_cache_bytes(header: CacheHeader, static_bytes: int = 0) -> int:
    """Upper bound for a finished cache: every frame at full capacity plus fixed data."""
    return static_bytes+_NPZ_OVERHEAD+header.frame_count*(_frame_bound(header)+512)+64*1024


def _canonical(settings):
    return json.loads(json.dumps(settings, sort_keys=True))


def _check_arrays(arrays):
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype.kind not in 'biuf':
            raise ValueError(f'{name}: only numeric NumPy arrays can be cached')
        if array.dtype.kind == 'f' and not np.isfinite(array).all():
            raise ValueError(f'{name}: nonfinite values')


def _validate_frame(frame: CachedFrame, header: CacheHeader):
    if set(frame.arrays) != set(PARTICLE_FIELDS):
        raise ValueError('Frame particle fields do not match the cache schema')
    count = None
    for name, (dtype, width) in PARTICLE_FIELDS.items():
        array = frame.arrays[name]
        expected = (len(array),) if width == 1 else (len(array), width)
        if array.dtype != np.dtype(dtype) or array.shape != expected:
            raise ValueError(f'{name}: expected {dtype} {expected}, got {array.dtype} {array.shape}')
        count = len(array) if count is None else count
        if len(array) != count: raise ValueError('Particle field lengths differ')
    if count > header.capacity: raise ValueError('Frame holds more particles than capacity')
    if frame.wetness.dtype != np.float32 or frame.wetness.shape != (header.field_nodes,):
        raise ValueError('Wetness does not match the field node count')
    if frame.counters.dtype != np.int64 or frame.counters.shape != (4,):
        raise ValueError('Counters must be four int64 values')
    if frame.ledger.dtype != np.float64 or frame.ledger.shape != (2,):
        raise ValueError('Ledger must be two float64 values')
    _check_arrays({**frame.arrays, 'wetness': frame.wetness, 'ledger': frame.ledger})


def _frame_arrays(frame: CachedFrame):
    return {**{f'p_{k}': v for k, v in frame.arrays.items()}, 'wetness': frame.wetness,
            'counters': frame.counters, 'ledger': frame.ledger,
            'meta': np.array([frame.frame, frame.next_id], dtype=np.int64)}


class CacheWriter:
    def __init__(self, path: Path, header: CacheHeader):
        self.path = Path(path)
        if self.path.exists(): raise FileExistsError(f'Cache destination already exists: {self.path}')
        if header.schema_version != SCHEMA_VERSION: raise ValueError('Unsupported cache schema')
        if header.frame_count < 1 or header.capacity < 1 or header.field_nodes < 0:
            raise ValueError('Invalid cache header')
        self.header = header
        self.path.mkdir(parents=True)
        self.manifest = dict(schema_version=SCHEMA_VERSION, status='writing', world_units='meters',
                             header=asdict(header)|{'physical_settings': _canonical(header.physical_settings)},
                             static=None, frames={})
        self._save_manifest()

    def _store(self, name, arrays):
        _check_arrays(arrays)
        buffer = BytesIO(); np.savez(buffer, **arrays); data = buffer.getvalue()
        temporary = self.path/(name+'.tmp')
        temporary.write_bytes(data); os.replace(temporary, self.path/name)
        return dict(file=name, sha256=sha256(data).hexdigest(), bytes=len(data),
                    arrays={k: [v.dtype.str, list(v.shape)] for k, v in arrays.items()})

    def _save_manifest(self):
        temporary = self.path/'manifest.json.tmp'
        temporary.write_text(json.dumps(self.manifest, indent=1), encoding='utf8')
        os.replace(temporary, self.path/'manifest.json')

    def write_static(self, arrays: dict):
        if self.manifest['static'] is not None: raise ValueError('Static data is already written')
        self.manifest['static'] = self._store('static.npz', arrays); self._save_manifest()

    def write(self, frame: CachedFrame):
        if self.manifest['status'] != 'writing': raise RuntimeError('Cache is closed')
        expected = self.header.start_frame+len(self.manifest['frames'])
        if frame.frame != expected: raise ValueError(f'Expected frame {expected}, got {frame.frame}')
        _validate_frame(frame, self.header)
        self.manifest['frames'][str(frame.frame)] = self._store(f'frame_{frame.frame:07d}.npz', _frame_arrays(frame))
        self._save_manifest()

    def finish(self):
        if self.manifest['static'] is None: raise ValueError('Static data is missing')
        if len(self.manifest['frames']) != self.header.frame_count: raise ValueError('Frames are missing')
        for record in [self.manifest['static'], *self.manifest['frames'].values()]:
            data = (self.path/record['file']).read_bytes()
            if len(data) != record['bytes'] or sha256(data).hexdigest() != record['sha256']:
                raise ValueError(f'{record["file"]} changed on disk')
        self.manifest['status'] = 'complete'; self._save_manifest()

    def cancel(self):
        if self.manifest['status'] == 'writing':
            self.manifest['status'] = 'cancelled'; self._save_manifest()


class CacheReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        manifest_path = self.path/'manifest.json'
        if not manifest_path.is_file(): raise ValueError('Cache manifest is missing')
        if manifest_path.stat().st_size > _MANIFEST_LIMIT: raise ValueError('Cache manifest is too large')
        try: manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error: raise ValueError('Cache manifest is corrupt') from error
        if manifest.get('schema_version') != SCHEMA_VERSION: raise ValueError('Unsupported cache schema')
        if manifest.get('status') != 'complete': raise ValueError(f"Cache is not complete ({manifest.get('status')})")
        try: self.header = CacheHeader(**manifest['header'])
        except (TypeError, KeyError) as error: raise ValueError('Cache header is invalid') from error
        frames = manifest.get('frames') or {}
        wanted = {str(f) for f in range(self.header.start_frame, self.header.end_frame+1)}
        if set(frames) != wanted or not manifest.get('static'): raise ValueError('Cache frames are incomplete')
        self.manifest = manifest

    def validate(self, source_fingerprint: str, physical_settings: dict):
        if source_fingerprint != self.header.source_fingerprint:
            raise ValueError('Cache was baked from different source geometry')
        if _canonical(physical_settings) != _canonical(self.header.physical_settings):
            raise ValueError('Cache was baked with different physical settings')

    def _load(self, record, bound):
        name = record['file']
        if Path(name).name != name: raise ValueError('Invalid cache file name')
        path = self.path/name
        if not path.is_file(): raise ValueError(f'{name} is missing')
        size = path.stat().st_size
        # Bound allocation by the header before reading anything.
        if size != record['bytes'] or size > bound: raise ValueError(f'{name} has an unexpected size')
        data = path.read_bytes()
        if sha256(data).hexdigest() != record['sha256']: raise ValueError(f'{name} checksum mismatch')
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        if {k: [v.dtype.str, list(v.shape)] for k, v in arrays.items()} != record['arrays']:
            raise ValueError(f'{name} arrays do not match the manifest')
        _check_arrays(arrays)
        return arrays

    def read_static(self) -> dict:
        return self._load(self.manifest['static'], _STATIC_LIMIT)

    def read(self, frame: int) -> CachedFrame:
        record = self.manifest['frames'].get(str(frame))
        if record is None: raise ValueError(f'Frame {frame} is outside the cache')
        arrays = self._load(record, _frame_bound(self.header))
        meta = arrays.pop('meta')
        if meta.shape != (2,) or int(meta[0]) != frame: raise ValueError('Frame number mismatch')
        result = CachedFrame(frame, {k[2:]: v for k, v in arrays.items() if k.startswith('p_')},
                             arrays['wetness'], arrays['counters'], arrays['ledger'], int(meta[1]))
        _validate_frame(result, self.header)
        return result
