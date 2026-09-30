"""Fixed-capacity CUDA storage and explicit diagnostics/readback."""
from dataclasses import dataclass
from math import pi
import numpy as np
import warp as wp
from .source import Samples


@wp.struct
class ParticleArrays:
    position: wp.array(dtype=wp.vec3)
    velocity: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    age: wp.array(dtype=float)
    volume: wp.array(dtype=float)
    active: wp.array(dtype=int)
    state: wp.array(dtype=int)
    island: wp.array(dtype=int)
    face: wp.array(dtype=int)
    bary: wp.array(dtype=wp.vec2)
    ids: wp.array(dtype=wp.int64)
    path: wp.array(dtype=wp.int64)
    limited: wp.array(dtype=int)


@dataclass
class DisplayBatch:
    positions: np.ndarray
    radii: np.ndarray
    ids: np.ndarray


@dataclass
class FrameStats:
    frame: int
    live_count: int
    requested: int
    accepted: int
    capacity_rejected: int
    source_rejected: int
    emitted_volume: float
    live_volume: float
    removed_volume: float
    substeps: int
    limited_count: int
    solver_ms: float = 0.0
    transfer_ms: float = 0.0


class ParticlePool:
    def __init__(self, config, device):
        config.validate()
        self.capacity, self.device = config.capacity, device.alias
        self.data = ParticleArrays()
        for name, dtype in [('position',wp.vec3),('velocity',wp.vec3),('normal',wp.vec3),
            ('age',float),('volume',float),('active',int),('state',int),('island',int),
            ('face',int),('bary',wp.vec2),('ids',wp.int64),('path',wp.int64),('limited',int)]:
            setattr(self.data,name,wp.zeros(self.capacity,dtype=dtype,device=self.device))
        self.mask = wp.zeros(self.capacity,dtype=int,device=self.device)
        self.prefix = wp.zeros(self.capacity,dtype=int,device=self.device)
        self.candidate_ids = wp.zeros(self.capacity,dtype=wp.int64,device=self.device)
        self.samples = Samples(self.capacity,self.device)
        self.counters = wp.zeros(4,dtype=wp.int64,device=self.device)
        self.ledger = wp.zeros(2,dtype=wp.float64,device=self.device)
        self.step_count = wp.zeros(1,dtype=int,device=self.device)
        self.next_id = 0
        self.substeps = 0
        self.solver_ms = self.transfer_ms = 0.0

    def close(self):
        self.data = self.samples = self.mask = self.prefix = self.candidate_ids = None
        self.counters = self.ledger = self.step_count = None


def read_stats(pool, frame) -> FrameStats:
    counts = pool.counters.numpy()
    ledger = pool.ledger.numpy()
    active = pool.data.active.numpy().astype(bool)
    volume = pool.data.volume.numpy()[active].sum(dtype=np.float64)
    return FrameStats(frame,int(active.sum()),*(int(x) for x in counts),float(ledger[0]),
        float(volume),float(ledger[1]),int(pool.step_count.numpy()[0]),
        int(pool.data.limited.numpy()[active].sum()),pool.solver_ms,pool.transfer_ms)


def snapshot(pool) -> DisplayBatch:
    active = pool.data.active.numpy().astype(bool)
    radius = np.cbrt(pool.data.volume.numpy()[active]*(3.0/(4*pi)))
    positions = pool.data.position.numpy()[active]
    attached = pool.data.state.numpy()[active] == 0
    positions[attached] += pool.data.normal.numpy()[active][attached]*radius[attached,None]
    return DisplayBatch(np.ascontiguousarray(positions),np.ascontiguousarray(radius),
                        np.ascontiguousarray(pool.data.ids.numpy()[active]))
