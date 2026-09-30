"""Fixed-capacity CUDA storage and explicit diagnostics/readback."""
from dataclasses import dataclass
from math import pi
import numpy as np
import warp as wp
from warp.utils import array_scan
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
    normals: np.ndarray | None = None
    velocities: np.ndarray | None = None
    volumes: np.ndarray | None = None
    states: np.ndarray | None = None
    faces: np.ndarray | None = None
    islands: np.ndarray | None = None


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
        self.summary = wp.zeros(3,dtype=wp.float64,device=self.device)
        self.read_count = wp.zeros(1,dtype=int,device=self.device)
        self.display = wp.zeros(self.capacity,dtype=wp.vec4,device=self.device)
        self.display_ids = wp.zeros(self.capacity,dtype=wp.int64,device=self.device)
        self.host_display = wp.zeros(self.capacity,dtype=wp.vec4,device='cpu',pinned=True)
        self.host_ids = wp.zeros(self.capacity,dtype=wp.int64,device='cpu',pinned=True)
        self.display_aux={}
        self.host_aux={}
        for name,dtype in [('normals',wp.vec3),('velocities',wp.vec3),('volumes',float),
                           ('states',int),('faces',int),('islands',int)]:
            self.display_aux[name]=wp.zeros(self.capacity,dtype=dtype,device=self.device)
            self.host_aux[name]=wp.zeros(self.capacity,dtype=dtype,device='cpu',pinned=True)
        self.next_id = 0
        self.substeps = 0
        self.solver_ms = self.transfer_ms = 0.0

    def close(self):
        self.data = self.samples = self.mask = self.prefix = self.candidate_ids = None
        self.counters = self.ledger = self.step_count = None
        self.summary = self.read_count = self.display = self.display_ids = None
        self.host_display = self.host_ids = None
        self.display_aux.clear(); self.host_aux.clear()


@wp.kernel
def summarize_kernel(d: ParticleArrays, summary: wp.array(dtype=wp.float64)):
    i=wp.tid()
    if d.active[i] == 1:
        wp.atomic_add(summary,0,wp.float64(1.0))
        wp.atomic_add(summary,1,wp.float64(d.volume[i]))
        wp.atomic_add(summary,2,wp.float64(d.limited[i]))


@wp.kernel
def gather_kernel(d: ParticleArrays, prefix: wp.array(dtype=int), display: wp.array(dtype=wp.vec4),
                  ids: wp.array(dtype=wp.int64), count: wp.array(dtype=int),
                  normals:wp.array(dtype=wp.vec3),velocities:wp.array(dtype=wp.vec3),
                  volumes:wp.array(dtype=float),states:wp.array(dtype=int),
                  faces:wp.array(dtype=int),islands:wp.array(dtype=int)):
    i=wp.tid()
    if i == 0:
        last=d.active.shape[0]-1
        count[0]=prefix[last]+d.active[last]
    if d.active[i] == 1:
        radius=wp.pow(d.volume[i]*0.238732414637843,1.0/3.0)
        p=d.position[i]
        if d.state[i] == 0: p+=d.normal[i]*radius
        display[prefix[i]]=wp.vec4(p[0],p[1],p[2],radius)
        ids[prefix[i]]=d.ids[i]
        normals[prefix[i]]=d.normal[i]
        velocities[prefix[i]]=d.velocity[i]
        volumes[prefix[i]]=d.volume[i]
        states[prefix[i]]=d.state[i]
        faces[prefix[i]]=d.face[i]
        islands[prefix[i]]=d.island[i]


def read_stats(pool, frame) -> FrameStats:
    pool.summary.zero_()
    wp.launch(summarize_kernel,pool.capacity,inputs=[pool.data,pool.summary],device=pool.device)
    counts = pool.counters.numpy()
    ledger = pool.ledger.numpy()
    summary = pool.summary.numpy()
    return FrameStats(frame,int(summary[0]),*(int(x) for x in counts),float(ledger[0]),
        float(summary[1]),float(ledger[1]),int(pool.step_count.numpy()[0]),
        int(summary[2]),pool.solver_ms,pool.transfer_ms)


def snapshot(pool) -> DisplayBatch:
    array_scan(pool.data.active,pool.prefix,inclusive=False)
    wp.launch(gather_kernel,pool.capacity,inputs=[pool.data,pool.prefix,pool.display,
              pool.display_ids,pool.read_count,*pool.display_aux.values()],device=pool.device)
    count=int(pool.read_count.numpy()[0])
    if count:
        wp.copy(pool.host_display,pool.display,count=count)
        wp.copy(pool.host_ids,pool.display_ids,count=count)
        for name,array in pool.display_aux.items():
            wp.copy(pool.host_aux[name],array,count=count)
        wp.synchronize_device(pool.device)
    display=pool.host_display.numpy()[:count]
    return DisplayBatch(np.ascontiguousarray(display[:,:3]),np.ascontiguousarray(display[:,3]),
                        pool.host_ids.numpy()[:count].copy(),
                        *(array.numpy()[:count].copy() for array in pool.host_aux.values()))
