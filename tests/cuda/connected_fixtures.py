import numpy as np
from math import pi
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.state import ParticlePool


def make_fixture(vertices,triangles,positions,states=None,volumes=None,velocities=None,config=None):
    cfg=config or FlowConfig(capacity=len(positions),gravity=(0,0,-1),source_start=0,source_softness=0)
    device=require_cuda()
    source=build_source(vertices,triangles,[0]*len(triangles),cfg,device)
    pool=ParticlePool(cfg,device)
    count=len(positions)
    pool.data.active.assign([1]*count+[0]*(cfg.capacity-count))
    pool.data.position.assign(list(positions)+[[0,0,0]]*(cfg.capacity-count))
    pool.data.ids.assign(list(range(cfg.capacity)))
    pool.data.state.assign((states or [0]*count)+[0]*(cfg.capacity-count))
    pool.data.volume.assign((volumes or [4*pi/3*cfg.radius**3]*count)+[0]*(cfg.capacity-count))
    pool.data.velocity.assign((velocities or [[0,0,0]]*count)+[[0,0,0]]*(cfg.capacity-count))
    points=np.asarray(vertices)[triangles[0]]
    normal=np.cross(points[1]-points[0],points[2]-points[0]); normal=normal/np.linalg.norm(normal)
    pool.data.normal.assign([normal.tolist()]*cfg.capacity)
    pool.ledger.assign([float(pool.data.volume.numpy().sum(dtype=np.float64)),0])
    return source,pool,device
