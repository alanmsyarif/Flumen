"""Integer-frame coordinator. Emission occurs once after each completed interval."""
from time import perf_counter
import warp as wp
from .config import frame_dt
from .state import ParticlePool, read_stats, snapshot
from .emission import emit
from .motion import advance


class FlowSolver:
    def __init__(self, config, source, device, start_frame=1, fps=30, fps_base=1.0):
        config.validate()
        if not isinstance(start_frame,int):
            raise ValueError('Simulation start must be an integer frame')
        if not wp.get_device(device.alias).is_cuda:
            raise ValueError('GPU Flow requires CUDA')
        self.config,self.source,self.device = config,source,device
        self.start_frame = start_frame
        self.dt = frame_dt(fps,fps_base)
        self.pool = ParticlePool(config,device)
        self.current_frame = None
        self.stats = None

    def seek(self, frame: int):
        if isinstance(frame,bool) or not isinstance(frame,int):
            raise ValueError('GPU Flow supports integer frames only')
        if self.pool is None:
            raise RuntimeError('GPU runtime is closed')
        if frame == self.current_frame:
            return self.stats
        if self.current_frame is not None and frame < self.current_frame:
            self.reset()
        begin = self.start_frame if self.current_frame is None else max(self.start_frame,self.current_frame+1)
        start = perf_counter()
        for f in range(begin,frame+1):
            if f > self.start_frame:
                advance(self.pool,self.source,self.config,self.dt)
            emit(self.pool,self.source,self.config,f)
        wp.synchronize_device(self.device.alias)
        self.pool.solver_ms = (perf_counter()-start)*1000
        self.current_frame = frame
        self.stats = read_stats(self.pool,frame)
        return self.stats

    def snapshot(self):
        if self.pool is None:
            raise RuntimeError('GPU runtime is closed')
        start = perf_counter()
        batch = snapshot(self.pool)
        self.pool.transfer_ms = (perf_counter()-start)*1000
        if self.stats is not None:
            self.stats.transfer_ms = self.pool.transfer_ms
        return batch

    def reset(self):
        if self.pool is not None:
            self.pool.close()
        self.pool = ParticlePool(self.config,self.device)
        self.current_frame = self.stats = None

    def close(self):
        if self.pool is not None:
            self.pool.close()
        if self.source is not None:
            self.source.close()
        self.pool = self.source = self.stats = None
