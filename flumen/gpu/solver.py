"""Integer-frame coordinator. Emission occurs once after each completed interval."""
from time import perf_counter
import warp as wp
from .config import frame_dt
from .state import ParticlePool, read_stats, snapshot
from .emission import emit, emit_batch
from .motion import advance, advance_coupled


class FlowSolver:
    def __init__(self, config, source, device, start_frame=1, fps=30, fps_base=1.0):
        config.validate()
        if not isinstance(start_frame,int):
            raise ValueError('Simulation start must be an integer frame')
        if not wp.get_device(device.alias).is_cuda:
            raise ValueError('GPU Flow requires CUDA')
        self.config,self.source,self.device = config,source,device
        self.start_frame = start_frame
        self.dt = frame_dt(fps,fps_base)*config.time_scale
        self.pool = ParticlePool(config,device)
        self.topology=self.interaction=self.surface=None
        try:
            if config.interactions_enabled or config.display_mode=='CONNECTED':
                from .topology import build_topology
                self.topology=build_topology(source,config.radius*config.interaction_radius_scale,
                    1.5*config.radius*config.reconstruction_scale)
            if config.interactions_enabled:
                from .interaction import InteractionBuffers
                self.interaction=InteractionBuffers(config.capacity,device.alias)
            if config.display_mode=='CONNECTED':
                from .surface import build_surface
                self.surface=build_surface(self.topology,config,device)
        except Exception:
            self.pool.close()
            if self.interaction is not None: self.interaction.close()
            if self.topology is not None: self.topology.close()
            raise
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
                if self.surface is not None:
                    from .surface import advance_wetness
                    advance_wetness(self.surface,self.config,self.dt)
                if self.interaction is None:
                    advance(self.pool,self.source,self.config,self.dt)
                else:
                    advance_coupled(self.pool,self.source,self.topology,self.config,self.interaction,self.dt)
            else:
                emit_batch(self.pool,self.source,self.config,f,self.config.initial_coating_count)
            emit(self.pool,self.source,self.config,f)
            if self.surface is not None:
                from .surface import update_surface
                update_surface(self.surface,self.pool,self.source,self.config,0)
        wp.synchronize_device(self.device.alias)
        self.pool.solver_ms = (perf_counter()-start)*1000
        self.current_frame = frame
        self.stats = read_stats(self.pool,frame)
        if self.surface is not None:
            summary=self.surface.summary.numpy()
            self.stats.proxy_vertices=len(self.topology.vertices)
            self.stats.coarsening_factor=self.topology.coarsening_factor
            self.stats.unrepresented_volume=float(summary[1])
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
        if self.surface is not None: self.surface.reset()
        self.current_frame = self.stats = None

    def close(self):
        if self.pool is not None:
            self.pool.close()
        if self.source is not None:
            self.source.close()
        if self.interaction is not None: self.interaction.close()
        if self.surface is not None: self.surface.close()
        if self.topology is not None: self.topology.close()
        self.interaction=self.topology=self.surface=None
        self.pool = self.source = self.stats = None

    def surface_snapshot(self):
        if self.surface is None: return None
        from .surface import snapshot_surface
        return snapshot_surface(self.surface)
