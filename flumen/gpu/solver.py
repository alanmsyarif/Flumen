"""Integer-frame coordinator. Emission occurs once after each completed interval."""
from time import perf_counter
import warp as wp
from .config import frame_dt
from .state import ParticlePool, read_stats, snapshot, point_snapshot
from .emission import emit, emit_batch
from .motion import advance, advance_coupled


def owned_array_bytes(*roots) -> int:
    """CUDA wp.array bytes reachable from Flumen objects; BVH and graph storage excluded."""
    seen=set(); total=0; stack=list(roots)
    while stack:
        value=stack.pop()
        if isinstance(value,wp.array):
            if value.device.is_cuda and value.ptr and value.ptr not in seen:
                seen.add(value.ptr); total+=value.capacity
        elif isinstance(value,(list,tuple)):
            stack.extend(value)
        elif id(value) not in seen and any(c.__module__.startswith('flumen') for c in type(value).__mro__):
            seen.add(id(value)); stack.extend(vars(value).values())
    return total


class FlowSolver:
    def __init__(self, config, source, device, start_frame=1, fps=30, fps_base=1.0, *, prepared=None):
        config.validate()
        if not isinstance(start_frame,int):
            raise ValueError('Simulation start must be an integer frame')
        if not wp.get_device(device.alias).is_cuda:
            raise ValueError('GPU Flow requires CUDA')
        self.config,self.source,self.device = config,source,device
        self.start_frame = start_frame
        self.dt = frame_dt(fps,fps_base)*config.time_scale
        self.pool = ParticlePool(config,device)
        self.prepared = None
        self.field = None
        self._field_step = None
        self.topology=self.interaction=self.surface=self.geometry_buffers=self.free_buffers=None
        self._water_cache=None
        try:
            if prepared is not None:
                if prepared.source is not source:
                    raise ValueError('Prepared contacts belong to a different source')
                self.prepared = prepared.retain()
            if config.solver_backend=='FIELD':
                if self.prepared is None:
                    from hashlib import sha256
                    from .prepared import prepare_source
                    digest=sha256(source.vertices_cpu.tobytes()+source.triangles_cpu.tobytes()+source.islands_cpu.tobytes()).hexdigest()
                    self.prepared=prepare_source(source,digest,config.field_spacing,config.contact_spacing)
                from .field_solver import FieldBuffers
                self.field=FieldBuffers(self.prepared.chart,device.alias)
            elif config.interactions_enabled or config.display_mode=='CONNECTED':
                from .topology import build_topology
                self.topology=build_topology(source,config.radius*config.interaction_radius_scale,
                    1.5*config.radius*config.reconstruction_scale)
            if config.interactions_enabled and config.solver_backend=='LEGACY':
                from .interaction import InteractionBuffers
                self.interaction=InteractionBuffers(config.capacity,device.alias)
            if config.display_mode=='CONNECTED':
                from .surface import build_surface
                self.surface=build_surface(self.topology,config,device)
                from .surface_mesh import GeometryBuffers
                self.geometry_buffers=GeometryBuffers(len(self.topology.triangles),device.alias)
                from .free_mesh import FreeMeshBuffers
                self.free_buffers=FreeMeshBuffers(config.capacity,device.alias,self.geometry_buffers)
        except Exception:
            self.pool.close()
            if self.interaction is not None: self.interaction.close()
            if self.surface is not None: self.surface.close()
            if self.geometry_buffers is not None: self.geometry_buffers.close()
            if self.free_buffers is not None: self.free_buffers.close()
            if self.topology is not None: self.topology.close()
            if self.prepared is not None: self.prepared.release()
            if self.field is not None: self.field.close()
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
        stream = wp.get_stream(self.device.alias); events = []; field_ms = 0.
        def timed(name, function, *args, **kwargs):
            # CUDA events time queued GPU work without adding host synchronization.
            pair = (wp.Event(self.device.alias,enable_timing=True),wp.Event(self.device.alias,enable_timing=True))
            stream.record_event(pair[0]); result = function(*args,**kwargs); stream.record_event(pair[1])
            events.append((name,*pair)); return result
        for f in range(begin,frame+1):
            self.source.refresh_sampling(self.config)
            if f > self.start_frame:
                if self.surface is not None:
                    from .surface import advance_wetness
                    advance_wetness(self.surface,self.config,self.dt)
                if self.field is not None:
                    from .field_motion import advance_field_particles
                    from .field_aggregate import aggregate_field_particles
                    self._field_step=timed('advance',advance_field_particles,self.pool,self.prepared,self.field,
                                           self.config,self.dt,reuse_deposit=True)
                    field_ms+=self._field_step.field_ms
                    timed('aggregation',aggregate_field_particles,self.pool,self.prepared,self.config)
                elif self.interaction is None:
                    advance(self.pool,self.source,self.config,self.dt)
                else:
                    advance_coupled(self.pool,self.source,self.topology,self.config,self.interaction,self.dt)
            else:
                emit_batch(self.pool,self.source,self.config,f,self.config.initial_coating_count)
            emit(self.pool,self.source,self.config,f)
            if self.field is not None:
                from .field_solver import deposit_attached
                from .field_aggregate import resample_particles
                timed('resample',resample_particles,self.pool,self.prepared,self.config.resample_target)
                timed('deposit',deposit_attached,self.pool,self.prepared,self.field)
            if self.surface is not None:
                from .surface import update_surface
                update_surface(self.surface,self.pool,self.source,self.config,0)
            self.current_frame=f
        wp.synchronize_device(self.device.alias)
        self.pool.solver_ms = (perf_counter()-start)*1000
        self.current_frame = frame
        self._water_cache=None
        self.stats = read_stats(self.pool,frame)
        self.stats.backend=self.config.solver_backend
        if self.field is not None:
            self.stats.field_nodes=len(self.prepared.chart.vertices)
            self.stats.contact_samples=self.prepared.contact.sample_count
            self.stats.coarsening_factor=self.prepared.chart.coarsening_factor
            self.stats.unrepresented_volume=float(self.field.unsupported.numpy()[0])
            # Stage times sum over every interval this seek processed, like solver_ms.
            stages = dict.fromkeys(('advance','aggregation','resample','deposit'),0.)
            for name,begin_event,end_event in events:
                stages[name]+=wp.get_event_elapsed_time(begin_event,end_event,synchronize=False)
            self.stats.field_ms=field_ms
            if self._field_step is not None: self.stats.field_courant=self._field_step.courant
            self.stats.contact_ms=max(0.,stages['advance']-field_ms)
            self.stats.aggregation_ms=stages['aggregation']
            self.stats.resample_ms=stages['resample']
            self.stats.deposit_ms=stages['deposit']
            self.stats.owned_array_bytes=owned_array_bytes(self.pool,self.field,self.prepared)
            if self.pool.field_motion is not None:
                self.stats.contact_fallback_count=self.pool.field_motion.last_fallback_count
            if self.pool.field_aggregate is not None:
                self.stats.merged_pairs=int(self.pool.field_aggregate.merged.numpy()[0])
                self.stats.resampled_count=int(self.pool.field_aggregate.resampled.numpy()[0])
        if self.interaction is not None:
            self.stats.interaction_ms=self.interaction.timing_ms()
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

    def point_snapshot(self, limit=None):
        """Read-only xyz+radius stream; never seeks, emits or deposits."""
        if self.pool is None:
            raise RuntimeError('GPU runtime is closed')
        start = perf_counter()
        batch = point_snapshot(self.pool,limit,self.current_frame)
        if self.stats is not None:
            self.stats.readback_ms = (perf_counter()-start)*1000
            self.stats.displayed_count = batch.displayed_count
        return batch

    def cache_snapshot(self):
        """Exact host copy of every active particle and the frame's accounting, for baking."""
        import numpy as np
        from ..particle_cache import CachedFrame, PARTICLE_FIELDS
        if self.pool is None or self.current_frame is None:
            raise RuntimeError('GPU runtime has no evaluated frame')
        data=self.pool.data
        slots=np.flatnonzero(data.active.numpy()==1).astype(np.int32)
        arrays={'slot':slots}
        for name,(dtype,_) in PARTICLE_FIELDS.items():
            if name!='slot': arrays[name]=getattr(data,name).numpy()[slots].astype(dtype)
        wetness=self.field.wetness.numpy().astype(np.float32) if self.field is not None else np.zeros(0,np.float32)
        return CachedFrame(self.current_frame,arrays,wetness,self.pool.counters.numpy().astype(np.int64),
                           self.pool.ledger.numpy().astype(np.float64),int(self.pool.next_id))

    def reset(self):
        if self.pool is not None:
            self.pool.close()
        self.pool = ParticlePool(self.config,self.device)
        if self.surface is not None: self.surface.reset()
        if self.interaction is not None: self.interaction.timing_count=0
        if self.field is not None:
            self.field.wetness.zero_(); self.field.velocity.zero_()
        self._field_step=None
        self.current_frame = self.stats = None
        self._water_cache=None

    def close(self):
        if self.pool is not None:
            self.pool.close()
        if self.field is not None:
            self.field.close(); self.field=None
        if self.source is not None:
            if self.prepared is not None:
                self.prepared.release()
                self.prepared = None
            else:
                self.source.close()
        if self.interaction is not None: self.interaction.close()
        if self.surface is not None: self.surface.close()
        if self.geometry_buffers is not None: self.geometry_buffers.close()
        if self.free_buffers is not None: self.free_buffers.close()
        if self.topology is not None: self.topology.close()
        self.interaction=self.topology=self.surface=self.geometry_buffers=self.free_buffers=None
        self._water_cache=None
        self.pool = self.source = self.stats = None

    def surface_snapshot(self):
        if self.surface is None: return None
        from .surface import snapshot_surface
        start=perf_counter()
        result=snapshot_surface(self.surface)
        if self.stats is not None:
            self.stats.transfer_ms=self.geometry_buffers.transfer_ms+(perf_counter()-start)*1000
        return result

    def water_snapshot(self):
        if self.surface is None: return None
        if self._water_cache is None:
            from .state import WaterGeometry
            from .surface_mesh import build_attached_mesh
            from .free_mesh import build_free_mesh
            start=perf_counter()
            self.geometry_buffers.transfer_ms=0.
            attached=build_attached_mesh(self.topology,self.surface,self.config,self.geometry_buffers)
            free=build_free_mesh(self.pool,self.source,self.config,
                self.geometry_buffers.vertex_budget-len(attached.vertices),
                self.geometry_buffers.triangle_budget-len(attached.triangles),self.free_buffers)
            represented=attached.diagnostics.get('represented_volume',0.)+free.diagnostics.get('represented_volume',0.)
            volume=attached.diagnostics.get('mesh_volume',0.)+free.diagnostics.get('mesh_volume',0.)
            diagnostics=dict(rendered_volume_error=abs(volume-represented)/max(represented,1.e-20),
                coarsening_factor=max(attached.diagnostics.get('coarsening_factor',1.),free.diagnostics.get('coarsening_factor',1.)),
                volumetric_samples=free.diagnostics.get('volumetric_samples',0),
                unrepresented_volume=float(self.surface.summary.numpy()[1])+attached.diagnostics.get('unrepresented_volume',0.)+free.diagnostics.get('unrepresented_volume',0.),
                excluded_volume=attached.diagnostics.get('excluded_volume',0.))
            errors=[m.diagnostics['error'] for m in (attached,free) if 'error' in m.diagnostics]
            if errors: diagnostics['error']='; '.join(errors)
            self._water_cache=WaterGeometry(attached,free,diagnostics)
            if self.stats is not None:
                self.stats.reconstruction_ms=(perf_counter()-start)*1000
                self.stats.water_vertices=len(attached.vertices)+len(free.vertices)
                self.stats.water_triangles=len(attached.triangles)+len(free.triangles)
                self.stats.rendered_volume_error=diagnostics['rendered_volume_error']
                self.stats.coarsening_factor=diagnostics['coarsening_factor']
                self.stats.unrepresented_volume=diagnostics['unrepresented_volume']
        return self._water_cache
