"""Explicit particle-cache bakes for Surface Field hosts. Live editing never writes caches."""
from pathlib import Path
import numpy as np
import bpy
from .particle_cache import CacheHeader, CacheWriter, SCHEMA_VERSION, PARTICLE_ROW_BYTES, estimate_cache_bytes
from .gpu.config import physical_key


def physical_settings(config) -> dict:
    """Settings that change particle motion; materials and mesh quality are excluded."""
    return {name:(list(value) if isinstance(value,tuple) else value) for name,value in physical_key(config)}


def _surface_attributes(obj, kept):
    """Per-corner UVs and material indices aligned with the solver's evaluated triangles."""
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        count=len(mesh.loop_triangles)
        loops=np.empty(count*3,np.int32); mesh.loop_triangles.foreach_get('loops',loops); loops=loops.reshape(-1,3)
        materials=np.empty(count,np.int32); mesh.loop_triangles.foreach_get('material_index',materials)
        uv=np.zeros((len(mesh.loops),2),np.float32)
        if mesh.uv_layers.active is not None:
            mesh.uv_layers.active.data.foreach_get('uv',uv.reshape(-1))
        # Match extract_source: mirrored transforms reverse triangle winding.
        if np.linalg.det(np.asarray(evaluated.matrix_world,dtype=np.float64)[:3,:3]) < 0: loops=loops[:,[0,2,1]]
        return dict(corner_uv=np.ascontiguousarray(uv[loops][kept]),material_index=materials[kept])
    finally:
        evaluated.to_mesh_clear()


def _static_arrays(host, solver):
    source,chart=solver.source,solver.prepared.chart
    arrays=dict(source_vertices=source.vertices_cpu.astype(np.float32),source_triangles=source.triangles_cpu.astype(np.int32),
        source_islands=source.islands_cpu.astype(np.int32),evaluated_triangle_ids=source.evaluated_triangle_ids.astype(np.int32),
        chart_vertices=np.asarray(chart.vertices),chart_triangles=np.asarray(chart.triangles,np.int32),
        chart_areas=np.asarray(chart.areas),chart_normals=np.asarray(chart.normals),
        chart_original_faces=np.asarray(chart.original_faces,np.int32),chart_level=np.array([chart.level],np.int32))
    arrays.update(_surface_attributes(host.flumen_gpu.source,source.evaluated_triangle_ids))
    return arrays


def _record(host, scene):
    from .gpu_runtime import get_runtime, RUNTIMES
    solver=get_runtime(host,scene)
    if solver.config.solver_backend!='FIELD' or solver.prepared is None:
        raise ValueError('Particle caches bake Surface Field hosts only')
    return RUNTIMES[host.as_pointer()]


class BakeJob:
    """Bakes every integer frame with its own solver; the live host's state is untouched."""
    def __init__(self, host, scene, path):
        from .gpu.solver import FlowSolver
        record=_record(host,scene); live=record.solver
        self.host,self.path=host,Path(path)
        self.solver=self.writer=None
        self.solver=FlowSolver(live.config,live.prepared.source,live.device,start_frame=scene.frame_start,
                               fps=scene.render.fps,fps_base=scene.render.fps_base,prepared=live.prepared)
        try:
            self.header=CacheHeader(SCHEMA_VERSION,record.geometry_hash,physical_settings(live.config),self.solver.dt,
                scene.frame_start,scene.frame_end,live.config.capacity,live.prepared.fingerprint,
                len(live.prepared.chart.vertices))
            static=_static_arrays(host,self.solver)
            self.estimated_bytes=estimate_cache_bytes(self.header,sum(a.nbytes for a in static.values()))
            self.writer=CacheWriter(self.path,self.header)
            self.writer.write_static(static)
        except BaseException:
            self.cancel(); raise
        self.frame=self.header.start_frame

    @property
    def progress(self):
        return (self.frame-self.header.start_frame)/self.header.frame_count

    def step(self) -> bool:
        """Bake one frame; True once the cache is complete. One frame is held in memory."""
        try:
            self.solver.seek(self.frame)
            self.writer.write(self.solver.cache_snapshot())
            self.frame+=1
            if self.frame<=self.header.end_frame: return False
            self.writer.finish()
        except BaseException:
            self.cancel(); raise
        self.host['sf_particle_cache']=str(self.path)
        self._release()
        return True

    def cancel(self):
        if self.writer is not None: self.writer.cancel()
        self._release()

    def _release(self):
        if self.solver is not None: self.solver.close()
        self.solver=None


class SF_OT_bake_particle_cache(bpy.types.Operator):
    bl_idname='flumen.bake_particle_cache'
    bl_label='Bake Particle Cache'
    bl_description='Bake every scene frame of this Surface Field host to a new, validated cache folder'

    directory: bpy.props.StringProperty(name='Parent Folder',subtype='DIR_PATH')
    cache_name: bpy.props.StringProperty(name='Cache Name',default='flumen_particles')

    _job=None
    _timer=None

    @classmethod
    def poll(cls,context):
        host=context.active_object
        return host is not None and host.get('sf_gpu_host') and host.flumen_gpu.solver_backend=='FIELD'

    def invoke(self,context,event):
        scene=context.scene; host=context.active_object
        frames=scene.frame_end-scene.frame_start+1
        # Upper bound shown before baking: full capacity rows for every frame.
        self.estimate=frames*host.flumen_gpu.capacity*PARTICLE_ROW_BYTES/1024**3
        return context.window_manager.invoke_props_dialog(self)

    def draw(self,context):
        layout=self.layout
        layout.prop(self,'directory'); layout.prop(self,'cache_name')
        layout.label(text=f'Up to {self.estimate:.2f} GB on disk; the folder must not exist',icon='INFO')

    def execute(self,context):
        if not self.directory or not self.cache_name or Path(self.cache_name).name!=self.cache_name:
            self.report({'ERROR'},'Choose a parent folder and a plain cache name'); return {'CANCELLED'}
        path=Path(bpy.path.abspath(self.directory))/self.cache_name
        try:
            self._job=BakeJob(context.active_object,context.scene,path)
        except (ValueError,RuntimeError,OSError) as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        wm=context.window_manager
        self._timer=wm.event_timer_add(.001,window=context.window)
        wm.progress_begin(0,1); wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self,context,event):
        if event.type=='ESC':
            self._job.cancel(); self._end(context)
            self.report({'WARNING'},'Bake cancelled; the partial cache is marked incomplete'); return {'CANCELLED'}
        if event.type!='TIMER': return {'PASS_THROUGH'}
        try:
            done=self._job.step()
        except (ValueError,RuntimeError,OSError) as exc:
            self._end(context); self.report({'ERROR'},f'Bake failed: {exc}'); return {'CANCELLED'}
        context.window_manager.progress_update(self._job.progress)
        if not done: return {'RUNNING_MODAL'}
        self._end(context); self.report({'INFO'},f'Baked {self._job.header.frame_count} frames to {self._job.path}')
        return {'FINISHED'}

    def _end(self,context):
        wm=context.window_manager
        if self._timer is not None: wm.event_timer_remove(self._timer)
        wm.progress_end(); self._timer=None
