"""Blender mesh extraction and GPU host lifecycle."""
import numpy as np
import bpy
from bpy.app.handlers import persistent
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4
from .gpu_properties import config_for
from .gpu_display import create_display, update_display, set_material

RUNTIMES = {}
_BUSY = False
_RENDERING = False


@dataclass
class HostRuntime:
    host: object
    solver: object
    signature: tuple
    geometry_hash: str


def _fingerprint(arrays):
    digest=sha256()
    for array in arrays: digest.update(array.tobytes())
    return digest.hexdigest()


def _signature(host,scene):
    cfg=config_for(host)
    source=host.flumen_gpu.source
    if source is None or source.type!='MESH':
        raise ValueError('Choose a collision mesh and Reset GPU Flow')
    if host.parent or any(abs(host.matrix_world[r][c]-(1 if r==c else 0))>1e-6 for r in range(4) for c in range(4)):
        raise ValueError('Keep the GPU Flow host unparented at identity transforms')
    if abs(scene.unit_settings.scale_length-1)>1e-8:
        raise ValueError('Set Scene Unit Scale to 1.0')
    return (cfg,source.as_pointer(),tuple(v for row in source.matrix_world for v in row),
            scene.frame_start,scene.render.fps,scene.render.fps_base,scene.as_pointer())


def mark_dirty(host):
    if not _BUSY and host and host.get('sf_gpu_host') and host.as_pointer() in RUNTIMES:
        host['sf_gpu_error']='Settings changed. Reset GPU Flow.'


def purge_deleted():
    alive={obj.as_pointer() for obj in bpy.data.objects}
    for key in list(RUNTIMES):
        if key not in alive:
            RUNTIMES.pop(key).solver.close()
    from .gpu_water_display import purge_orphan_water_displays
    purge_orphan_water_displays()


def get_runtime(host,scene):
    if host is None or not host.get('sf_gpu_host'):
        raise ValueError('Select a GPU Flow host')
    purge_deleted()
    signature=_signature(host,scene)
    key=host.as_pointer()
    record=RUNTIMES.get(key)
    if record:
        if record.signature!=signature or host.get('sf_gpu_error'):
            raise RuntimeError('Simulation inputs changed. Reset GPU Flow.')
        return record.solver
    from .gpu.device import require_cuda
    from .gpu.source import build_source
    from .gpu.solver import FlowSolver
    device=require_cuda()
    arrays=extract_source(host.flumen_gpu.source,bpy.context.evaluated_depsgraph_get())
    source=build_source(*arrays,signature[0],device)
    solver=None
    try:
        solver=FlowSolver(signature[0],source,device,start_frame=scene.frame_start,
                          fps=scene.render.fps,fps_base=scene.render.fps_base)
        solver.seek(scene.frame_current)
        host['sf_gpu_uid']=str(uuid4())
        host['sf_gpu_device']=device.name
        host['sf_gpu_error']=''
        if signature[0].display_mode=='CONNECTED':
            from .gpu_water_display import create_water_display
            create_water_display(host,solver.topology)
        else:
            from .gpu_water_display import release_water_display
            release_water_display(host)
            host['sf_gpu_water_display']=False
            host['sf_gpu_geometry_error']=''
            if not any(m.type=='NODES' and m.node_group and m.node_group.get('sf_gpu_display') for m in host.modifiers):
                create_display(host)
        set_material(host,host.flumen_gpu.material)
        RUNTIMES[key]=HostRuntime(host,solver,signature,_fingerprint(arrays))
        return solver
    except Exception:
        from .gpu_water_display import release_water_display
        release_water_display(host)
        if solver is not None: solver.close()
        else: source.close()
        raise


def evaluate_host(host,scene,frame=None):
    global _BUSY
    if _BUSY: return None
    if _RENDERING:
        raise RuntimeError('GPU Flow is live preview only; render baking is not available')
    if frame is None:
        if scene.frame_subframe!=0:
            raise ValueError('GPU Flow supports integer frames only')
        frame=scene.frame_current
    if not isinstance(frame,int):
        raise ValueError('GPU Flow supports integer frames only')
    _BUSY=True
    try:
        solver=get_runtime(host,scene)
        stats=solver.seek(frame)
        if solver.config.display_mode=='CONNECTED':
            from .gpu_water_display import update_water_display
            from time import perf_counter
            geometry=solver.water_snapshot(); fields=solver.surface_snapshot()
            start=perf_counter()
            update_water_display(host,geometry,fields)
            stats.display_update_ms=(perf_counter()-start)*1000
        else:
            update_display(host,solver.snapshot())
        return stats
    finally:
        _BUSY=False


def reset_host(host):
    if host is None or not host.get('sf_gpu_host'):
        raise ValueError('Select a GPU Flow host')
    record=RUNTIMES.pop(host.as_pointer(),None)
    if record: record.solver.close()
    from .gpu_water_display import release_water_display
    release_water_display(host)
    host['sf_gpu_error']=''
    return evaluate_host(host,bpy.context.scene)


def create_gpu_host(source,scene,display_mode='DROPS'):
    global _BUSY
    if source is None or source.type!='MESH' or source.get('sf_gpu_host') or source.get('sf_simulation_host'):
        raise ValueError('Select a stationary collision mesh')
    if display_mode not in ('DROPS','CONNECTED'):
        raise ValueError('Unknown water display mode')
    from .gpu.device import require_cuda
    require_cuda()
    mesh=bpy.data.meshes.new('GPU Flow Points')
    host=bpy.data.objects.new('Flumen GPU Flow',mesh)
    scene.collection.objects.link(host)
    host['sf_gpu_host']=True
    # Live state is deliberately excluded from offline rendering until baked export exists.
    host.hide_render=True
    _BUSY=True
    try:
        host.flumen_gpu.source=source
        host.flumen_gpu.display_mode=display_mode
        host.flumen_gpu.interactions_enabled=display_mode=='CONNECTED'
        host.flumen_gpu.emission_start=scene.frame_start
        host.flumen_gpu.emission_end=scene.frame_end
    finally:
        _BUSY=False
    try:
        evaluate_host(host,scene)
        return host
    except Exception:
        record=RUNTIMES.pop(host.as_pointer(),None)
        if record: record.solver.close()
        from .gpu_water_display import release_water_display
        release_water_display(host)
        material=host.flumen_gpu.material
        trees=[m.node_group for m in host.modifiers if m.type=='NODES']
        bpy.data.objects.remove(host,do_unlink=True)
        if mesh.users==0: bpy.data.meshes.remove(mesh)
        for tree in trees:
            if tree and tree.users==0: bpy.data.node_groups.remove(tree)
        if material and material.users==0 and material.get('sf_gpu_water_material'):
            bpy.data.materials.remove(material)
        raise


def release_all():
    from .gpu_water_display import release_water_display,purge_orphan_water_displays
    for record in list(RUNTIMES.values()):
        record.solver.close()
        try: release_water_display(record.host)
        except ReferenceError: pass
    RUNTIMES.clear()
    purge_orphan_water_displays()


@persistent
def on_frame(scene,depsgraph=None):
    if _BUSY or _RENDERING: return
    purge_deleted()
    for host in list(scene.objects):
        if host.get('sf_gpu_host'):
            try:
                evaluate_host(host,scene)
            except (ValueError,RuntimeError,ImportError,OSError) as exc:
                message=str(exc)
                if host.get('sf_gpu_error')!=message: host['sf_gpu_error']=message


@persistent
def on_depsgraph(scene,depsgraph):
    global _BUSY
    if _BUSY: return
    _BUSY=True
    try:
        purge_deleted()
        changed={update.id.original.as_pointer() for update in depsgraph.updates
                 if update.is_updated_geometry or update.is_updated_transform}
        for record in list(RUNTIMES.values()):
            host=record.host
            source=host.flumen_gpu.source
            if source and (source.as_pointer() in changed or source.data.as_pointer() in changed):
                arrays=extract_source(source,depsgraph)
                if _fingerprint(arrays)!=record.geometry_hash:
                    host['sf_gpu_error']='Collision surface changed. Reset GPU Flow.'
    finally:
        _BUSY=False


@persistent
def on_load(_):
    global _RENDERING
    release_all()
    _RENDERING=False


@persistent
def on_render(scene):
    global _RENDERING
    _RENDERING=True
    for host in scene.objects:
        if host.get('sf_gpu_host'):
            host['sf_gpu_error']='Live GPU Flow has no render bake; use viewport preview.'


@persistent
def on_render_end(scene):
    global _RENDERING
    _RENDERING=False


def register_handlers():
    for handlers,callback in _handlers():
        if callback not in handlers: handlers.append(callback)


def unregister_handlers():
    release_all()
    for handlers,callback in _handlers():
        if callback in handlers: handlers.remove(callback)


def _handlers():
    h=bpy.app.handlers
    return [(h.frame_change_post,on_frame),(h.depsgraph_update_post,on_depsgraph),
            (h.load_pre,on_load),(h.undo_pre,on_load),(h.redo_pre,on_load),
            (h.render_init,on_render),(h.render_complete,on_render_end),(h.render_cancel,on_render_end)]


def extract_source(obj, depsgraph) -> tuple:
    if obj is None or obj.type != 'MESH':
        raise ValueError('Select a collision mesh')
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = np.empty(len(mesh.vertices)*3,dtype=np.float32)
        mesh.vertices.foreach_get('co',vertices)
        vertices = vertices.reshape(-1,3)
        matrix = np.asarray(evaluated.matrix_world,dtype=np.float32)
        vertices = vertices @ matrix[:3,:3].T + matrix[:3,3]
        faces = np.empty(len(mesh.loop_triangles)*3,dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices',faces)
        faces = faces.reshape(-1,3)
        parents = list(range(len(vertices)))
        def root(a):
            while parents[a] != a:
                parents[a] = parents[parents[a]]
                a = parents[a]
            return a
        for a,b,c in faces:
            ra = root(int(a))
            parents[root(int(b))] = ra
            parents[root(int(c))] = ra
        islands = np.asarray([root(int(face[0])) for face in faces],dtype=np.int32)
        return np.ascontiguousarray(vertices), faces, islands
    finally:
        evaluated.to_mesh_clear()
