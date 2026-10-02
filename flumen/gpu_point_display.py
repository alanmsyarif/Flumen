"""Viewport-only GPU point preview. Draws published batches; never advances physics."""
from time import perf_counter
import bpy

_RECORDS = {}
_HANDLER = None
_WATER = None   # ScreenWater, created on first Water-style draw (needs a GPU context)


class _Record:
    def __init__(self, host):
        self.host = host
        self.batch = None
        self.gpu_batch = None
        self.uploaded = None


def handler_installed():
    return _HANDLER is not None


def owned_hosts():
    return set(_RECORDS)


def published_batch(host):
    record = _RECORDS.get(host.as_pointer())
    return record.batch if record else None


def create_point_display(host):
    global _HANDLER
    key = host.as_pointer()
    if key in _RECORDS: return
    _RECORDS[key] = _Record(host)
    if _HANDLER is None:
        try:
            _HANDLER = bpy.types.SpaceView3D.draw_handler_add(_draw,(),'WINDOW','POST_VIEW')
        except Exception:
            _RECORDS.pop(key,None)
            raise


def update_point_display(host, batch):
    record = _RECORDS.get(host.as_pointer())
    if record is None:
        raise RuntimeError('Point display was not created for this host')
    # Upload waits for the next draw: GPU calls need a drawing context.
    record.batch = batch
    _redraw()


def release_point_display(host):
    _release(host.as_pointer())


def purge_point_displays(alive):
    for key in list(_RECORDS):
        if key not in alive: _release(key)


def release_all_point_displays():
    for key in list(_RECORDS): _release(key)
    _remove_handler()


def _release(key):
    _RECORDS.pop(key,None)
    if not _RECORDS: _remove_handler()


def _remove_handler():
    global _HANDLER, _WATER
    _WATER = None
    if _HANDLER is not None:
        try: bpy.types.SpaceView3D.draw_handler_remove(_HANDLER,'WINDOW')
        finally: _HANDLER = None


def _redraw():
    window_manager = getattr(bpy.context,'window_manager',None)
    for window in (window_manager.windows if window_manager else ()):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()


def _upload(record, style):
    import gpu
    batch = record.batch
    record.uploaded = (batch, style)
    if style == 'WATER':
        from .gpu_water_screen import ScreenWater
        record.gpu_batch = ScreenWater.batch(batch.xyzr)   # contiguous xyz+radius rows, no copy
        return
    fmt = gpu.types.GPUVertFormat()
    fmt.attr_add(id='pos',comp_type='F32',len=3,fetch_mode='FLOAT')
    buffer = gpu.types.GPUVertBuf(fmt,batch.displayed_count)
    buffer.attr_fill('pos',batch.xyzr[:,:3])  # strided view: no host-side copy
    record.gpu_batch = gpu.types.GPUBatch(type='POINTS',buf=buffer)


def _solver(host):
    from .gpu_runtime import RUNTIMES
    runtime = RUNTIMES.get(host.as_pointer())
    return runtime.solver if runtime else None


def ScreenWater_occluder(source):
    from .gpu_water_screen import ScreenWater
    return ScreenWater.occluder(source.vertices_cpu, source.triangles_cpu)


def _stats(host):
    from .gpu_runtime import RUNTIMES
    runtime = RUNTIMES.get(host.as_pointer())
    return runtime.solver.stats if runtime else None


def _draw():
    import gpu
    global _WATER
    region = bpy.context.region_data
    clipped = bool(region and region.use_clip_planes)
    shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR',config='CLIPPED' if clipped else 'DEFAULT')
    for key,record in list(_RECORDS.items()):
        try:
            host = record.host
            visible = host.visible_get()
        except ReferenceError:
            continue
        batch = record.batch
        if not visible or batch is None or batch.displayed_count == 0: continue
        stats = _stats(host)
        settings = host.flumen_gpu
        style = settings.point_style
        if record.uploaded is None or record.uploaded[0] is not batch or record.uploaded[1] != style:
            start = perf_counter(); _upload(record, style)
            if stats is not None: stats.upload_ms = (perf_counter()-start)*1000
        start = perf_counter()
        depth_mask = gpu.state.depth_mask_get()
        if style == 'WATER':
            try:
                if _WATER is None:
                    from .gpu_water_screen import ScreenWater
                    _WATER = ScreenWater()
                solver = _solver(host)
                if solver is not None and getattr(record, 'occluder_source', None) is not solver.source:
                    record.occluder = ScreenWater_occluder(solver.source); record.occluder_source = solver.source
                _WATER.draw(record.gpu_batch, settings, getattr(record, 'occluder', None))
            finally:
                gpu.state.depth_mask_set(depth_mask); gpu.state.depth_test_set('NONE'); gpu.state.blend_set('NONE')
            if stats is not None: stats.draw_ms = (perf_counter()-start)*1000
            continue
        try:
            shader.bind()
            shader.uniform_float('color',tuple(settings.point_color))
            if clipped:
                shader.uniform_float('WorldClipPlanes',[tuple(plane) for plane in region.clip_planes])
                gpu.state.clip_distances_set(6)
            gpu.state.point_size_set(settings.point_size)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            record.gpu_batch.draw(shader)
        finally:
            gpu.state.depth_mask_set(depth_mask)
            gpu.state.depth_test_set('NONE')
            gpu.state.point_size_set(1.)
            if clipped: gpu.state.clip_distances_set(0)
        if stats is not None: stats.draw_ms = (perf_counter()-start)*1000
