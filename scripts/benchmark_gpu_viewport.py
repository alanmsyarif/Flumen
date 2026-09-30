"""Measure 600 actual completed viewport draws, including timeline/update/scheduling.

Run in a fresh Blender GUI process, not --background. A one-pixel framebuffer
readback in POST_PIXEL synchronizes prior graphics work; CUDA is synchronized
by the solver. No frame is requested until the preceding draw completes.
"""
from pathlib import Path
import sys
import time
import json
import traceback
import os
import ctypes
from ctypes import wintypes
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'examples'),str(ROOT/'scripts')]
import bpy
bpy.context.preferences.view.show_splash=False
import gpu
from create_gpu_demo import create_demo
from flumen.gpu_runtime import get_runtime
from gpu_report import summarize
from gpu_memory import memory_snapshot

OUTPUT=ROOT/'artifacts'/'benchmark-gpu-viewport.json'
scene,host=create_demo(8192,benchmark=True)
for frame in range(1,121): scene.frame_set(frame)
window=bpy.context.window
area=next(a for a in window.screen.areas if a.type=='VIEW_3D')
samples=[]; stages=[]
pending=None; interval_start=None; wall_start=None; handler=None
started=time.perf_counter()
resize_attempts=0


def resize_region(width,height):
    """Resize only this process's Blender window; never touch another app."""
    user32=ctypes.windll.user32
    found=[]
    callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    @callback_type
    def visit(hwnd,_):
        pid=wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        if pid.value==os.getpid():
            name=ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd,name,256)
            if name.value=='GHOST_WindowClass': found.append(hwnd)
        return True
    user32.EnumWindows(visit,0)
    if not found: raise RuntimeError('Cannot locate this Blender process window')
    hwnd=found[0]
    rect=wintypes.RECT()
    user32.GetWindowRect(wintypes.HWND(hwnd),ctypes.byref(rect))
    user32.SetWindowPos(wintypes.HWND(hwnd),None,0,0,
        rect.right-rect.left+1920-width,rect.bottom-rect.top+1080-height,0x14)


def finish(error=None):
    global handler
    if handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(handler,'WINDOW'); handler=None
    region=next(r for r in area.regions if r.type=='WINDOW')
    report=dict(error=error,samples_ms=samples,stages=stages)
    if samples:
        report.update(summarize('viewport_frame',samples,len(samples),len(samples),
                                time.perf_counter()-wall_start,region.width,region.height))
    else: report['realtime_viewport_pass']=False
    solver=get_runtime(host,scene)
    report.update(device=asdict(solver.device),config=asdict(solver.config),
        final_stats=asdict(solver.stats),blender=bpy.app.version_string,
        synchronization='POST_PIXEL framebuffer read_color(1x1) after each integer frame; CUDA synchronized',
        dropped_simulation_frames=0)
    report['memory']=memory_snapshot()
    OUTPUT.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('VIEWPORT_RESULT',report.get('median_ms'),report.get('p95_ms'),report.get('average_fps'),
          report['realtime_viewport_pass'],error,flush=True)
    bpy.ops.wm.quit_blender()


def draw_done():
    global pending,interval_start
    if pending is None or bpy.context.area!=area: return
    try:
        gpu.state.active_framebuffer_get().read_color(0,0,1,1,4,0,'UBYTE')
        now=time.perf_counter()
        samples.append((now-interval_start)*1000)
        stages.append(asdict(get_runtime(host,scene).stats))
        interval_start=now
        pending=None
    except Exception:
        finish(traceback.format_exc())


def tick():
    global pending,interval_start,wall_start
    try:
        if time.perf_counter()-started>120:
            finish('Viewport draw timeout'); return None
        if len(samples)==600:
            finish(); return None
        if pending is None:
            if wall_start is None:
                wall_start=interval_start=time.perf_counter()
            pending=121+len(samples)
            scene.frame_set(pending)
            area.tag_redraw()
        return .001
    except Exception:
        finish(traceback.format_exc()); return None


def prepare():
    global area,handler,resize_attempts
    try:
        if resize_attempts==0:
            with bpy.context.temp_override(window=window,area=area):
                bpy.ops.wm.window_fullscreen_toggle()
                bpy.ops.screen.screen_full_area(use_hide_panels=True)
            area=next(a for a in window.screen.areas if a.type=='VIEW_3D')
            if hasattr(window.screen,'show_statusbar'):
                window.screen.show_statusbar=False
            area.spaces.active.show_region_ui=False
            area.spaces.active.show_region_toolbar=False
            area.spaces.active.show_region_header=False
        region=next(r for r in area.regions if r.type=='WINDOW')
        print('RESIZE_CHECK',resize_attempts,window.width,window.height,region.width,region.height,flush=True)
        if (region.width,region.height)!=(1920,1080) and resize_attempts<4:
            resize_region(region.width,region.height)
            resize_attempts+=1
            return 1.0
        print('VIEWPORT_DIMENSIONS',region.width,region.height,flush=True)
        handler=bpy.types.SpaceView3D.draw_handler_add(draw_done,(),'WINDOW','POST_PIXEL')
        bpy.app.timers.register(tick,first_interval=.01)
    except Exception:
        finish(traceback.format_exc())
    return None


bpy.app.timers.register(prepare,first_interval=1.0)
