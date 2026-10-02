"""Short actual 1080p million-point viewport probe; not the 120+600 acceptance gate.

Run with Blender GUI (no --background):
blender --factory-startup --python scripts/point_viewport_probe.py -- --output artifacts/point-viewport-probe.json
"""
from dataclasses import asdict
from math import radians
from pathlib import Path
import argparse
import json
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'scripts'),str(ROOT/'examples')]
import bpy
import gpu
from mathutils import Euler
import flumen
from flumen import gpu_runtime as runtime
from connected_viewport import resize_region
from gpu_memory import memory_snapshot


def build(args):
    flumen.register()
    scene=bpy.context.scene; scene.render.fps=30; scene.frame_start=1; scene.frame_set(1)
    for obj in [o for o in scene.objects if o.type=='MESH']: bpy.data.objects.remove(obj,do_unlink=True)
    window=bpy.context.window_manager.windows[0]
    area=next(a for a in window.screen.areas if a.type=='VIEW_3D')
    with bpy.context.temp_override(window=window,area=area):
        bpy.ops.mesh.primitive_monkey_add()
    source=bpy.context.view_layer.objects.active; source.scale=(.115,)*3; source.location.z=.15
    source.modifiers.new('Fixture','SUBSURF').levels=2
    bpy.context.view_layer.update()
    host=runtime.create_gpu_host(source,scene,display_mode='POINTS')
    s=host.flumen_gpu; n=args.count
    for name,value in dict(solver_backend='FIELD',capacity=n,initial_coating_count=n,particles_per_frame=0,
            source_start=0,source_softness=0,radius=.0001,time_scale=.5,lifetime=1000,kill_height=-10000,
            resistance=60,field_spacing=.002,contact_spacing=.002,resample_target=n,minimum_substeps=8,
            point_size=args.point_size).items():
        setattr(s,name,value)
    runtime.reset_host(host)
    return scene,host


class Probe:
    def __init__(self,args):
        self.args=args; self.window=bpy.context.window_manager.windows[0]
        if self.window is None: raise RuntimeError('Run in Blender GUI, without --background')
        self.scene,self.host=build(args)
        self.attempts=0; self.handler=None

    @property
    def area(self):
        # Full-screen toggles replace the area; always resolve the current one.
        return next(a for a in self.window.screen.areas if a.type=='VIEW_3D')

    def region(self):
        return next(r for r in self.area.regions if r.type=='WINDOW')

    def prepare(self):
        try:
            if self.attempts==0:
                with bpy.context.temp_override(window=self.window,area=self.area):
                    bpy.ops.wm.window_fullscreen_toggle(); bpy.ops.screen.screen_full_area(use_hide_panels=True)
                space=self.area.spaces.active
                space.show_region_ui=space.show_region_toolbar=space.show_region_header=False
                space.shading.type='SOLID'; space.overlay.show_overlays=False; space.clip_start=.001
                view=space.region_3d; view.view_perspective='PERSP'
                view.view_location=(0,0,.15); view.view_distance=.55
                view.view_rotation=Euler((radians(80),0,radians(20))).to_quaternion()
            region=self.region()
            if (region.width,region.height)!=(1920,1080) and self.attempts<4:
                resize_region(region.width,region.height); self.attempts+=1; return 1.
            if (region.width,region.height)!=(1920,1080): raise RuntimeError(f'Viewport is {region.width}x{region.height}')
            self.measure()
        except Exception:
            self.finish(error=traceback.format_exc())
        return None

    def redraw(self):
        with bpy.context.temp_override(window=self.window,area=self.area,region=self.region()):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1)

    def measure(self):
        # Reading one pixel after drawing waits for queued GPU work to finish.
        sync=lambda: gpu.state.active_framebuffer_get().read_color(0,0,1,1,4,0,'FLOAT')
        self.handler=bpy.types.SpaceView3D.draw_handler_add(sync,(),'WINDOW','POST_PIXEL')
        samples=[]; frame=self.scene.frame_current
        for index in range(self.args.warmup+self.args.frames):
            frame+=1; start=time.perf_counter()
            self.scene.frame_set(frame); updated=time.perf_counter()
            self.redraw(); end=time.perf_counter()
            stats=runtime.get_runtime(self.host,self.scene).stats
            if self.host.get('sf_gpu_error'): raise RuntimeError(self.host['sf_gpu_error'])
            if index>=self.args.warmup:
                samples.append(dict(frame=frame,frame_ms=(end-start)*1000,update_ms=(updated-start)*1000,
                                    redraw_ms=(end-updated)*1000,stats=asdict(stats)))
        image=self.args.output.resolve().with_suffix('.png')
        with bpy.context.temp_override(window=self.window,area=self.area,region=self.region()):
            bpy.ops.screen.screenshot(filepath=str(image))
        if not image.exists(): raise RuntimeError(f'Viewport screenshot was not saved: {image}')
        frame_ms=sorted(s['frame_ms'] for s in samples)
        last=samples[-1]['stats']
        self.finish(report=dict(measurement_kind='short_viewport_probe',acceptance_gate=False,
            viewport=(1920,1080),shading='SOLID',backend=gpu.platform.backend_type_get(),
            gpu=gpu.platform.renderer_get(),warmup_frames=self.args.warmup,
            point_size=self.args.point_size,live_count=last['live_count'],displayed_count=last['displayed_count'],
            mean_frame_ms=statistics.mean(frame_ms),median_frame_ms=statistics.median(frame_ms),
            p95_frame_ms=frame_ms[max(0,int(len(frame_ms)*.95+.999)-1)],fps=1000/statistics.mean(frame_ms),
            median_update_ms=statistics.median(s['update_ms'] for s in samples),
            median_redraw_ms=statistics.median(s['redraw_ms'] for s in samples),
            median_solver_ms=statistics.median(s['stats']['solver_ms'] for s in samples),
            median_readback_ms=statistics.median(s['stats']['readback_ms'] for s in samples),
            median_upload_ms=statistics.median(s['stats']['upload_ms'] for s in samples),
            median_draw_submit_ms=statistics.median(s['stats']['draw_ms'] for s in samples),
            screenshot=image.name,samples=samples,memory=memory_snapshot(),
            timing_notes='frame_ms = frame_set (solver + point readback) + forced DRAW_WIN_SWAP redraw with '
                         'POST_PIXEL one-pixel readback; upload/draw_submit are CPU-side times inside the draw handler'))

    def finish(self,report=None,error=None):
        if self.handler is not None: bpy.types.SpaceView3D.draw_handler_remove(self.handler,'WINDOW')
        output=self.args.output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
        if error: report=dict(measurement_kind='short_viewport_probe',status='failed',error=error)
        output.write_text(json.dumps(report,indent=2),encoding='utf8')
        print('POINT_VIEWPORT_RESULT',output,'failed' if error else round(report['fps'],2),flush=True)
        if error: print(error,flush=True)
        bpy.ops.wm.quit_blender()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=1_000_000)
    parser.add_argument('--frames',type=int,default=30)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--point-size',type=float,default=2.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    bpy.context.preferences.view.show_splash=False
    def start():
        try:
            probe=Probe(args); bpy.app.timers.register(probe.prepare,first_interval=.5)
        except Exception:
            print('POINT_VIEWPORT_RESULT failed',traceback.format_exc(),flush=True); bpy.ops.wm.quit_blender()
        return None
    bpy.app.timers.register(start,first_interval=1.)


main()
