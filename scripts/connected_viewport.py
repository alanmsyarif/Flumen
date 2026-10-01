"""Capture/measure completed viewport draws of the same Connected fixture."""
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict
import json
import os
import subprocess
import sys
import time
import traceback
import bpy
import gpu
from create_connected_water_demo import create_connected_demo
from flumen.gpu_runtime import get_runtime
from gpu_report import summarize
from gpu_memory import memory_snapshot


def resize_region(width,height):
    user32=ctypes.windll.user32; found=[]
    callback_type=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    @callback_type
    def visit(hwnd,_):
        pid=wintypes.DWORD(); user32.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        if pid.value==os.getpid():
            name=ctypes.create_unicode_buffer(256); user32.GetClassNameW(hwnd,name,256)
            if name.value=='GHOST_WindowClass': found.append(hwnd)
        return True
    user32.EnumWindows(visit,0)
    if not found: raise RuntimeError('Cannot locate this Blender process window')
    rect=wintypes.RECT(); hwnd=wintypes.HWND(found[0]); user32.GetWindowRect(hwnd,ctypes.byref(rect))
    user32.SetWindowPos(hwnd,None,0,0,rect.right-rect.left+1920-width,rect.bottom-rect.top+1080-height,0x14)


class ViewportJob:
    def __init__(self,benchmark,args):
        self.benchmark=benchmark; self.args=args
        args.output.parent.mkdir(parents=True,exist_ok=True)
        self.scene,self.host=create_connected_demo(args.view,720 if benchmark else 120)
        if benchmark:
            get_runtime(self.host,self.scene).interaction.enable_timing()
        self.window=bpy.context.window
        if self.window is None: raise RuntimeError('Run this script in Blender GUI, without --background')
        self.samples=[]; self.stages=[]; self.records=[]; self.completed=0; self.pending=None
        self.interval_start=None; self.wall_start=None; self.handler=None; self.draw_passes=0
        self.resize_attempts=0; self.started=time.perf_counter(); self.geometry_errors=0
        self.simulated_frames=0; self.finished=False; self.frame_update_ms=0.; self.ready_stage=None
        self.frames_dir=args.output.with_suffix('').with_name(args.output.stem+'_frames')
        if not benchmark: self.frames_dir.mkdir(parents=True,exist_ok=True)
        if not benchmark and args.view=='WATER':
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output.parent/'Flumen_Connected_Water_Demo.blend'))

    @property
    def area(self):
        # Screen/fullscreen changes can free an Area while its Python wrapper
        # remains reachable. Resolve from the live window for every UI call.
        return next((a for a in self.window.screen.areas if a.type=='VIEW_3D'),None)

    def finish(self,error=None):
        if self.finished: return
        self.finished=True
        if self.handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handler,'WINDOW'); self.handler=None
        solver=get_runtime(self.host,self.scene)
        region=next(r for r in self.area.regions if r.type=='WINDOW')
        report=dict(error=error,view=self.args.view,config=asdict(solver.config),device=asdict(solver.device),
            source_dimensions=list(self.scene['sf_source_dimensions']),source_triangles=self.scene['sf_source_triangles'],
            blender=bpy.app.version_string,final_stats=asdict(solver.stats),stages=self.stages,
            synchronization='POST_PIXEL read_color plus next-event-loop DwmFlush; CUDA synchronized; capture uses the final composited Blender screenshot',
            timing_notes='interaction_ms is CUDA event time within solver_ms; reconstruction_ms includes geometry transfer; transfer_ms measures completed geometry and field copies; display_update_ms is Blender mesh/wetness update; frame_update_ms includes all frame handling; draw_and_event_ms includes event scheduling and composition',
            completed_frames=self.completed,frame_records=self.records,samples_ms=self.samples,memory=memory_snapshot())
        if self.benchmark:
            report['warmup_draws']=min(self.completed,120)
            report.update(summarize('viewport_frame',self.samples,self.simulated_frames,len(self.samples),
                time.perf_counter()-self.wall_start,region.width,region.height,
                display_mode=solver.config.display_mode,
                engine=self.scene.render.engine if self.args.view=='WATER' else 'BLENDER_WORKBENCH',
                fixture=self.scene['sf_connected_fixture'],source_triangles=self.scene['sf_source_triangles'],
                dropped_simulation_frames=0,geometry_errors=self.geometry_errors)) if self.samples else report.update(realtime_connected_pass=False)
        elif error is None:
            try:
                result=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','30',
                    '-i',str(self.frames_dir/'frame-%04d.png'),'-c:v','libx264','-crf','18',
                    '-pix_fmt','yuv420p',str(self.args.output)],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode: raise RuntimeError(result.stderr)
            except Exception: report['error']=traceback.format_exc()
        path=self.args.output if self.benchmark else self.args.output.with_suffix('.json')
        path.write_text(json.dumps(report,indent=2),encoding='utf8')
        print('CONNECTED_RESULT',path,report.get('average_fps'),report.get('p95_ms'),report.get('error'),flush=True)
        bpy.ops.wm.quit_blender()

    def draw_done(self):
        if self.pending is None or bpy.context.area!=self.area or self.finished or self.ready_stage is not None: return
        try:
            self.draw_passes+=1
            if not self.benchmark and self.draw_passes<4: return
            framebuffer=gpu.state.active_framebuffer_get()
            framebuffer.read_color(0,0,1,1,4,0,'UBYTE')
            solver=get_runtime(self.host,self.scene)
            if solver.current_frame!=self.pending: raise RuntimeError('A simulation step was skipped')
            diagnostic=self.host.get('sf_gpu_geometry_error','')
            stage=dict(asdict(solver.stats),frame_update_ms=self.frame_update_ms,geometry_error=diagnostic)
            self.ready_stage=stage
        except Exception: self.finish(traceback.format_exc())

    def tick(self):
        if self.finished: return None
        try:
            if time.perf_counter()-self.started>1800:
                self.finish('Viewport draw timeout'); return None
            if self.ready_stage is not None:
                # POST_PIXEL precedes final viewport composition on this Blender
                # build. Process the completed window frame before advancing.
                ctypes.windll.dwmapi.DwmFlush()
                now=time.perf_counter(); stage=self.ready_stage
                if self.benchmark:
                    if self.pending>120:
                        total_ms=(now-self.interval_start)*1000
                        stage['draw_and_event_ms']=max(0.,total_ms-self.frame_update_ms)
                        stage['update_and_field_transfer_ms']=max(0.,self.frame_update_ms-stage['solver_ms']-stage['reconstruction_ms'])
                        self.samples.append(total_ms); self.stages.append(stage)
                        self.simulated_frames+=1
                        if stage['geometry_error']: self.geometry_errors+=1
                else:
                    path=self.frames_dir/f'frame-{self.pending:04d}.png'
                    result=bpy.ops.screen.screenshot(filepath=str(path))
                    if result!={'FINISHED'} or not path.is_file():
                        raise RuntimeError(f'Viewport screenshot was not saved: {path}')
                    fields=get_runtime(self.host,self.scene).surface_snapshot()
                    stage.update(wetness_mean=float(fields.wetness.mean()),wetness_max=float(fields.wetness.max()))
                    self.records.append(stage)
                self.completed+=1; self.pending=None; self.ready_stage=None; self.interval_start=now
                if self.completed%30==0:
                    print('CONNECTED_PROGRESS',self.completed,stage['live_count'],stage['water_vertices'],stage['geometry_error'],flush=True)
                return .001
            total=720 if self.benchmark else self.args.frames
            if self.completed==total:
                self.finish(); return None
            if self.pending is None:
                self.pending=self.completed+1; self.draw_passes=0
                if self.pending==121 and self.benchmark: self.wall_start=self.interval_start=time.perf_counter()
                started=time.perf_counter(); self.scene.frame_set(self.pending)
                self.frame_update_ms=(time.perf_counter()-started)*1000
            area=self.area
            if area is None: raise RuntimeError('The validation viewport was closed')
            region=next(r for r in area.regions if r.type=='WINDOW')
            if (region.width,region.height)!=(1920,1080): raise RuntimeError('The validation viewport was resized')
            area.tag_redraw()
            return .001
        except Exception:
            self.finish(traceback.format_exc()); return None

    def prepare(self):
        try:
            if self.resize_attempts==0:
                with bpy.context.temp_override(window=self.window,area=self.area):
                    bpy.ops.wm.window_fullscreen_toggle(); bpy.ops.screen.screen_full_area(use_hide_panels=True)
                if hasattr(self.window.screen,'show_statusbar'): self.window.screen.show_statusbar=False
                space=self.area.spaces.active
                space.show_region_ui=False; space.show_region_toolbar=False; space.show_region_header=False
                space.region_3d.view_camera_zoom=29.3
            region=next(r for r in self.area.regions if r.type=='WINDOW')
            if (region.width,region.height)!=(1920,1080) and self.resize_attempts<4:
                resize_region(region.width,region.height); self.resize_attempts+=1; return 1.
            if (region.width,region.height)!=(1920,1080): raise RuntimeError('Viewport is not 1920x1080')
            self.handler=bpy.types.SpaceView3D.draw_handler_add(self.draw_done,(),'WINDOW','POST_PIXEL')
            bpy.app.timers.register(self.tick,first_interval=.01)
        except Exception: self.finish(traceback.format_exc())
        return None


def run(benchmark=False):
    bpy.context.preferences.view.show_splash=False
    parser=argparse.ArgumentParser()
    parser.add_argument('--view',choices=('WATER','GEOMETRY'),default='WATER')
    parser.add_argument('--frames',type=int,default=180)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    args.output=args.output.resolve()
    if args.frames<1: raise ValueError('Frames must be positive')
    job=ViewportJob(benchmark,args)
    bpy.app.timers.register(job.prepare,first_interval=1.)
    return job
