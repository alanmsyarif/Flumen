"""Full-count 1080p particle viewport gate: 120 warmup + 600 measured draws.

Each measured draw advances exactly one integer interval; nothing is skipped to catch up.
Run with Blender GUI (no --background):
blender --factory-startup --python scripts/benchmark_particle_viewport.py -- --distribution attached --output PATH
"""
from dataclasses import asdict
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
import numpy as np
from flumen import gpu_runtime as runtime
from flumen.gpu.preview_report import validate_particle_preview_report, STAGES
from point_viewport_probe import Probe
from benchmark_particle_solver import apply_distribution
from gpu_memory import memory_snapshot


class ViewportGate(Probe):
    def __init__(self,args):
        self.device_before=memory_snapshot().get('whole_device_used_mib')
        started=time.perf_counter()
        super().__init__(args)
        self.setup_seconds=time.perf_counter()-started
        solver=runtime.get_runtime(self.host,self.scene)
        apply_distribution(solver,args.distribution)
        self.device_after_setup=memory_snapshot().get('whole_device_used_mib')
        self.samples=[]; self.failure=None; self.wall_seconds=None

    def measure(self):
        sync=lambda: gpu.state.active_framebuffer_get().read_color(0,0,1,1,4,0,'FLOAT')
        self.handler=bpy.types.SpaceView3D.draw_handler_add(sync,(),'WINDOW','POST_PIXEL')
        solver=runtime.get_runtime(self.host,self.scene)
        frame=self.scene.frame_current; wall_start=None
        try:
            for index in range(self.args.warmup+self.args.frames):
                if index==self.args.warmup: wall_start=time.perf_counter()
                previous=solver.current_frame; frame+=1
                start=time.perf_counter(); self.scene.frame_set(frame); updated=time.perf_counter()
                if self.host.get('sf_gpu_error'): raise RuntimeError(self.host['sf_gpu_error'])
                self.redraw(); end=time.perf_counter()
                region=self.region(); stats=solver.stats
                if index>=self.args.warmup:
                    sample=dict(frame=frame,intervals=solver.current_frame-previous,frame_ms=(end-start)*1000,
                                update_ms=(updated-start)*1000,redraw_ms=(end-updated)*1000,
                                width=region.width,height=region.height)
                    sample.update({k:v for k,v in asdict(stats).items() if k!='frame'})
                    self.samples.append(sample)
            self.wall_seconds=time.perf_counter()-wall_start
        except Exception:
            self.failure=dict(frame=frame,error=traceback.format_exc())
        self.report(solver)

    def report(self,solver):
        pool=solver.pool
        finite=bool(np.isfinite(pool.data.position.numpy()).all() and np.isfinite(pool.data.velocity.numpy()).all())
        last=self.samples[-1] if self.samples else None
        ledger=None
        if last:
            ledger=abs(last['emitted_volume']-last['live_volume']-last['removed_volume'])/max(last['emitted_volume'],1e-20)
        frame_ms=sorted(s['frame_ms'] for s in self.samples)
        median=lambda name: statistics.median(s[name] for s in self.samples) if self.samples else None
        memory=memory_snapshot()
        memory['owned_array_bytes']=last['owned_array_bytes'] if last else None
        memory['device_before_setup_mib']=self.device_before
        memory['device_after_setup_mib']=self.device_after_setup
        prepared=solver.prepared
        report=dict(measurement_kind='particle_viewport',status='failed' if self.failure else 'passed',
            failure=self.failure,distribution=self.args.distribution,capacity=self.args.count,
            warmup_frames=self.args.warmup,measured_frames=len(self.samples),wall_seconds=self.wall_seconds,
            viewport=(1920,1080),shading='SOLID',point_size=self.args.point_size,style=self.args.style,
            gpu_backend=gpu.platform.backend_type_get(),gpu=gpu.platform.renderer_get(),
            gpu_driver=gpu.platform.version_get(),blender=bpy.app.version_string,
            physical_dt=solver.dt,source_triangles=len(solver.source.triangles_cpu),
            field_operator_spacing=prepared.chart.operator_spacing if prepared else None,
            contact_spacing=prepared.contact.effective_spacing if prepared else None,
            setup_seconds=self.setup_seconds,config=asdict(solver.config),finite_state=finite,
            ledger_relative_error=ledger,
            mean_fps=len(self.samples)/self.wall_seconds if self.wall_seconds else None,
            median_frame_ms=statistics.median(frame_ms) if frame_ms else None,
            p95_frame_ms=frame_ms[max(0,-(-len(frame_ms)*95//100)-1)] if frame_ms else None,
            max_frame_ms=frame_ms[-1] if frame_ms else None,
            medians={name:median(name) for name in STAGES+('update_ms','redraw_ms')},
            minimum_live=min((s['live_count'] for s in self.samples),default=None),
            minimum_displayed=min((s['displayed_count'] for s in self.samples),default=None),
            resampled_count=last['resampled_count'] if last else None,
            merged_pairs=last['merged_pairs'] if last else None,
            final_contact_fallback=last['contact_fallback_count'] if last else None,
            memory=memory,samples=self.samples)
        report['gate_errors']=validate_particle_preview_report(report)
        report['gate_pass']=not report['gate_errors']
        output=self.args.output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf8')
        print('PARTICLE_VIEWPORT_RESULT',output,report['status'],'PASS' if report['gate_pass'] else 'FAIL',
              report['mean_fps'],report['p95_frame_ms'],flush=True)
        for error in report['gate_errors'][:8]: print('GATE_ERROR',error,flush=True)
        if self.failure: print(self.failure['error'],flush=True)
        if self.handler is not None: bpy.types.SpaceView3D.draw_handler_remove(self.handler,'WINDOW')
        self.handler=None
        bpy.ops.wm.quit_blender()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--distribution',choices=['attached','free','mixed','dense'],default='attached')
    parser.add_argument('--count',type=int,default=1_000_000)
    parser.add_argument('--warmup',type=int,default=120)
    parser.add_argument('--frames',type=int,default=600)
    parser.add_argument('--point-size',type=float,default=2.)
    parser.add_argument('--style',choices=['POINTS','WATER'],default='POINTS')
    parser.add_argument('--water-smoothing',type=float,default=.003)
    parser.add_argument('--water-scale',type=float,default=3.)
    parser.add_argument('--contact-hysteresis',type=float,default=0.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    bpy.context.preferences.view.show_splash=False
    def start():
        try:
            gate=ViewportGate(args); bpy.app.timers.register(gate.prepare,first_interval=.5)
        except Exception:
            print('PARTICLE_VIEWPORT_RESULT failed',traceback.format_exc(),flush=True); bpy.ops.wm.quit_blender()
        return None
    bpy.app.timers.register(start,first_interval=1.)


if __name__=='__main__': main()
