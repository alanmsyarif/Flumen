"""Actual particle-solver feasibility; deliberately not a viewport FPS claim."""
import argparse
from dataclasses import asdict
import json
from math import ceil
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'scripts')]
import bpy
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.solver import FlowSolver
from flumen.gpu_runtime import extract_source
from gpu_memory import memory_snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capacity',type=int,default=1_000_000)
    parser.add_argument('--frames',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--minimum-substeps',type=int,default=8)
    parser.add_argument('--interactions',action='store_true')
    parser.add_argument('--merge-distance',type=float,default=.25)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if args.frames<1 or args.warmup<0: parser.error('Frames must be positive; warmup must be nonnegative')
    cfg=FlowConfig(capacity=args.capacity,initial_coating_count=args.capacity,particles_per_frame=0,
        source_start=0,source_softness=0,radius=.0001,time_scale=.5,lifetime=1000,
        minimum_substeps=args.minimum_substeps,interactions_enabled=args.interactions,
        display_mode='DROPS',merge_distance_scale=args.merge_distance)
    cfg.validate()
    bpy.ops.mesh.primitive_monkey_add()
    obj=bpy.context.object; obj.scale=(.115,)*3; obj.location.z=.15
    modifier=obj.modifiers.new('Fixture','SUBSURF'); modifier.levels=2
    bpy.context.view_layer.update()
    arrays=extract_source(obj,bpy.context.evaluated_depsgraph_get())
    device=require_cuda(); source=build_source(*arrays,cfg,device)
    solver=None
    try:
        solver=FlowSolver(cfg,source,device)
        setup=time.perf_counter(); solver.seek(1)
        setup_seconds=time.perf_counter()-setup
        for frame in range(2,2+args.warmup): solver.seek(frame)
        samples=[]; stages=[]
        for frame in range(2+args.warmup,2+args.warmup+args.frames):
            start=time.perf_counter(); solver.seek(frame)
            elapsed=(time.perf_counter()-start)*1000
            samples.append(elapsed); stages.append(asdict(solver.stats))
            print('PARTICLE_SOLVER_FRAME',frame,round(elapsed,2),solver.stats.live_count,flush=True)
        ordered=sorted(samples)
        finite_state=bool(np.isfinite(solver.pool.data.position.numpy()).all()
                          and np.isfinite(solver.pool.data.velocity.numpy()).all())
        final=stages[-1]
        ledger_error=abs(final['emitted_volume']-final['live_volume']-final['removed_volume'])/max(final['emitted_volume'],1.e-20)
        report=dict(measurement_kind='solver_only_feasibility',viewport_tested=False,
            reconstruction_enabled=False,setup_seconds=setup_seconds,warmup_frames=args.warmup,
            samples_ms=samples,stages=stages,mean_ms=statistics.mean(samples),
            p95_ms=ordered[ceil(len(ordered)*.95)-1],max_ms=max(samples),
            solver_only_fps=1000/statistics.mean(samples),config=asdict(cfg),
            device=asdict(device),blender=bpy.app.version_string,source_triangles=len(arrays[1]),
            minimum_live_count=min(s['live_count'] for s in stages),finite_state=finite_state,
            ledger_relative_error=ledger_error,memory=memory_snapshot())
        output=args.output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf8')
        print('PARTICLE_SOLVER_RESULT',output,report['solver_only_fps'],flush=True)
    finally:
        if solver is not None: solver.close()
        else: source.close()


if __name__=='__main__': main()
