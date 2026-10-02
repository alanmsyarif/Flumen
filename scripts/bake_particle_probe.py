"""Short actual million-particle cache bake/read probe (background Blender).

blender --background --factory-startup --python scripts/bake_particle_probe.py -- --cache DIR --output REPORT.json
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'scripts')]
import bpy
import numpy as np
import flumen
from flumen import gpu_runtime as runtime
from flumen.gpu_bake import BakeJob, physical_settings
from flumen.particle_cache import CacheReader
from gpu_memory import memory_snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=10)
    parser.add_argument('--count',type=int,default=1_000_000)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    flumen.register()
    scene=bpy.context.scene; scene.render.fps=30; scene.frame_start=1; scene.frame_end=args.frames
    bpy.ops.mesh.primitive_monkey_add(); source=bpy.context.object
    source.scale=(.115,)*3; source.location.z=.15; source.modifiers.new('Fixture','SUBSURF').levels=2
    bpy.context.view_layer.update()
    host=runtime.create_gpu_host(source,scene,display_mode='POINTS',solver_backend='FIELD')
    s=host.flumen_gpu; n=args.count
    for name,value in dict(capacity=n,initial_coating_count=n,particles_per_frame=0,source_start=.55,
            source_softness=.05,radius=.0001,time_scale=.5,lifetime=1000,kill_height=-10000,resistance=60,
            field_spacing=.002,contact_spacing=.002,resample_target=n,minimum_substeps=8).items():
        setattr(s,name,value)
    runtime.reset_host(host); scene.frame_set(5)
    live=runtime.get_runtime(host,scene); before=live.cache_snapshot()
    start=time.perf_counter(); job=BakeJob(host,scene,args.cache); setup=time.perf_counter()-start
    frame_seconds=[]
    while True:
        t=time.perf_counter(); done=job.step(); frame_seconds.append(time.perf_counter()-t)
        if done: break
    bake_seconds=time.perf_counter()-start
    after=live.cache_snapshot()
    unchanged=live.current_frame==5 and all(np.array_equal(before.arrays[k],after.arrays[k]) for k in before.arrays)
    t=time.perf_counter(); reader=CacheReader(args.cache)
    reader.validate(job.header.source_fingerprint,physical_settings(live.config))
    for frame in range(1,args.frames+1): reader.read(frame)
    read_seconds=time.perf_counter()-t
    baked=reader.read(5).arrays
    exact=all(np.array_equal(baked[k],before.arrays[k]) for k in before.arrays)
    # Two separate solvers can differ by GPU atomic ordering; report the size of any difference.
    differences={k:(float(np.abs(baked[k].astype(np.float64)-before.arrays[k].astype(np.float64)).max())
                    if baked[k].shape==before.arrays[k].shape else f'shape {baked[k].shape} vs {before.arrays[k].shape}')
                 for k in before.arrays}
    volume_totals=(float(baked['volume'].astype(np.float64).sum()),float(before.arrays['volume'].astype(np.float64).sum()))
    size=sum(p.stat().st_size for p in args.cache.iterdir())
    report=dict(measurement_kind='particle_cache_probe',frames=args.frames,particles=n,
        live_count_frame5=len(before.arrays['ids']),setup_seconds=setup,bake_seconds=bake_seconds,
        median_frame_seconds=float(np.median(frame_seconds)),read_all_seconds=read_seconds,
        cache_bytes=size,estimated_bytes=job.estimated_bytes,bytes_per_frame=size/args.frames,
        live_state_unchanged=unchanged,frame5_matches_live_exactly=exact,frame5_max_differences=differences,frame5_volume_totals=volume_totals,memory=memory_snapshot())
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('BAKE_PROBE',json.dumps({k:v for k,v in report.items() if k!='memory'}),report['memory'],flush=True)


main()
