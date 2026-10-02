"""Bake a few million-particle frames, then mesh them offline and report cost (background Blender).

blender --background --factory-startup --python scripts/mesh_particle_probe.py -- --work DIR --output REPORT.json
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'scripts')]
import bpy
import flumen
from flumen import gpu_runtime as runtime
from flumen.gpu_bake import BakeJob
from flumen.particle_cache import CacheReader
from flumen.offline_mesher import MeshOptions, mesh_cached_frame
from gpu_memory import memory_snapshot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=3)
    parser.add_argument('--count',type=int,default=1_000_000)
    parser.add_argument('--spacing',nargs='+',default=['.0001:.001','.0001:.0005'],help='drop:film spacing pairs, meters')
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
    runtime.reset_host(host)
    job=BakeJob(host,scene,args.work/'cache')
    while not job.step(): pass
    reader=CacheReader(args.work/'cache'); results=[]
    for pair in args.spacing:
        drop,film=(float(x) for x in pair.split(':'))
        t=time.perf_counter()
        result=mesh_cached_frame(reader,args.frames,MeshOptions(spacing=drop,film_spacing=film),args.work/f'mesh_{drop}_{film}',lambda: False)
        result['film_spacing']=film
        result['wall_seconds']=time.perf_counter()-t
        results.append(result)
        print('MESH_PROBE',json.dumps({k:result[k] for k in ('spacing','film_spacing','film_max_spacing','seconds','bytes','attached_triangles','free_triangles',
            'attached_represented_volume','attached_mesh_volume','free_volume','free_mesh_volume','subresolution_volume','tiles','peak_traced_bytes')}),flush=True)
    frame=reader.read(args.frames)
    report=dict(measurement_kind='offline_mesh_probe',particles=n,frame=args.frames,
        attached=int((frame.arrays['state']==0).sum()),free=int((frame.arrays['state']==1).sum()),
        cache_frame_bytes=reader.manifest['frames'][str(args.frames)]['bytes'],results=results,memory=memory_snapshot())
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')


main()
