"""Background evaluation benchmark, explicitly excluding viewport redraw."""
from pathlib import Path
import sys
import json
import time
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'examples'),str(ROOT/'scripts')]
import bpy
from create_gpu_demo import create_demo
from flumen.gpu_runtime import get_runtime,release_all
from gpu_report import summarize
from gpu_memory import memory_snapshot


def run(capacity=8192,frames=600):
    start=time.perf_counter()
    scene,host=create_demo(capacity,benchmark=True)
    for frame in range(1,121): scene.frame_set(frame)
    setup=time.perf_counter()-start
    samples=[]; stages=[]
    start=time.perf_counter()
    for frame in range(121,121+frames):
        tick=time.perf_counter()
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples.append((time.perf_counter()-tick)*1000)
        solver=get_runtime(host,scene)
        stages.append(asdict(solver.stats))
    wall=time.perf_counter()-start
    report=summarize('background_evaluation',samples,frames,0,wall,0,0)
    solver=get_runtime(host,scene)
    report.update(setup_seconds=setup,capacity=capacity,device=asdict(solver.device),
                  config=asdict(solver.config),blender=bpy.app.version_string,
                  samples_ms=samples,stages=stages,final_stats=asdict(solver.stats))
    report['triangles']=len(host.flumen_gpu.source.data.polygons)*2
    report['memory']=memory_snapshot()
    output=ROOT/'artifacts'/f'benchmark-gpu-{capacity}.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print('GPU_BENCHMARK',capacity,{k:report[k] for k in ('median_ms','p95_ms','max_ms','average_fps')},flush=True)
    release_all()
    return report


if __name__=='__main__':
    for capacity in (2048,8192,32768): run(capacity)
