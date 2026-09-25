"""Measure sequential simulation plus realized drop output in a clean Blender process."""
import argparse
from collections import Counter
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import bpy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flumen.build_simulation import create_simulation_host


def peak_memory_bytes():
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=ctypes.c_void_p
        api=ctypes.WinDLL('psapi').GetProcessMemoryInfo
        api.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        if api(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            return counters.PeakWorkingSetSize
        return None
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:
        return None


def main():
    parser=argparse.ArgumentParser()
    for key,default in [('particles',512),('faces',2000),('frames',240),('fps',24),('substeps',8)]:
        parser.add_argument('--'+key,type=int,default=default)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if min(args.particles,args.faces,args.frames,args.fps,args.substeps)<=0:
        parser.error('All counts must be positive')
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene; scene.render.fps=args.fps; scene.frame_end=args.frames
    # UV sphere triangulated by the simulation, close to requested face count.
    segments=max(8,round(math.sqrt(args.faces))); rings=max(4,round(args.faces/(2*segments))+1)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,radius=1)
    source=bpy.context.object
    source.data.calc_loop_triangles(); actual_faces=len(source.data.loop_triangles)
    host=create_simulation_host(source,**{'Particle Budget':args.particles,'Seed Density':10000,
        'Source Start':0,'Source Softness':0,
        'Minimum Substeps':args.substeps,'Lifetime':100,'Kill Height':-1000})
    mod=host.modifiers[0]; simulation=mod.node_group
    # Append ONE isolated diagnostic vertex after the rendered mesh; no second simulation.
    wrapper=bpy.data.node_groups.new('Benchmark Output','GeometryNodeTree')
    wrapper.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    node=wrapper.nodes.new('GeometryNodeGroup'); node.node_tree=simulation
    for s in simulation.interface.items_tree:
        if s.item_type=='SOCKET' and s.in_out=='INPUT':
            node.inputs[s.name].default_value=getattr(mod.properties.inputs,s.identifier).value
    points=wrapper.nodes.new('GeometryNodePointsToVertices')
    wrapper.links.new(node.outputs['Diagnostics'],points.inputs['Points'])
    join=wrapper.nodes.new('GeometryNodeJoinGeometry')
    wrapper.links.new(node.outputs['Geometry'],join.inputs['Geometry'])
    wrapper.links.new(points.outputs[0],join.inputs['Geometry'])
    output=wrapper.nodes.new('NodeGroupOutput'); wrapper.links.new(join.outputs[0],output.inputs[0])
    mod.node_group=wrapper
    durations=[]; samples=[]; errors=[]; particle_max=0; limited_max=0
    for frame in range(1,args.frames+1):
        start=time.perf_counter(); scene.frame_set(frame)
        evaluated=host.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=evaluated.to_mesh()
        durations.append((time.perf_counter()-start)*1000)
        diagnostic_index=next((i for i,v in enumerate(mesh.attributes['sf_initial_volume'].data) if v.value>0),None)
        if diagnostic_index is None: raise RuntimeError('Benchmark has no seeded diagnostic volume')
        values={name:mesh.attributes[name].data[diagnostic_index].value for name in
            ('sf_initial_volume','sf_live_volume','sf_removed_volume','sf_count','sf_substeps','sf_limited_count')}
        total=values['sf_initial_volume']
        errors.append(abs(total-values['sf_live_volume']-values['sf_removed_volume'])/total if total else 0)
        particle_max=max(particle_max,round(values['sf_count']))
        limited_max=max(limited_max,round(values['sf_limited_count']))
        if frame>1: samples.append(round(values['sf_substeps']))
        evaluated.to_mesh_clear()
        if frame%60==0: print(f'BENCHMARK frame {frame}/{args.frames}',flush=True)
    ordered=sorted(durations[1:] or durations)
    report={'blender':bpy.app.version_string,'build_hash':bpy.app.build_hash.decode(),
        'cpu':platform.processor(),'platform':platform.platform(),'logical_cpus':os.cpu_count(),
        'schema':simulation['sf_schema'],'requested_particles':args.particles,'particle_max':particle_max,
        'source_start':0,'source_softness':0,'seed_density':10000,
        'target_faces':args.faces,'actual_triangles':actual_faces,'frames':args.frames,'fps':args.fps,
        'minimum_substeps':args.substeps,'actual_substeps':dict(Counter(samples)),
        'timing_scope':'simulation + realized render mesh + one diagnostic vertex; excludes metric readback',
        'first_frame_ms':durations[0],'median_ms':statistics.median(ordered),
        'p95_ms':ordered[math.ceil(.95*len(ordered))-1],'max_ms':max(ordered),
        'peak_process_memory_bytes':peak_memory_bytes(),'max_relative_volume_error':max(errors),
        'max_limited_particles':limited_max,'trail_max':0,'trails_implemented':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
