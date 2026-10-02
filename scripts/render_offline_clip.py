"""Bake, mesh offline and render opaque + water clips of the stationary drainage fixture (background Blender).

blender --background --factory-startup --python scripts/render_offline_clip.py -- --work DIR --prefix artifacts/offline-clip
"""
from math import radians
from mathutils import Euler, Vector
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'scripts'),str(ROOT/'examples')]
import bpy
import flumen
from flumen import gpu_runtime as runtime
from flumen.gpu_bake import BakeJob
from flumen.particle_cache import CacheReader
from flumen.offline_mesher import MeshOptions, iter_mesh_cache
from flumen.gpu_baked_display import create_baked_water, wet_proxy
from create_field_water_demo import create_field_demo
from gpu_memory import memory_snapshot

STILLS=(1,30,60,90,120,180)


def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def stage(scene, water, source):
    camera=bpy.data.objects.new('Clip Camera',bpy.data.cameras.new('Clip Camera'))
    scene.collection.objects.link(camera); scene.camera=camera
    rotation=Euler((radians(82),0,radians(25)))
    camera.rotation_euler=rotation; camera.location=Vector((0,0,.14))+rotation.to_matrix()@Vector((0,0,.62))
    light=bpy.data.objects.new('Key',bpy.data.lights.new('Key','AREA')); light.data.energy=60; light.data.size=.6
    light.location=(.4,-.5,.6); light.rotation_euler=(radians(40),0,radians(35)); scene.collection.objects.link(light)
    world=scene.world or bpy.data.worlds.new('World'); scene.world=world
    world.use_nodes=True; world.node_tree.nodes['Background'].inputs['Color'].default_value=(.02,.02,.02,1)
    scene.render.resolution_x,scene.render.resolution_y=1280,720
    # The wet proxy carries the source surface in render; the original stays untouched but hidden.
    source.hide_render=True
    source.color=wet_proxy(water).color=(.08,.08,.08,1); water.color=(.85,.9,1.,1.)


def render(scene, engine, path, stills, frames_dir, shutter=0.):
    """Render every frame to PNG, keep the still frames, then encode an MP4 with system ffmpeg."""
    import subprocess
    scene.render.engine=engine
    if engine=='BLENDER_WORKBENCH':
        scene.display.shading.light='STUDIO'; scene.display.shading.color_type='OBJECT'
    if engine=='CYCLES':
        preferences=bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type='CUDA'; preferences.get_devices()
        scene.cycles.device='GPU'; scene.cycles.samples=64
    # Cycles blurs drops from the mesh 'velocity' attribute; EEVEE ignores it.
    scene.render.use_motion_blur=shutter > 0.; scene.render.motion_blur_shutter=shutter or .5
    frames_dir.mkdir(parents=True,exist_ok=True)
    scene.render.image_settings.file_format='PNG'
    scene.render.filepath=str(frames_dir/'f####')
    started=time.perf_counter()
    bpy.ops.render.render(animation=True)
    seconds=time.perf_counter()-started
    for frame in stills:
        image=frames_dir/f'f{frame:04d}.png'
        if image.exists(): shutil.copyfile(image,path.with_name(f'{path.stem}-f{frame:03d}.png'))
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(scene.render.fps),
                    '-i',str(frames_dir/'f%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p','-crf','18',
                    str(path.with_suffix('.mp4'))],check=True)
    return seconds


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--count',type=int,default=250_000)
    parser.add_argument('--frames',type=int,default=180)
    parser.add_argument('--drop-spacing',type=float,default=.0001)
    parser.add_argument('--water-scale',type=float,default=1.,help='Particle radius multiplier on top of the 1M-volume radius')
    parser.add_argument('--film-spacing',type=float,default=.002)
    parser.add_argument('--film-smoothing',type=int,default=20)
    parser.add_argument('--film-max-thickness',type=float,default=.002)
    parser.add_argument('--drop-kernel',choices=['velocity','pca'],default='pca')
    parser.add_argument('--film-sheen',type=float,default=2e-5)
    parser.add_argument('--contact-hysteresis',type=float,default=0.,help='Dry-surface contact-line pinning (rivulets); 0 = off')
    parser.add_argument('--water-engine',choices=['eevee','cycles'],default='eevee')
    parser.add_argument('--shutter',type=float,default=0.,help='Motion blur shutter in frames (Cycles; 0 = off)')
    parser.add_argument('--free-crop',type=float,nargs=6,default=[-.3,-.3,-.02,.3,.3,.4],
                        help='Mesh free drops only inside x0 y0 z0 x1 y1 z1 (meters); out-of-shot drops are reported')
    parser.add_argument('--workers',type=int,default=max(1,min(12,(os.cpu_count() or 2)//2)))
    parser.add_argument('--reuse-cache',action='store_true',help='Mesh an existing complete cache instead of rebaking')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    flumen.register()
    scene,host=create_field_demo(count=args.count)
    # Keep the 1M fixture's total water volume at a reduced particle count.
    radius=.0001*(1_000_000/args.count)**(1/3)*args.water_scale
    host.flumen_gpu.radius=radius; host.flumen_gpu.contact_hysteresis=args.contact_hysteresis; runtime.reset_host(host)
    bust=bpy.data.materials.new('Bust'); bust.diffuse_color=(.08,.08,.08,1)
    bust.use_nodes=True; bust.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.06,.06,.06,1)
    host.flumen_gpu.source.data.materials.append(bust)
    scene.frame_end=args.frames
    source=host.flumen_gpu.source
    report=dict(measurement_kind='offline_clip',particles=args.count,frames=args.frames,particle_radius=radius,water_scale=args.water_scale,
                drop_spacing=args.drop_spacing,film_spacing=args.film_spacing,film_smoothing=args.film_smoothing,free_crop=args.free_crop,
                film_max_thickness=args.film_max_thickness,drop_kernel=args.drop_kernel,film_sheen=args.film_sheen,
                contact_hysteresis=args.contact_hysteresis,shutter=args.shutter,water_engine=args.water_engine)
    if args.reuse_cache:
        cached=CacheReader(args.work/'cache')
        if cached.header.frame_count!=args.frames: raise ValueError('Existing cache has a different frame range')
        report.update(bake_seconds=None,cache_reused=True,cache_bytes=folder_bytes(args.work/'cache'))
    else:
        t=time.perf_counter(); job=BakeJob(host,scene,args.work/'cache')
        while not job.step(): pass
        report.update(bake_seconds=time.perf_counter()-t,cache_bytes=folder_bytes(args.work/'cache'))
    bpy.data.objects.remove(host,do_unlink=True); runtime.release_all()
    reader=CacheReader(args.work/'cache'); t=time.perf_counter(); frames=[]
    report['workers']=args.workers
    for result in iter_mesh_cache(reader,MeshOptions(spacing=args.drop_spacing,film_spacing=args.film_spacing,film_smoothing=args.film_smoothing,
                                               free_crop=(tuple(args.free_crop[:3]),tuple(args.free_crop[3:])),
                                               film_max_thickness=args.film_max_thickness,drop_kernel=args.drop_kernel,film_sheen=args.film_sheen),args.work/'mesh',args.workers):
        frames.append(result)
        if result['frame'] in STILLS: print('MESHED',json.dumps({k:result[k] for k in ('frame','seconds','attached_triangles','free_triangles')}),flush=True)
    report.update(mesh_seconds=time.perf_counter()-t,mesh_bytes=folder_bytes(args.work/'mesh'),mesh_frames=frames)
    water=create_baked_water(args.work/'cache',args.work/'mesh',scene,source=source)
    stage(scene,water,source)
    report['opaque_render_seconds']=render(scene,'BLENDER_WORKBENCH',args.prefix.with_name(args.prefix.name+'-opaque'),STILLS,args.work/'opaque_frames')
    report['water_render_seconds']=render(scene,'CYCLES' if args.water_engine=='cycles' else 'BLENDER_EEVEE',
                                         args.prefix.with_name(args.prefix.name+'-water'),STILLS,args.work/'water_frames',args.shutter)
    report['memory']=memory_snapshot()
    args.prefix.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('OFFLINE_CLIP',json.dumps({k:v for k,v in report.items() if k not in ('mesh_frames','memory')}),flush=True)


main()
