"""Generate a replayable GPU preview scene; requires Warp in the current environment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps')]
import bpy
from mathutils import Quaternion
import flumen
from flumen.gpu_runtime import create_gpu_host,reset_host,release_all


def create_demo(capacity=8192,benchmark=False):
    release_all()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj,do_unlink=True)
    if not hasattr(bpy.types.Object,'flumen_gpu'): flumen.register()
    scene=bpy.context.scene
    scene.frame_start=1; scene.frame_end=720; scene.render.fps=30
    scene.frame_set(1)
    scene.sync_mode='NONE'
    if benchmark:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=102,ring_count=100,radius=1)
    else:
        bpy.ops.mesh.primitive_monkey_add()
        bpy.context.object.scale=(.08,.08,.08)
        modifier=bpy.context.object.modifiers.new('Smooth Surface','SUBSURF')
        modifier.levels=2
    source=bpy.context.object; source.name='Collision Surface'
    for p in source.data.polygons: p.use_smooth=True
    source.color=(.15,.16,.19,1)
    host=create_gpu_host(source,scene)
    cfg=host.flumen_gpu
    cfg.capacity=capacity; cfg.source_start=0 if benchmark else .72
    cfg.source_softness=0 if benchmark else .08
    cfg.kill_height=-1000 if benchmark else -.25
    cfg.emission_end=720
    cfg.radius=.001
    reset_host(host)
    for obj in bpy.context.selected_objects: obj.select_set(False)
    host.select_set(True); bpy.context.view_layer.objects.active=host
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active
                space.shading.type='SOLID'; space.shading.color_type='OBJECT'
                space.overlay.show_overlays=False
                space.region_3d.view_distance=4 if benchmark else .35
                space.region_3d.view_location=(0,0,0 if benchmark else -.035)
                space.region_3d.view_rotation=Quaternion((.888,.325,.116,.304)).normalized()
    return scene,host


if __name__=='__main__':
    scene,host=create_demo()
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'artifacts'/'Flumen_GPU_Demo.blend'))
    print('GPU_DEMO_SAVED',host.name)
