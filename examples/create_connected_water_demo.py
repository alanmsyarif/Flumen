"""Reproducible stationary Suzanne fixture for Connected water validation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps')]
import bpy
from mathutils import Vector
import flumen
from flumen.gpu_runtime import create_gpu_host,reset_host,get_runtime


def look_at(obj,point):
    obj.rotation_euler=(Vector(point)-obj.location).to_track_quat('-Z','Y').to_euler()


def create_connected_demo(view='WATER',emission_end=120):
    if view not in ('WATER','GEOMETRY'): raise ValueError('Unknown material view')
    if not hasattr(bpy.types.Object,'flumen_gpu'): flumen.register()
    scene=bpy.data.scenes.new('Flumen Connected Validation')
    if bpy.context.window: bpy.context.window.scene=scene
    scene.frame_start=1; scene.frame_end=max(180,emission_end); scene.render.fps=30
    scene.render.engine='BLENDER_EEVEE'; scene.sync_mode='NONE'
    if hasattr(scene.eevee,'use_raytracing'): scene.eevee.use_raytracing=True
    scene.render.resolution_x=1920; scene.render.resolution_y=1080; scene.render.resolution_percentage=100
    scene.view_settings.view_transform='AgX'
    scene.world=bpy.data.worlds.new('Connected Dark Environment'); scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.035,.045,.065,1.)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
    bpy.ops.mesh.primitive_monkey_add(location=(0,0,.15))
    source=bpy.context.object; source.name='Connected Suzanne Collision'
    source.scale=(.115,.115,.115)
    modifier=source.modifiers.new('Reference Surface Resolution','SUBSURF'); modifier.levels=2
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    for face in source.data.polygons: face.use_smooth=True
    material=bpy.data.materials.new('Neutral Ceramic'); material.use_nodes=True
    shader=material.node_tree.nodes['Principled BSDF']
    shader.inputs['Base Color'].default_value=(.19,.23,.28,1.); shader.inputs['Roughness'].default_value=.4
    source.data.materials.append(material); source.color=(.19,.23,.28,1.)
    # Construct the inexpensive legacy host before applying the full Connected settings.
    host=create_gpu_host(source,scene)
    cfg=host.flumen_gpu
    for name,value in dict(display_mode='CONNECTED',interactions_enabled=True,capacity=8192,
        initial_coating_count=4096,particles_per_frame=64,emission_end=emission_end,
        time_scale=.5,source_start=.45,source_softness=.08,radius=.001,lifetime=4.,
        resistance=60.,kill_height=-.12,reconstruction_scale=1.).items():
        setattr(cfg,name,value)
    reset_host(host)
    if view=='GEOMETRY':
        neutral=bpy.data.materials.new('Neutral Liquid Geometry'); neutral.use_nodes=True
        node=neutral.node_tree.nodes['Principled BSDF']
        node.inputs['Base Color'].default_value=(.08,.45,.62,1.); node.inputs['Roughness'].default_value=.28
        cfg.material=neutral
    host.color=(.08,.45,.62,1.)
    bpy.ops.object.camera_add(location=(.36,-.65,.29))
    camera=bpy.context.object; camera.name='Connected Fixed Camera'; look_at(camera,(0,0,.11))
    camera.data.type='ORTHO'; camera.data.ortho_scale=.75; camera.data.clip_start=.001; camera.data.clip_end=3.
    look_at(camera,(0,0,.085))
    scene.camera=camera
    for name,position,power,size,color in (
        ('Reflection Key',(.18,-.35,.55),6.,.45,(.8,.9,1.)),
        ('Reflection Rim',(-.35,.18,.4),8.,.35,(.5,.75,1.)),
        ('Reflection Fill',(.4,.25,.2),3.,.3,(1.,.85,.65))):
        bpy.ops.object.light_add(type='AREA',location=position)
        light=bpy.context.object; light.name=name; light.data.energy=power; light.data.shape='DISK'
        light.data.size=size; light.data.color=color; look_at(light,(0,0,.15))
    scene['sf_connected_fixture']='SUZANNE_0P3M'; scene['sf_connected_view']=view
    source.data.calc_loop_triangles(); scene['sf_source_triangles']=len(source.data.loop_triangles)
    scene['sf_source_dimensions']=source.dimensions[:]
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space=area.spaces.active; space.clip_start=.001; space.clip_end=3.
                space.shading.type='RENDERED' if view=='WATER' else 'SOLID'
                space.shading.color_type='OBJECT'; space.shading.light='STUDIO'
                space.overlay.show_overlays=False; space.region_3d.view_perspective='CAMERA'
                space.region_3d.view_camera_zoom=0.
    for obj in bpy.context.selected_objects: obj.select_set(False)
    host.select_set(True); bpy.context.view_layer.objects.active=host
    print('CONNECTED_FIXTURE',scene['sf_source_triangles'],source.dimensions[:],
          len(host.data.vertices),get_runtime(host,scene).stats,flush=True)
    return scene,host


if __name__=='__main__':
    scene,host=create_connected_demo()
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'artifacts/Flumen_Connected_Water_Demo.blend'))
