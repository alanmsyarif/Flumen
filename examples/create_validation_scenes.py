"""Create six reproducible, packed-bake M1 scenes and a viewport-style preview."""
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flumen.build_simulation import create_simulation_host


def mesh(name,vertices,faces):
    data=bpy.data.meshes.new(name); data.from_pydata(vertices,[],faces); data.update()
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj)
    return obj


def bottle():
    profile=[(.55,-1),(.65,-.85),(.65,.3),(.6,.5),(.28,.75),(.28,1),(.36,1),(.36,1.12),(.23,1.12),(.23,.88)]
    vertices=[(r*math.cos(i*math.tau/64),r*math.sin(i*math.tau/64),z) for r,z in profile for i in range(64)]
    faces=[(j*64+i,j*64+(i+1)%64,(j+1)*64+(i+1)%64,(j+1)*64+i)
        for j in range(len(profile)-1) for i in range(64)]
    return mesh('Bottle rim',vertices,faces)


def main():
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    target=artifacts/'Flumen_M1_Demo.blend'
    bpy.context.preferences.filepaths.save_version=0
    bpy.context.preferences.filepaths.file_preview_type='NONE'
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    stone=bpy.data.materials.new('Ceramic'); stone.diffuse_color=(.34,.38,.44,1)
    water=bpy.data.materials.new('Water display'); water.diffuse_color=(.02,.65,1,1)
    settings={'Seed Density':1000,'Particle Budget':512,'Drop Radius':.018,'Max Travel':.02,
              'Capture Distance':.025,'Resistance':4,'Adhesion Acceleration':6,'Lifetime':10}
    scenes=[]
    for index,name in enumerate(('Sphere','Slope','Bottle rim','Opposing sheets','Concave fold','Suzanne')):
        scene=bpy.context.scene if index==0 else bpy.data.scenes.new(name)
        bpy.context.window.scene=scene; scene.name=name
        scene.frame_start=1; scene.frame_end=72; scene.render.fps=24
        scene.render.engine='BLENDER_WORKBENCH'; scene.render.resolution_x=900
        scene.render.resolution_y=900; scene.render.resolution_percentage=100
        scene.display.shading.light='STUDIO'; scene.display.shading.color_type='MATERIAL'
        scene.display.shading.show_shadows=True; scene.display.shading.show_cavity=True
        scene.display.shading.background_type='WORLD'; scene.world=bpy.data.worlds.new(name+' world')
        scene.world.color=(.055,.055,.055)
        if name=='Sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24); source=bpy.context.object
        elif name=='Slope':
            source=mesh(name,[(-1,-1,.8),(1,-1,.8),(1,1,-.4),(-1,1,-.4)],[(0,1,2,3)])
        elif name=='Bottle rim': source=bottle()
        elif name=='Opposing sheets':
            source=mesh(name,[(-1,-.07,-1),(1,-.07,-1),(1,-.07,1),(-1,-.07,1),
                             (-1,.07,-1),(1,.07,-1),(1,.07,1),(-1,.07,1)],[(3,2,1,0),(4,5,6,7)])
        elif name=='Concave fold':
            source=mesh(name,[(-1,-1,1),(0,-1,-.3),(1,-1,1),(-1,1,1),(0,1,-.3),(1,1,1)],[(0,1,4,3),(1,2,5,4)])
        else:
            bpy.ops.mesh.primitive_monkey_add(); source=bpy.context.object
            sub=source.modifiers.new('Smooth collision','SUBSURF'); sub.levels=2
        source.name=name+' collision'; source.data.materials.append(stone)
        for polygon in source.data.polygons: polygon.use_smooth=True
        host=create_simulation_host(source,**settings); host.name=name+' water'
        mod=host.modifiers[0]; mod.bake_target='PACKED'
        socket=next(s for s in mod.node_group.interface.items_tree if s.name=='Flow Material' and s.in_out=='INPUT')
        getattr(mod.properties.inputs,socket.identifier).value=water
        camera_data=bpy.data.cameras.new(name+' camera'); camera=bpy.data.objects.new(name+' camera',camera_data)
        scene.collection.objects.link(camera); camera.location=(3.5,-6,2.8)
        if name in ('Slope','Concave fold'): camera.location=(3.5,-6,6)
        camera.rotation_euler=(Vector((0,0,-.35))-camera.location).to_track_quat('-Z','Y').to_euler()
        camera_data.type='ORTHO'; camera_data.ortho_scale=3.8; scene.camera=camera
        scene['Flumen notes']='M1 procedural particles. Stationary sources; no wetness, merging, or liquid film. Packed frames 1-72.'
        for obj in bpy.context.selected_objects: obj.select_set(False)
        host.select_set(True); bpy.context.view_layer.objects.active=host
        scenes.append((scene,host))
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    for scene,host in scenes:
        bpy.context.window.scene=scene
        with bpy.context.temp_override(object=host,active_object=host,selected_objects=[host],selected_editable_objects=[host]):
            result=bpy.ops.object.simulation_nodes_cache_bake(selected=False)
        if 'FINISHED' not in result: raise RuntimeError('Bake failed: '+scene.name)
        scene.frame_set(30)
        print('BAKED',scene.name,flush=True)
    bpy.context.window.scene=scenes[0][0]
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_perspective='CAMERA'
                area.spaces.active.shading.color_type='MATERIAL'
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    bpy.context.scene.render.filepath=str(artifacts/'Flumen_M1_Preview.png')
    bpy.ops.render.render(write_still=True)
    print('DEMO_SAVED',target,flush=True)


if __name__=='__main__': main()
