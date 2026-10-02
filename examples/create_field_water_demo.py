"""Stationary Suzanne coated on its upper part with one million Surface Field particles.

Mirrors the reference bust setup: water starts on the crown and drains down the face.
Run inside Blender with the Flumen add-on registered (see scripts/capture_field_drainage.py).
"""
import bpy
from flumen import gpu_runtime as runtime


def create_field_demo(count=1_000_000, source_start=.55, source_softness=.05):
    scene = bpy.context.scene
    scene.render.fps = 30; scene.frame_start = 1; scene.frame_set(1)
    for obj in [o for o in scene.objects if o.type == 'MESH']: bpy.data.objects.remove(obj, do_unlink=True)
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
    with bpy.context.temp_override(window=window, area=area):
        bpy.ops.mesh.primitive_monkey_add()
    source = bpy.context.view_layer.objects.active
    source.name = 'Suzanne 0.3 m'; source.scale = (.115,)*3; source.location.z = .15
    source.modifiers.new('Fixture', 'SUBSURF').levels = 2
    source.color = (.08, .08, .08, 1)
    bpy.context.view_layer.update()
    host = runtime.create_gpu_host(source, scene, display_mode='POINTS', solver_backend='FIELD')
    settings = host.flumen_gpu
    for name, value in dict(capacity=count, initial_coating_count=count, particles_per_frame=0,
            source_start=source_start, source_softness=source_softness, radius=.0001, time_scale=.5,
            lifetime=1000, kill_height=-10000, resistance=60, field_spacing=.002, contact_spacing=.002,
            resample_target=count, minimum_substeps=8, point_size=2., point_color=(.85, .9, 1., 1.)).items():
        setattr(settings, name, value)
    runtime.reset_host(host)
    space = area.spaces.active
    space.shading.type = 'SOLID'; space.shading.color_type = 'OBJECT'; space.shading.light = 'STUDIO'
    space.shading.background_type = 'VIEWPORT'; space.shading.background_color = (0., 0., 0.)
    space.overlay.show_overlays = False; space.clip_start = .001
    return scene, host
