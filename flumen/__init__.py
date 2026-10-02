bl_info = {
    "name": "Flumen",
    "author": "Alan Syarif / project scaffold",
    "version": (0, 0, 3),
    "blender": (5, 2, 0),
    "location": "3D View > Sidebar > Flumen",
    "description": "Procedural surface drainage paths using Geometry Nodes",
    "category": "Node",
}

# Keep pure-python helper modules importable outside Blender for unit tests.
try:
    import bpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - normal outside Blender
    bpy = None

if bpy is not None:
    from .operators import SF_OT_build, SF_OT_rebuild, SF_OT_create_simulation
    from .ui import SF_PT_panel
    from .gpu_properties import SF_GPUSettings
    from .gpu_operators import SF_OT_create_gpu_flow, SF_OT_create_field_preview, SF_OT_reset_gpu_flow
    from .gpu_bake import SF_OT_bake_particle_cache, SF_OT_mesh_particle_cache
    from .gpu_ui import SF_PT_gpu
    CLASSES = (SF_OT_build, SF_OT_rebuild, SF_OT_create_simulation, SF_PT_panel,
               SF_GPUSettings, SF_OT_create_gpu_flow, SF_OT_create_field_preview,
               SF_OT_reset_gpu_flow, SF_OT_bake_particle_cache,
               SF_OT_mesh_particle_cache, SF_PT_gpu)
else:
    CLASSES = ()


def register():
    if bpy is None:
        raise RuntimeError("Flumen register() must run inside Blender")
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.flumen_gpu = bpy.props.PointerProperty(type=SF_GPUSettings)
    from .gpu_runtime import register_handlers
    register_handlers()


def unregister():
    if bpy is None:
        return
    from .gpu_runtime import unregister_handlers
    unregister_handlers()
    if hasattr(bpy.types.Object,'flumen_gpu'):
        del bpy.types.Object.flumen_gpu
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
