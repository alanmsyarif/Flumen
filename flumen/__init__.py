bl_info = {
    "name": "Flumen",
    "author": "Alan Syarif / project scaffold",
    "version": (0, 0, 2),
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
    CLASSES = (SF_OT_build, SF_OT_rebuild, SF_OT_create_simulation, SF_PT_panel)
else:
    CLASSES = ()


def register():
    if bpy is None:
        raise RuntimeError("Flumen register() must run inside Blender")
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    if bpy is None:
        return
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
