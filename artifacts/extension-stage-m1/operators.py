from __future__ import annotations

import bpy
from bpy.types import Operator

from .build_nodes import build_flumen_group
from .constants import MIN_BLENDER_VERSION, MODIFIER_NAME


class SF_OT_build(Operator):
    bl_idname = "flumen.build"
    bl_label = "Build Flumen"
    bl_description = "Create/rebuild the Flumen Geometry Nodes setup on the active mesh"
    bl_options = {'REGISTER', 'UNDO'}

    force_rebuild: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        if bpy.app.version < MIN_BLENDER_VERSION:
            self.report({'ERROR'}, f"Flumen requires Blender {MIN_BLENDER_VERSION[0]}.{MIN_BLENDER_VERSION[1]}+")
            return {'CANCELLED'}

        obj = context.active_object
        try:
            tree = build_flumen_group(force_rebuild=self.force_rebuild)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        modifier = next((m for m in obj.modifiers if m.type == 'NODES' and m.node_group == tree), None)
        if modifier is None:
            modifier = obj.modifiers.new(name=MODIFIER_NAME, type='NODES')
        modifier.node_group = tree

        self.report({'INFO'}, "Flumen MVP created. Adjust inputs in the Geometry Nodes modifier.")
        return {'FINISHED'}


class SF_OT_rebuild(Operator):
    bl_idname = "flumen.rebuild"
    bl_label = "Rebuild Node Group"
    bl_description = "Rebuild FL_SurfaceFlow from the current implementation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            build_flumen_group(force_rebuild=True)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "FL_SurfaceFlow rebuilt")
        return {'FINISHED'}


class SF_OT_create_simulation(Operator):
    bl_idname = 'flumen.create_simulation'
    bl_label = 'Create Animated Flow'
    bl_description = 'Create bakeable surface particles and drips on a stationary collision mesh'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj=context.active_object
        return obj is not None and obj.type=='MESH' and not obj.get('sf_simulation_host')

    def execute(self, context):
        from .build_simulation import create_simulation_host
        try:
            host=create_simulation_host(context.active_object)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}
        for obj in context.selected_objects:
            obj.select_set(False)
        host.select_set(True)
        context.view_layer.objects.active=host
        self.report({'INFO'},'Animated flow created. Play from the start frame; save the file before baking.')
        return {'FINISHED'}
