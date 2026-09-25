import bpy
from bpy.types import Panel


class SF_PT_panel(Panel):
    bl_label = "Flumen"
    bl_idname = "SF_PT_flumen"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Flumen"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            layout.label(text="Select a mesh object.", icon='INFO')
            return

        if obj.get('sf_simulation_host'):
            layout.label(text='Animated Flow', icon='TIME')
            layout.label(text='Stationary collision surfaces only.')
            layout.label(text='Play from the start frame.')
            layout.label(text='Save, then use Simulation Nodes bake.')
            if obj.parent or any(abs(v) > 1e-8 for v in obj.location) or any(abs(v-1) > 1e-8 for v in obj.scale) or any(abs(v)>1e-8 for v in obj.rotation_euler):
                layout.label(text='Keep this host at identity transforms.', icon='ERROR')
        else:
            layout.label(text='Static Paths')
            layout.operator("flumen.build", icon='NODETREE')
            layout.operator("flumen.rebuild", icon='FILE_REFRESH')
            layout.separator()
            layout.label(text='Animated Flow')
            layout.operator('flumen.create_simulation', icon='TIME')
        layout.separator()
        layout.label(text="Tune parameters in the modifier.")
