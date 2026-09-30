import bpy


class SF_OT_create_gpu_flow(bpy.types.Operator):
    bl_idname='flumen.create_gpu_flow'
    bl_label='Create GPU Flow'
    bl_description='Create live CUDA surface particles with continuous frame emission'
    bl_options={'REGISTER','UNDO'}

    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return obj is not None and obj.type=='MESH' and not obj.get('sf_gpu_host') and not obj.get('sf_simulation_host')

    def execute(self,context):
        from .gpu_runtime import create_gpu_host
        try:
            host=create_gpu_host(context.active_object,context.scene)
        except (ValueError,RuntimeError,ImportError,OSError) as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        for obj in context.selected_objects: obj.select_set(False)
        host.select_set(True); context.view_layer.objects.active=host
        return {'FINISHED'}


class SF_OT_reset_gpu_flow(bpy.types.Operator):
    bl_idname='flumen.reset_gpu_flow'
    bl_label='Reset GPU Flow'
    bl_description='Rebuild collision data and replay to the current integer frame'

    def execute(self,context):
        from .gpu_runtime import reset_host
        try:
            reset_host(context.active_object)
        except (ValueError,RuntimeError,ImportError,OSError) as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}
