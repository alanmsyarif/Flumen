import bpy


class SF_PT_gpu(bpy.types.Panel):
    bl_label='GPU Flow'
    bl_idname='SF_PT_gpu'
    bl_space_type='VIEW_3D'
    bl_region_type='UI'
    bl_category='Flumen'

    def draw(self,context):
        layout=self.layout; host=context.active_object
        if not host or not host.get('sf_gpu_host'):
            layout.operator('flumen.create_gpu_flow',icon='PARTICLES')
            return
        settings=host.flumen_gpu
        layout.label(text='Stationary surface • live preview')
        layout.prop(settings,'source')
        layout.operator('flumen.reset_gpu_flow',icon='FILE_REFRESH')
        error=host.get('sf_gpu_error','')
        if error:
            box=layout.box(); box.alert=True
            box.label(text=error,icon='ERROR')
        layout.label(text=host.get('sf_gpu_device','CUDA not initialized'))
        for name in ('mode','particles_per_frame' if settings.mode=='CONTINUOUS' else 'burst_count',
                     'emission_start','emission_end','capacity','lifetime','seed','radius',
                     'source_start','source_softness','gravity','resistance','adhesion',
                     'capture_distance','capture_speed','minimum_substeps','max_travel',
                     'normal_turn_limit','kill_height','material'):
            layout.prop(settings,name)
        from .gpu_runtime import RUNTIMES
        record=RUNTIMES.get(host.as_pointer())
        if record and record.solver.stats:
            s=record.solver.stats
            layout.label(text=f'{s.live_count:,} live • {s.accepted:,} emitted')
            layout.label(text=f'{s.capacity_rejected:,} capacity / {s.source_rejected:,} source rejected')
            layout.label(text=f'{s.solver_ms:.2f} ms solver • {s.transfer_ms:.2f} ms transfer')
            layout.label(text=f'{s.substeps} substeps • {s.limited_count} limited')
