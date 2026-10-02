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
            layout.operator('flumen.create_field_preview',icon='POINTCLOUD_DATA')
            return
        settings=host.flumen_gpu
        layout.label(text='Stationary surface • live preview')
        layout.prop(settings,'source')
        layout.operator('flumen.reset_gpu_flow',icon='FILE_REFRESH')
        if settings.solver_backend=='FIELD':
            layout.operator('flumen.bake_particle_cache',icon='FILE_CACHE')
            if host.get('sf_particle_cache'):
                layout.label(text=f"Cache: {host['sf_particle_cache']}",icon='CHECKMARK')
                layout.operator('flumen.mesh_particle_cache',icon='MOD_FLUIDSIM')
        error=host.get('sf_gpu_error','')
        if error:
            box=layout.box(); box.alert=True
            box.label(text=error,icon='ERROR')
        layout.label(text=host.get('sf_gpu_device','CUDA not initialized'))
        geometry_error=host.get('sf_gpu_geometry_error','')
        if geometry_error:
            box=layout.box(); box.alert=True
            box.label(text=geometry_error,icon='ERROR')
        layout.prop(settings,'display_mode')
        if settings.display_mode=='POINTS':
            for name in ('solver_backend','point_style','display_limit'):
                layout.prop(settings,name)
            for name in (('water_color','water_smoothing','water_radius_scale') if settings.point_style=='WATER'
                         else ('point_size','point_color')):
                layout.prop(settings,name)
        if settings.solver_backend=='FIELD':
            for name in ('field_spacing','contact_spacing','field_viscosity','surface_tension','resample_target'):
                layout.prop(settings,name)
        elif settings.interactions_enabled:
            layout.label(text='Pairwise backend: bounded neighbors, not for million-particle editing',icon='INFO')
        layout.prop(settings,'interactions_enabled')
        for name in ('mode','particles_per_frame' if settings.mode=='CONTINUOUS' else 'burst_count',
                     'emission_start','emission_end','initial_coating_count','time_scale','capacity','lifetime','seed','radius',
                     'source_start','source_softness','gravity','resistance','adhesion',
                     'capture_distance','capture_speed','minimum_substeps','max_travel',
                     'normal_turn_limit','kill_height','material'):
            layout.prop(settings,name)
        if settings.display_mode=='CONNECTED' or settings.interactions_enabled:
            for name in ('interaction_radius_scale','cohesion_acceleration','repulsion_acceleration',
                         'surface_damping','merge_distance_scale','maximum_merged_radius_scale',
                         'reconstruction_scale','wetness_deposit_rate','wetness_drying_rate'):
                layout.prop(settings,name)
        from .gpu_runtime import RUNTIMES
        record=RUNTIMES.get(host.as_pointer())
        if record and record.solver.stats:
            s=record.solver.stats
            layout.label(text=f'{s.live_count:,} live • {s.accepted:,} emitted')
            layout.label(text=f'{s.capacity_rejected:,} capacity / {s.source_rejected:,} source rejected')
            layout.label(text=f'{s.solver_ms:.2f} ms solver • {s.transfer_ms:.2f} ms transfer')
            layout.label(text=f'{s.substeps} substeps • {s.limited_count} limited')
            prepared=record.solver.prepared
            if prepared is not None:
                layout.label(text=f'{prepared.chart.operator_spacing*1000:.2f} mm field • '
                                  f'{prepared.contact.effective_spacing*1000:.2f} mm contact spacing')
                layout.label(text=f'{s.contact_fallback_count:,} contact fallback • {s.attached_count:,} attached / {s.free_count:,} free')
            if settings.display_mode=='POINTS':
                layout.label(text=f'{s.live_count:,} simulated • {s.displayed_count:,} displayed')
                layout.label(text=f'{s.readback_ms:.2f} readback • {s.upload_ms:.2f} upload • {s.draw_ms:.2f} draw ms')
            if settings.display_mode=='CONNECTED':
                layout.label(text=f'{s.water_vertices:,} water vertices / {s.water_triangles:,} triangles')
                layout.label(text=f'{s.reconstruction_ms:.2f} ms reconstruction')
                layout.label(text=f'{s.coarsening_factor:.1f}x resolution / {s.neighbor_overflow:,} neighbor overflow')
                layout.label(text=f'{s.unrepresented_volume:.3g} m³ unsupported / {s.rendered_volume_error:.1%} mesh volume error')
