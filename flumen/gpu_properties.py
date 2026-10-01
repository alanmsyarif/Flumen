"""Persistent Blender settings; runtime/device resources live elsewhere."""
from dataclasses import fields
from struct import pack, unpack
import bpy
from .gpu.config import FlowConfig


def changed(self, context):
    from .gpu_runtime import mark_dirty
    mark_dirty(self.id_data)


def material_changed(self, context):
    from .gpu_display import set_material
    set_material(self.id_data,self.material)


class SF_GPUSettings(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(name='Collision Surface',type=bpy.types.Object,update=changed)
    material: bpy.props.PointerProperty(name='Water Material',type=bpy.types.Material,update=material_changed)


_bounds = {
    'particles_per_frame':(0,2**31-1),'burst_count':(0,2**31-1),'capacity':(1,1000000),
    'seed':(0,2**31-1),'emission_start':(-1000000,1000000),'emission_end':(-1000000,1000000),
    'minimum_substeps':(1,64),'radius':(.00001,.1),'lifetime':(.01,1000),
    'kill_height':(-10000,10000),'source_start':(0,1),'source_softness':(0,.5),
    'resistance':(0,1000),'adhesion':(0,1000),'capture_distance':(.00001,.1),
    'capture_speed':(0,100),'max_travel':(.00001,.1),'normal_turn_limit':(1,89),
    'time_scale':(.01,2),'initial_coating_count':(0,2**31-1),
    'interaction_radius_scale':(1,16),'cohesion_acceleration':(0,100),
    'repulsion_acceleration':(0,100),'surface_damping':(0,1000),
    'merge_distance_scale':(0,1),'maximum_merged_radius_scale':(1,8),
    'reconstruction_scale':(.5,4),'wetness_deposit_rate':(0,1000),'wetness_drying_rate':(0,1000),
    'field_spacing':(.00005,.02),'contact_spacing':(.00005,.05),
    'field_viscosity':(0,.01),'surface_tension':(0,1),'resample_target':(0,1000000),
}
for field in fields(FlowConfig):
    opts={'name':field.name.replace('_',' ').title(),'default':field.default,'update':changed}
    if field.name == 'mode':
        prop=bpy.props.EnumProperty(items=[('BURST','Burst','Emit once'),('CONTINUOUS','Continuous','Emit every eligible frame')],**opts)
    elif field.name == 'display_mode':
        prop=bpy.props.EnumProperty(items=[('DROPS','Drops','Diagnostic particle display'),('CONNECTED','Connected Water','Reconstructed liquid surface'),('POINTS','Points','GPU particle preview')],**opts)
    elif field.name == 'solver_backend':
        prop=bpy.props.EnumProperty(items=[('LEGACY','Legacy','Independent or pairwise particle solver'),('FIELD','Surface Field','Bounded surface-field interactions')],**opts)
    elif isinstance(field.default,bool):
        prop=bpy.props.BoolProperty(**opts)
    elif field.name == 'gravity':
        prop=bpy.props.FloatVectorProperty(size=3,**opts)
    else:
        low,high=_bounds[field.name]
        prop=(bpy.props.IntProperty if isinstance(field.default,int) else bpy.props.FloatProperty)(min=low,max=high,**opts)
    SF_GPUSettings.__annotations__[field.name]=prop


def config_for(host):
    settings=host.flumen_gpu
    values={field.name:getattr(settings,field.name) for field in fields(FlowConfig)}
    # RNA stores float32: recover only exact endpoint representations, without
    # relaxing validation for arbitrary out-of-range or nonfinite inputs.
    for field in fields(FlowConfig):
        if isinstance(field.default,float):
            for endpoint in _bounds[field.name]:
                if values[field.name] == unpack('f',pack('f',endpoint))[0]:
                    values[field.name]=endpoint
                    break
    values['gravity']=tuple(values['gravity'])
    config=FlowConfig(**values)
    config.validate()
    return config
