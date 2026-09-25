"""Animated Geometry Nodes graph. All per-frame updates live in the node graph."""
import bpy
from .sim_nodes import Graph, owned_group
from .sim_schema import CONTROLS, SIM_NODE_GROUP_NAME, SIM_SCHEMA_VERSION
from .sim_source import build_source_group
from .sim_attached import build_attached_group
from .sim_free import build_free_group
from .sim_output import build_output_group
from .build_nodes import ensure_flow_material


def build_simulation_group(force_rebuild=False):
    existing = owned_group(SIM_NODE_GROUP_NAME)
    if existing:
        if force_rebuild:
            raise ValueError('Create a new simulation host to replace its state schema; existing bakes are preserved')
        if existing.get('sf_schema') != SIM_SCHEMA_VERSION or not existing.get('sf_owned'):
            raise ValueError('Simulation group name is occupied by an incompatible graph')
        return existing
    before=set(bpy.data.node_groups)
    try:
        return _build_simulation_group()
    except Exception:
        # Only remove groups created by this failed transaction, including children.
        for tree in list(bpy.data.node_groups):
            if tree not in before:
                bpy.data.node_groups.remove(tree,do_unlink=True)
        raise


def _build_simulation_group():
    g = Graph(SIM_NODE_GROUP_NAME, [('Collision Object','NodeSocketObject'), *CONTROLS,
              ('Flow Material','NodeSocketMaterial',ensure_flow_material())],
              [('Geometry','NodeSocketGeometry'),('Particles','NodeSocketGeometry'),('Diagnostics','NodeSocketGeometry')])
    i = g.input.outputs
    info = g.node('GeometryNodeObjectInfo', 'Collision Object', transform_space='RELATIVE')
    g.connect(i['Collision Object'],info,'Object')
    g.connect(False,info,'As Instance')
    realize = g.node('GeometryNodeRealizeInstances')
    g.connect(info.outputs['Geometry'],realize,'Geometry')
    triangulate = g.node('GeometryNodeTriangulate', 'Collision Mesh')
    g.connect(realize.outputs['Geometry'],triangulate,'Mesh')
    collision = triangulate.outputs['Mesh']
    source_tree = build_source_group()
    source = g.group(source_tree, {'Collision':collision, **{s.name:i[s.name] for s in source_tree.interface.items_tree
                     if s.item_type=='SOCKET' and s.in_out=='INPUT' and s.name!='Collision'}},'Initial Particles')
    sim_out = g.node('GeometryNodeSimulationOutput', 'Simulation State')
    sim_in = g.node('GeometryNodeSimulationInput', 'Simulation Start')
    sim_in.pair_with_output(sim_out)
    sim_out.state_items.clear()
    for kind,name in [('GEOMETRY','Particles'),('FLOAT','Elapsed'),('FLOAT','Removed Volume'),('INT','Next Path ID'),('FLOAT','Initial Volume'),('FLOAT','Substeps')]:
        sim_out.state_items.new(kind,name)
    g.connect(source.outputs['Particles'],sim_in,'Particles')
    g.connect(g.statistic(source.outputs['Particles'],g.named('sf_volume'),'Sum'),sim_in,'Initial Volume')
    g.connect(g.math('ADD',1,g.statistic(source.outputs['Particles'],g.named('sf_id','INT'),'Max')),sim_in,'Next Path ID')
    h_frame = g.math('MAXIMUM',sim_in.outputs['Delta Time'],0)
    max_speed = g.math('MAXIMUM',0,g.statistic(sim_in.outputs['Particles'],g.vector('LENGTH',g.named('sf_velocity','FLOAT_VECTOR'))))
    bound = g.math('ADD',max_speed,g.math('MULTIPLY',g.vector('LENGTH',i['Gravity']),h_frame))
    needed = g.math('CEIL',g.math('DIVIDE',g.math('MULTIPLY',bound,h_frame),g.math('MAXIMUM',i['Max Travel'],1e-5)))
    steps = g.math('MINIMUM',64,g.math('MAXIMUM',1,g.math('MAXIMUM',needed,i['Minimum Substeps'])))
    dt = g.math('DIVIDE',h_frame,steps)
    repeat_out = g.node('GeometryNodeRepeatOutput','Substeps End')
    repeat_in = g.node('GeometryNodeRepeatInput','Substeps Start')
    repeat_in.pair_with_output(repeat_out)
    repeat_out.repeat_items.clear()
    for kind,name in [('GEOMETRY','Particles'),('FLOAT','Removed Volume'),('INT','Next Path ID')]:
        repeat_out.repeat_items.new(kind,name)
        g.connect(sim_in.outputs[name],repeat_in,name)
    g.connect(steps,repeat_in,'Iterations')
    split = g.node('GeometryNodeSeparateGeometry','Original Substep State',domain='POINT')
    g.connect(repeat_in.outputs['Particles'],split,'Geometry')
    g.connect(g.math('LESS_THAN',g.named('sf_state','INT'),.5),split,'Selection')
    def step_group(tree, geometry, label):
        values = {'Particles':geometry,'Collision':collision,'dt':dt}
        for socket in tree.interface.items_tree:
            if socket.item_type == 'SOCKET' and socket.in_out == 'INPUT' and socket.name not in values:
                values[socket.name] = i[socket.name]
        return g.group(tree,values,label)
    attached = step_group(build_attached_group(),split.outputs['Selection'],'Attached Motion')
    free = step_group(build_free_group(),split.outputs['Inverted'],'Free Motion')
    join = g.node('GeometryNodeJoinGeometry')
    g.connect(attached.outputs['Particles'],join,'Geometry')
    g.connect(free.outputs['Particles'],join,'Geometry')
    particles = join.outputs['Geometry']
    transitions = g.named('_sf_transition','BOOLEAN')
    accumulate = g.node('GeometryNodeAccumulateField',data_type='INT',domain='POINT')
    g.connect(transitions,accumulate,'Value')
    new_id = g.integer('ADD',repeat_in.outputs['Next Path ID'],accumulate.outputs['Trailing'])
    next_id = g.integer('ADD',repeat_in.outputs['Next Path ID'],g.statistic(particles,transitions,'Sum'))
    particles = g.store(particles,'sf_path',g.switch('INT',transitions,g.named('sf_path','INT'),new_id),'INT')
    particles = g.store(particles,'sf_step_limited',g.boolean('OR',g.named('sf_step_limited','BOOLEAN'),g.math('GREATER_THAN',needed,64)),'BOOLEAN')
    position = g.node('GeometryNodeInputPosition').outputs[0]
    xyz = g.node('ShaderNodeSeparateXYZ'); g.connect(position,xyz,'Vector')
    expired = g.boolean('AND',g.math('GREATER_THAN',dt,0),g.boolean('OR',
        g.boolean('NOT',g.math('LESS_THAN',g.named('sf_age'),i['Lifetime'])),g.math('LESS_THAN',xyz.outputs['Z'],i['Kill Height'])))
    removed = g.statistic(particles,g.switch('FLOAT',expired,0,g.named('sf_volume')),'Sum')
    delete = g.node('GeometryNodeDeleteGeometry',domain='POINT')
    g.connect(particles,delete,'Geometry'); g.connect(expired,delete,'Selection')
    g.connect(delete.outputs['Geometry'],repeat_out,'Particles')
    g.connect(g.math('ADD',repeat_in.outputs['Removed Volume'],g.math('ADD',removed,free.outputs['Removed Volume'])),repeat_out,'Removed Volume')
    g.connect(next_id,repeat_out,'Next Path ID')
    for name in ('Particles','Removed Volume','Next Path ID'):
        g.connect(repeat_out.outputs[name],sim_out,name)
    g.connect(g.math('ADD',sim_in.outputs['Elapsed'],sim_in.outputs['Delta Time']),sim_out,'Elapsed')
    g.connect(sim_in.outputs['Initial Volume'],sim_out,'Initial Volume')
    g.connect(steps,sim_out,'Substeps')
    diagnostic = g.node('GeometryNodePoints','Diagnostics')
    g.connect(1,diagnostic,'Count')
    particles = sim_out.outputs['Particles']
    diagnostics = diagnostic.outputs['Points']
    for name,value in {
        'sf_initial_volume':sim_out.outputs['Initial Volume'],
        'sf_removed_volume':sim_out.outputs['Removed Volume'],
        'sf_live_volume':g.statistic(particles,g.named('sf_volume'),'Sum'),
        'sf_elapsed':sim_out.outputs['Elapsed'],
        'sf_substeps':sim_out.outputs['Substeps'],
        'sf_count':g.statistic(particles,1,'Sum'),
        'sf_free_count':g.statistic(particles,g.named('sf_state','INT'),'Sum'),
        'sf_blocked_count':g.statistic(particles,g.named('sf_blocked','BOOLEAN'),'Sum'),
        'sf_limited_count':g.statistic(particles,g.named('sf_step_limited','BOOLEAN'),'Sum'),
    }.items():
        diagnostics = g.store(diagnostics,name,value)
    g.tree['sf_schema'] = SIM_SCHEMA_VERSION
    display=g.group(build_output_group(),{'Particles':particles,'Flow Material':i['Flow Material']},'Water Display')
    return g.finish({'Geometry':display.outputs['Geometry'],'Particles':particles,'Diagnostics':diagnostics})


def create_simulation_host(source_object, **settings):
    """Create a world-space water host without changing the user's source object."""
    from math import isfinite
    if bpy.app.version < (5,2,0):
        raise ValueError('Animated Flumen requires Blender 5.2 or newer')
    if source_object is None or source_object.type != 'MESH' or source_object.get('sf_simulation_host'):
        raise ValueError('Select a collision mesh, not a Flumen simulation host')
    if abs(bpy.context.scene.unit_settings.scale_length-1.0) > 1e-8:
        raise ValueError('Set Scene Unit Scale to 1.0; Flumen uses one meter per Blender unit')
    evaluated=source_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if len(evaluated.data.polygons) == 0:
        raise ValueError('Collision mesh must contain faces')
    controls={c[0]:c for c in CONTROLS}
    for name,value in settings.items():
        if name not in controls:
            raise ValueError(f'Unknown simulation setting: {name}')
        _,kind,_,low,high=controls[name]
        components=tuple(value) if kind == 'NodeSocketVector' else (value,)
        if kind == 'NodeSocketVector' and len(components) != 3:
            raise ValueError(f'{name} needs three components')
        if any(not isfinite(float(v)) or low is not None and v < low or high is not None and v > high for v in components):
            raise ValueError(f'{name} is outside its finite supported range')
        if kind == 'NodeSocketInt' and int(value) != value:
            raise ValueError(f'{name} must be an integer')
    from mathutils import Vector
    gravity=Vector(settings.get('Gravity',controls['Gravity'][2]))
    if gravity.length < 1e-8:
        raise ValueError('Source height requires nonzero Gravity')
    up=-gravity.normalized()
    heights=[(evaluated.matrix_world @ vertex.co).dot(up) for vertex in evaluated.data.vertices]
    if max(heights)-min(heights)<=1e-8:
        raise ValueError('Source has no height variation along Gravity; tilt the mesh or change Gravity')
    tree=build_simulation_group()
    mesh=bpy.data.meshes.new('Flumen Simulation')
    host=bpy.data.objects.new('Flumen Simulation',mesh)
    bpy.context.collection.objects.link(host)
    host['sf_simulation_host']=True
    host['sf_schema']=SIM_SCHEMA_VERSION
    mod=host.modifiers.new('Animated Flumen','NODES')
    mod.node_group=tree
    values={'Collision Object':source_object,**settings}
    for socket in tree.interface.items_tree:
        if socket.item_type=='SOCKET' and socket.in_out=='INPUT' and socket.name in values:
            getattr(mod.properties.inputs,socket.identifier).value=values[socket.name]
    host.update_tag(); bpy.context.view_layer.update()
    return host
