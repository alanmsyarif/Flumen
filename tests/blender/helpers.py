"""Fixtures and output inspection for real node evaluation."""
import bpy


def reset_scene():
    previews = bpy.context.preferences.filepaths.bl_rna.properties['file_preview_type'].enum_items
    if 'NONE' in previews:
        bpy.context.preferences.filepaths.file_preview_type = 'NONE'
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for group in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(group, do_unlink=True)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250
    bpy.context.scene.render.fps = 24
    bpy.context.scene.render.fps_base = 1.0
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.context.scene.frame_set(1)


def input_socket(tree, name):
    return next(s for s in tree.interface.items_tree
                if s.item_type == 'SOCKET' and s.in_out == 'INPUT' and s.name == name)


def set_input_value(modifier, name, value):
    item = input_socket(modifier.node_group, name)
    getattr(modifier.properties.inputs, item.identifier).value = value
    modifier.id_data.update_tag()
    bpy.context.view_layer.update()


def read_input_value(modifier, name):
    return getattr(modifier.properties.inputs, input_socket(modifier.node_group, name).identifier).value


def interface_ids(tree):
    return {(s.in_out, s.name): s.identifier for s in tree.interface.items_tree if s.item_type == 'SOCKET'}


def mesh_object(vertices, faces, name='Fixture'):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def debug_modifier(obj, tree, output_name, settings=None):
    """Keep the tested tree intact; choose a geometry output in a test wrapper."""
    wrapper = bpy.data.node_groups.new('Test output wrapper', 'GeometryNodeTree')
    wrapper.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    wrapper.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = wrapper.nodes.new('NodeGroupInput')
    go = wrapper.nodes.new('NodeGroupOutput')
    group = wrapper.nodes.new('GeometryNodeGroup')
    group.node_tree = tree
    if 'Geometry' in group.inputs:
        wrapper.links.new(gi.outputs['Geometry'], group.inputs['Geometry'])
    for name, value in (settings or {}).items():
        group.inputs[name].default_value = value
    vertices = wrapper.nodes.new('GeometryNodePointsToVertices')
    wrapper.links.new(group.outputs[output_name], vertices.inputs['Points'])
    wrapper.links.new(vertices.outputs['Mesh'], go.inputs['Geometry'])
    mod = obj.modifiers.new('Test', 'NODES')
    mod.node_group = wrapper
    return mod, group


def read_points(obj):
    obj.update_tag()
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        result = []
        attrs = [a for a in mesh.attributes if a.domain == 'POINT' and a.name.startswith('sf_')]
        for i, vertex in enumerate(mesh.vertices):
            record = {'position': tuple(vertex.co)}
            for attr in attrs:
                item = attr.data[i]
                record[attr.name] = tuple(item.vector) if attr.data_type == 'FLOAT_VECTOR' else item.value
            result.append(record)
        return result
    finally:
        evaluated.to_mesh_clear()


def particle_at(position, *, volume=1e-9, velocity=(0,0,0), state=0, normal=(0,1,0), island=0, age=0):
    return {'position': position, 'sf_velocity': velocity, 'sf_volume':volume, 'sf_state':state,
            'sf_normal':normal,'sf_anchor':position,'sf_island':island,'sf_age':age,
            'sf_path':0,'sf_id':0,'sf_blocked':False,'sf_step_limited':False}


def particles_object(records):
    from flumen.sim_schema import PARTICLE_ATTRIBUTES
    obj = mesh_object([r['position'] for r in records], [], 'Test particles')
    for name, kind in PARTICLE_ATTRIBUTES.items():
        attr = obj.data.attributes.new(name, kind, 'POINT')
        for index, record in enumerate(records):
            value = record.get(name, (0,0,0) if kind == 'FLOAT_VECTOR' else 0)
            if kind == 'FLOAT_VECTOR':
                attr.data[index].vector = value
            else:
                attr.data[index].value = value
    return obj


def step_fixture(tree, records, collision, **settings):
    obj = particles_object(records)
    mod, group = debug_modifier(obj, tree, 'Particles', settings)
    wrapper = mod.node_group
    gi = next(n for n in wrapper.nodes if n.type == 'GROUP_INPUT')
    points = wrapper.nodes.new('GeometryNodeMeshToPoints'); points.mode = 'VERTICES'
    wrapper.links.new(gi.outputs['Geometry'], points.inputs['Mesh'])
    wrapper.links.new(points.outputs['Points'], group.inputs['Particles'])
    info = wrapper.nodes.new('GeometryNodeObjectInfo'); info.transform_space = 'RELATIVE'
    info.inputs['Object'].default_value = collision
    wrapper.links.new(info.outputs['Geometry'], group.inputs['Collision'])
    return read_points(obj)


def vertical_plane():
    return mesh_object([(-2,0,-2),(-2,0,2),(2,0,2),(2,0,-2)], [(0,1,2,3)], 'Vertical plane')


def simulation_fixture(records, collision, **settings):
    from flumen.build_simulation import build_simulation_group
    tree = build_simulation_group().copy()
    initial = particles_object(records)
    info = tree.nodes.new('GeometryNodeObjectInfo'); info.transform_space = 'RELATIVE'
    info.inputs['Object'].default_value = initial
    points = tree.nodes.new('GeometryNodeMeshToPoints'); points.mode = 'VERTICES'
    tree.links.new(info.outputs['Geometry'],points.inputs['Mesh'])
    sim_in = tree.nodes['Simulation Start']
    tree.links.new(points.outputs['Points'],sim_in.inputs['Particles'])
    for name,value in [('Next Path ID', max((p['sf_id'] for p in records),default=-1)+1),
                       ('Initial Volume',sum(p['sf_volume'] for p in records))]:
        if name in sim_in.inputs:
            for item in list(sim_in.inputs[name].links):
                tree.links.remove(item)
            sim_in.inputs[name].default_value = value
    host=mesh_object([],[],'Simulation fixture')
    debug_modifier(host,tree,'Particles',{'Collision Object':collision,**settings})
    return host


def step_frames(host, start, end):
    frames=[]
    for frame in range(start,end+1):
        bpy.context.scene.frame_set(frame)
        frames.append(read_points(host))
    return frames


def reset_simulation(host):
    with bpy.context.temp_override(object=host, active_object=host, selected_objects=[host], selected_editable_objects=[host]):
        bpy.ops.object.simulation_nodes_cache_delete(selected=False)
    bpy.context.scene.frame_set(1)


def read_diagnostics(host):
    wrapper=host.modifiers[0].node_group
    group=next(n for n in wrapper.nodes if n.type=='GROUP')
    points=next(n for n in wrapper.nodes if n.bl_idname=='GeometryNodePointsToVertices')
    wrapper.links.new(group.outputs['Diagnostics'],points.inputs['Points'])
    try:
        return read_points(host)[0]
    finally:
        wrapper.links.new(group.outputs['Particles'],points.inputs['Points'])
