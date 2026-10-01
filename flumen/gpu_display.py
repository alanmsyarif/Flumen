"""Bulk point-mesh transfer and display-only instanced drops."""
import bpy
import numpy as np


def create_display(host):
    tree=bpy.data.node_groups.new('Flumen GPU Drops','GeometryNodeTree')
    tree['sf_gpu_display']=True
    tree.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    tree.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes,links=tree.nodes,tree.links
    gi=nodes.new('NodeGroupInput'); go=nodes.new('NodeGroupOutput')
    sphere=nodes.new('GeometryNodeMeshIcoSphere')
    sphere.inputs['Radius'].default_value=1
    sphere.inputs['Subdivisions'].default_value=1
    material=nodes.new('GeometryNodeSetMaterial'); material.name='Water Material'
    links.new(sphere.outputs['Mesh'],material.inputs['Geometry'])
    attr=nodes.new('GeometryNodeInputNamedAttribute'); attr.data_type='FLOAT'
    attr.inputs['Name'].default_value='sf_radius'
    instance=nodes.new('GeometryNodeInstanceOnPoints')
    links.new(gi.outputs['Geometry'],instance.inputs['Points'])
    links.new(material.outputs['Geometry'],instance.inputs['Instance'])
    links.new(attr.outputs['Attribute'],instance.inputs['Scale'])
    links.new(instance.outputs['Instances'],go.inputs['Geometry'])
    modifier=host.modifiers.new('GPU Drop Display','NODES'); modifier.node_group=tree
    host.color=(.05,.45,.8,1)


def set_material(host, material):
    if host.flumen_gpu.display_mode=='CONNECTED':
        if host.data.users>1: host.data=host.data.copy()
        host.data.materials.clear()
        if material is not None: host.data.materials.append(material)
        return
    for modifier in host.modifiers:
        if modifier.type=='NODES' and modifier.node_group and modifier.node_group.get('sf_gpu_display'):
            if modifier.node_group.users>1:
                modifier.node_group=modifier.node_group.copy()
            modifier.node_group.nodes['Water Material'].inputs['Material'].default_value=material


def update_display(host,batch):
    # Duplicated hosts must not overwrite another host's point storage.
    if host.data.users>1:
        host.data=host.data.copy()
    mesh=host.data
    count=len(batch.ids)
    if len(mesh.vertices)!=count:
        mesh.clear_geometry()
        if count:
            mesh.vertices.add(count)
    if count:
        mesh.vertices.foreach_set('co',batch.positions.reshape(-1))
    for name,kind,values in [('sf_radius','FLOAT',batch.radii),
                             ('sf_id_low','INT',(batch.ids & 0x7fffffff).astype(np.int32)),
                             ('sf_id_high','INT',(batch.ids >> 31).astype(np.int32))]:
        attribute=mesh.attributes.get(name) or mesh.attributes.new(name,kind,'POINT')
        if count:
            attribute.data.foreach_set('value',values)
    mesh.update()
