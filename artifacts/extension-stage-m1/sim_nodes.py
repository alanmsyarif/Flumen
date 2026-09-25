"""Small socket-explicit building primitives used by the simulation groups."""
import bpy
import json
from .sim_schema import SIM_SCHEMA_VERSION
from .node_utils import add_interface_socket, resolve_socket


def interface_signature(tree):
    return json.dumps([(s.in_out,s.name,s.socket_type) for s in tree.interface.items_tree if s.item_type=='SOCKET'])


def owned_group(name):
    tree=bpy.data.node_groups.get(name)
    if tree and (not tree.get('sf_owned') or tree.get('sf_schema')!=SIM_SCHEMA_VERSION
                 or tree.get('sf_interface')!=interface_signature(tree)):
        raise ValueError(f'{name} is an incompatible existing group; rename it before creating Flumen')
    return tree


class Graph:
    def __init__(self, name, inputs, outputs):
        self.tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
        self.tree['sf_owned'] = True
        for entry in outputs:
            add_interface_socket(self.tree, entry[0], 'OUTPUT', entry[1])
        for entry in inputs:
            name, kind, *options = entry
            default, low, high = (options + [None]*3)[:3]
            add_interface_socket(self.tree, name, 'INPUT', kind, default=default, min_value=low, max_value=high)
        self.input = self.node('NodeGroupInput', 'Inputs')
        self.output = self.node('NodeGroupOutput', 'Outputs')
        self.output.is_active_output = True

    def node(self, kind, label=None, **properties):
        node = self.tree.nodes.new(kind)
        if label:
            node.name = node.label = label
        for key, value in properties.items():
            setattr(node, key, value)
        count = len(self.tree.nodes)
        node.location = (count % 8 * 210, -(count // 8)*240)
        return node

    def put(self, value, socket):
        if isinstance(value, bpy.types.NodeSocket):
            self.tree.links.new(value, socket)
        else:
            socket.default_value = value

    def connect(self, value, node, name):
        self.put(value, resolve_socket(node.inputs, name=name))

    def math(self, operation, a, b=None, c=None):
        n = self.node('ShaderNodeMath', operation, operation=operation)
        for index, value in enumerate((a,b,c)):
            if value is not None:
                self.put(value, n.inputs[index])
        return n.outputs[0]

    def vector(self, operation, a, b=None):
        n = self.node('ShaderNodeVectorMath', operation, operation=operation)
        self.put(a, n.inputs[0])
        if b is not None:
            self.put(b, n.inputs[3 if operation == 'SCALE' else 1])
        return n.outputs['Value' if operation in {'DOT_PRODUCT','LENGTH','DISTANCE'} else 'Vector']

    def boolean(self, operation, a, b=None):
        n = self.node('FunctionNodeBooleanMath', operation, operation=operation)
        self.put(a, n.inputs[0])
        if b is not None:
            self.put(b, n.inputs[1])
        return n.outputs[0]

    def integer(self, operation, a, b):
        n = self.node('FunctionNodeIntegerMath', operation, operation=operation)
        self.put(a,n.inputs[0]); self.put(b,n.inputs[1])
        return n.outputs[0]

    def switch(self, kind, condition, false, true):
        n = self.node('GeometryNodeSwitch', input_type=kind)
        for name, value in [('Switch',condition),('False',false),('True',true)]:
            self.connect(value,n,name)
        return n.outputs[0]

    def named(self, name, kind='FLOAT'):
        n = self.node('GeometryNodeInputNamedAttribute', name, data_type=kind)
        self.connect(name,n,'Name')
        return n.outputs['Attribute']

    def store(self, geometry, name, value, kind='FLOAT'):
        n = self.node('GeometryNodeStoreNamedAttribute', 'Store '+name, data_type=kind, domain='POINT')
        for key, data in [('Geometry',geometry),('Name',name),('Value',value)]:
            self.connect(data,n,key)
        return n.outputs['Geometry']

    def commit(self, geometry, values, position=None):
        """Capture every expression before changing attributes they depend on."""
        for name, (kind,value) in values.items():
            geometry = self.store(geometry, '_sf_next_'+name, value, kind)
        if position is not None:
            geometry = self.store(geometry, '_sf_next_position', position, 'FLOAT_VECTOR')
        for name, (kind,_) in values.items():
            geometry = self.store(geometry, name, self.named('_sf_next_'+name,kind), kind)
        if position is not None:
            n = self.node('GeometryNodeSetPosition')
            self.connect(geometry,n,'Geometry')
            self.connect(self.named('_sf_next_position','FLOAT_VECTOR'),n,'Position')
            geometry = n.outputs['Geometry']
        for name in [*('_sf_next_'+s for s in values), *(['_sf_next_position'] if position is not None else [])]:
            n = self.node('GeometryNodeRemoveAttribute')
            self.connect(geometry,n,'Geometry'); self.connect(name,n,'Name')
            geometry = n.outputs['Geometry']
        return geometry

    def sample(self, mesh, value, kind, position, island=None):
        n = self.node('GeometryNodeSampleNearestSurface', data_type=kind)
        for key,data in [('Mesh',mesh),('Value',value),('Sample Position',position)]:
            self.connect(data,n,key)
        if island is not None:
            islands = self.node('GeometryNodeInputMeshIsland').outputs['Island Index']
            self.connect(islands,n,'Group ID'); self.connect(island,n,'Sample Group ID')
        return n.outputs['Value'], n.outputs['Is Valid']

    def statistic(self, mesh, value, output='Max', domain='POINT'):
        n = self.node('GeometryNodeAttributeStatistic', data_type='FLOAT', domain=domain)
        self.connect(mesh,n,'Geometry'); self.connect(value,n,'Attribute')
        return n.outputs[output]

    def group(self, tree, values, label=None):
        n = self.node('GeometryNodeGroup', label)
        n.node_tree = tree
        for name,value in values.items():
            self.connect(value,n,name)
        return n

    def finish(self, values):
        for name,value in values.items():
            self.connect(value,self.output,name)
        self.tree['sf_schema']=SIM_SCHEMA_VERSION
        self.tree['sf_interface']=interface_signature(self.tree)
        return self.tree


def tangent(g, vector, normal):
    return g.vector('SUBTRACT', vector, g.vector('SCALE',normal,g.vector('DOT_PRODUCT',vector,normal)))
