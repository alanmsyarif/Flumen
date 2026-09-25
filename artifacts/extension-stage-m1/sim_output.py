"""Water-only display geometry; appearance never feeds back into particle state."""
from math import pi
import bpy
from .sim_nodes import Graph, owned_group


def build_output_group():
    old=owned_group('SF_Output')
    if old:
        return old
    g=Graph('SF_Output',[('Particles','NodeSocketGeometry'),('Flow Material','NodeSocketMaterial')],
            [('Geometry','NodeSocketGeometry')])
    radius=g.math('POWER',g.math('MULTIPLY',g.named('sf_volume'),3/(4*pi)),1/3)
    attached=g.math('LESS_THAN',g.named('sf_state','INT'),.5)
    offset=g.switch('VECTOR',attached,(0,0,0),g.vector('SCALE',g.named('sf_normal','FLOAT_VECTOR'),radius))
    positions=g.node('GeometryNodeSetPosition')
    g.connect(g.input.outputs['Particles'],positions,'Geometry'); g.connect(offset,positions,'Offset')
    sphere=g.node('GeometryNodeMeshIcoSphere')
    g.connect(1.0,sphere,'Radius'); g.connect(2,sphere,'Subdivisions')
    instance=g.node('GeometryNodeInstanceOnPoints')
    g.connect(positions.outputs['Geometry'],instance,'Points'); g.connect(sphere.outputs['Mesh'],instance,'Instance')
    g.connect(g.vector('SCALE',(1,1,1),radius),instance,'Scale')
    realize=g.node('GeometryNodeRealizeInstances'); g.connect(instance.outputs['Instances'],realize,'Geometry')
    material=g.node('GeometryNodeSetMaterial')
    g.connect(realize.outputs['Geometry'],material,'Geometry'); g.connect(g.input.outputs['Flow Material'],material,'Material')
    return g.finish({'Geometry':material.outputs['Geometry']})
