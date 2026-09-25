"""One-time, budget-limited surface particle initialization."""
from math import pi
import bpy
from .sim_nodes import Graph, owned_group
from .sim_schema import CONTROLS, PARTICLE_ATTRIBUTES


def build_source_group():
    existing = owned_group('SF_Source')
    if existing:
        return existing
    names = {'Gravity','Source Start','Source Softness','Seed Density','Seed','Particle Budget','Drop Radius'}
    g = Graph('SF_Source', [('Collision','NodeSocketGeometry'), *(c for c in CONTROLS if c[0] in names)],
              [('Particles','NodeSocketGeometry')])
    i = g.input.outputs
    position = g.node('GeometryNodeInputPosition').outputs[0]
    up = g.vector('SCALE', g.vector('NORMALIZE',i['Gravity']), -1.0)
    height = g.vector('DOT_PRODUCT',position,up)
    low = g.statistic(i['Collision'],height,'Min')
    high = g.statistic(i['Collision'],height,'Max')
    span = g.math('SUBTRACT',high,low)
    normalized = g.math('DIVIDE',g.math('SUBTRACT',height,low),g.math('MAXIMUM',span,1e-8))
    remap = g.node('ShaderNodeMapRange', interpolation_type='SMOOTHERSTEP', clamp=True)
    for name,value in [('Value',normalized),('From Min',g.math('SUBTRACT',i['Source Start'],i['Source Softness'])),
                       ('From Max',g.math('ADD',i['Source Start'],i['Source Softness'])),('To Min',0.0),('To Max',1.0)]:
        g.connect(value,remap,name)
    mask = g.switch('FLOAT',g.math('LESS_THAN',i['Source Softness'],1e-8),remap.outputs['Result'],
                    g.math('GREATER_THAN',normalized,i['Source Start']))
    area = g.node('GeometryNodeInputMeshFaceArea').outputs[0]
    # Full area is conservative regardless of the mask's corner interpolation.
    source_area = g.statistic(i['Collision'],area,'Sum','FACE')
    budget = g.math('MINIMUM',2048,g.math('MAXIMUM',1,i['Particle Budget']))
    density = g.math('MINIMUM',g.math('MAXIMUM',0,i['Seed Density']),
                     g.math('DIVIDE',budget,g.math('MAXIMUM',source_area,1e-8)))
    seeds = g.node('GeometryNodeDistributePointsOnFaces', distribute_method='RANDOM')
    g.connect(i['Collision'],seeds,'Mesh')
    g.connect(g.math('GREATER_THAN',span,1e-8),seeds,'Selection')
    g.connect(g.math('MULTIPLY',density,mask),seeds,'Density')
    g.connect(i['Seed'],seeds,'Seed')
    index = g.node('GeometryNodeInputIndex').outputs[0]
    normal = g.vector('NORMALIZE',seeds.outputs['Normal'])
    island, _ = g.sample(i['Collision'],g.node('GeometryNodeInputMeshIsland').outputs[0], 'INT',position)
    volume = g.math('MULTIPLY',4*pi/3,g.math('POWER',i['Drop Radius'],3))
    values = {'sf_id': index,'sf_age': 0.0, 'sf_velocity': (0,0,0), 'sf_volume': volume,
              'sf_state': 0,'sf_normal': normal,'sf_anchor': position,'sf_island': island,
              'sf_path': index,'sf_blocked': False,'sf_step_limited': False}
    geometry = g.commit(seeds.outputs['Points'],{name:(PARTICLE_ATTRIBUTES[name],value) for name,value in values.items()})
    delete = g.node('GeometryNodeDeleteGeometry', domain='POINT')
    g.connect(geometry,delete,'Geometry')
    g.connect(g.math('GREATER_THAN',index,g.math('SUBTRACT',budget,1)),delete,'Selection')
    return g.finish({'Particles':delete.outputs['Geometry']})
