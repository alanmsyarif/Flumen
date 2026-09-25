"""Ballistic drops with swept-segment collision and an explicit volume ledger."""
from math import pi
import bpy
from .sim_nodes import Graph, tangent, owned_group
from .sim_schema import CONTROLS


def build_free_group():
    old = owned_group('SF_FreeStep')
    if old:
        return old
    controls = {'Gravity','Capture Speed','Max Travel','Lifetime','Kill Height'}
    g = Graph('SF_FreeStep', [('Particles','NodeSocketGeometry'),('Collision','NodeSocketGeometry'),
                 ('dt','NodeSocketFloat',0.0,0.0,1.0), *(c for c in CONTROLS if c[0] in controls)],
              [('Particles','NodeSocketGeometry'),('Removed Volume','NodeSocketFloat')])
    i = g.input.outputs
    h = g.math('MAXIMUM',i['dt'],0)
    p = g.node('GeometryNodeInputPosition').outputs[0]
    v = g.named('sf_velocity','FLOAT_VECTOR')
    gravity = i['Gravity']
    displacement = g.vector('ADD',g.vector('SCALE',v,h),g.vector('SCALE',gravity,g.math('MULTIPLY',.5,g.math('MULTIPLY',h,h))))
    travel = g.vector('LENGTH',displacement)
    limit = g.math('MAXIMUM',1e-5,i['Max Travel'])
    factor = g.math('MINIMUM',1,g.math('DIVIDE',limit,g.math('MAXIMUM',travel,1e-12)))
    displacement = g.vector('SCALE',displacement,factor)
    velocity = g.vector('SCALE',g.vector('ADD',v,g.vector('SCALE',gravity,h)),factor)
    candidate = g.vector('ADD',p,displacement)
    radius = g.math('POWER',g.math('MULTIPLY',g.named('sf_volume'),3/(4*pi)),1/3)
    clearance = g.math('ADD',radius,1e-5)
    ray = g.node('GeometryNodeRaycast', 'Swept drop contact', data_type='FLOAT')
    for key,value in [('Target Geometry',i['Collision']),('Source Position',p),
                      ('Ray Direction',g.vector('NORMALIZE',displacement)),
                      ('Ray Length',g.math('ADD',g.vector('LENGTH',displacement),radius))]:
        g.connect(value,ray,key)
    hit = g.boolean('AND',ray.outputs['Is Hit'],g.math('GREATER_THAN',travel,1e-12))
    q, valid_q = g.sample(i['Collision'],p,'FLOAT_VECTOR',candidate)
    surface_normal, _ = g.sample(i['Collision'],g.node('GeometryNodeInputNormal').outputs[0],'FLOAT_VECTOR',q)
    surface_normal = g.vector('NORMALIZE',surface_normal)
    collision_point = g.switch('VECTOR',hit,q,ray.outputs['Hit Position'])
    canonical_n = g.switch('VECTOR',hit,surface_normal,g.vector('NORMALIZE',ray.outputs['Hit Normal']))
    front = g.math('GREATER_THAN',g.vector('DOT_PRODUCT',g.vector('SUBTRACT',p,collision_point),canonical_n),-1e-8)
    n = g.switch('VECTOR',front,g.vector('SCALE',canonical_n,-1),canonical_n)
    inward = g.vector('DOT_PRODUCT',velocity,n)
    near = g.boolean('AND',valid_q,g.math('LESS_THAN',g.vector('DISTANCE',candidate,q),clearance))
    collision = g.boolean('OR',hit,near)
    next_position = g.switch('VECTOR',collision,candidate,g.vector('ADD',collision_point,g.vector('SCALE',n,clearance)))
    stopped = g.vector('SUBTRACT',velocity,g.vector('SCALE',n,g.math('MINIMUM',inward,0)))
    next_velocity = g.switch('VECTOR',collision,velocity,stopped)
    capture = g.boolean('AND',hit,g.boolean('AND',front,g.boolean('AND',
        g.math('LESS_THAN',inward,0),g.math('LESS_THAN',g.math('ABSOLUTE',inward),g.math('ADD',i['Capture Speed'],1e-8)))))
    next_position = g.switch('VECTOR',capture,next_position,collision_point)
    next_velocity = g.switch('VECTOR',capture,next_velocity,tangent(g,velocity,n))
    active = g.boolean('AND',g.math('GREATER_THAN',g.named('sf_state','INT'),.5),g.math('GREATER_THAN',h,0))
    changed = g.boolean('AND',active,capture)
    island, _ = g.sample(i['Collision'],g.node('GeometryNodeInputMeshIsland').outputs[0],'INT',collision_point)
    values = {
        'sf_velocity':('FLOAT_VECTOR',g.switch('VECTOR',active,v,next_velocity)),
        'sf_age':('FLOAT',g.math('ADD',g.named('sf_age'),g.switch('FLOAT',active,0,h))),
        'sf_state':('INT',g.switch('INT',changed,g.named('sf_state','INT'),0)),
        'sf_normal':('FLOAT_VECTOR',g.switch('VECTOR',changed,g.named('sf_normal','FLOAT_VECTOR'),n)),
        'sf_anchor':('FLOAT_VECTOR',g.switch('VECTOR',changed,g.named('sf_anchor','FLOAT_VECTOR'),collision_point)),
        'sf_island':('INT',g.switch('INT',changed,g.named('sf_island','INT'),island)),
        'sf_step_limited':('BOOLEAN',g.boolean('OR',g.named('sf_step_limited','BOOLEAN'),g.boolean('AND',active,g.math('GREATER_THAN',travel,limit)))),
        '_sf_transition':('BOOLEAN',changed),
    }
    particles = g.commit(i['Particles'],values,g.switch('VECTOR',active,p,next_position))
    xyz = g.node('ShaderNodeSeparateXYZ'); g.connect(p,xyz,'Vector')
    expired = g.boolean('OR',g.boolean('NOT',g.math('LESS_THAN',g.named('sf_age'),i['Lifetime'])),
                        g.math('LESS_THAN',xyz.outputs['Z'],i['Kill Height']))
    expired = g.boolean('AND',g.math('GREATER_THAN',h,0),expired)
    removed = g.statistic(particles,g.switch('FLOAT',expired,0,g.named('sf_volume')),'Sum')
    delete = g.node('GeometryNodeDeleteGeometry',domain='POINT')
    g.connect(particles,delete,'Geometry'); g.connect(expired,delete,'Selection')
    return g.finish({'Particles':delete.outputs['Geometry'],'Removed Volume':removed})
