"""Constrained particle integration and an explicit artistic adhesion threshold."""
from math import pi
import bpy
from .sim_nodes import Graph, tangent, owned_group
from .sim_schema import CONTROLS


def build_attached_group():
    old = owned_group('SF_AttachedStep')
    if old:
        return old
    controls = {'Gravity','Resistance','Adhesion Acceleration','Capture Distance','Max Travel','Normal Turn Limit'}
    g = Graph('SF_AttachedStep', [('Particles','NodeSocketGeometry'),('Collision','NodeSocketGeometry'),
                ('dt','NodeSocketFloat',0.0,0.0,1.0), *(c for c in CONTROLS if c[0] in controls)],
              [('Particles','NodeSocketGeometry')])
    i = g.input.outputs
    h = g.math('MAXIMUM',i['dt'],0)
    p = g.node('GeometryNodeInputPosition').outputs[0]
    old_velocity = g.named('sf_velocity','FLOAT_VECTOR')
    old_normal = g.named('sf_normal','FLOAT_VECTOR')
    island = g.named('sf_island','INT')
    normal, valid_old = g.sample(i['Collision'],g.node('GeometryNodeInputNormal').outputs[0],'FLOAT_VECTOR',p,island)
    n = g.vector('NORMALIZE',normal)
    u = tangent(g,old_velocity,n)
    a = tangent(g,i['Gravity'],n)
    resistance = g.node('GeometryNodeInputNamedAttribute', 'Surface Resistance', data_type='FLOAT')
    g.connect('sf_resistance',resistance,'Name')
    k_field = g.switch('FLOAT',resistance.outputs['Exists'],i['Resistance'],resistance.outputs['Attribute'])
    k_sample, _ = g.sample(i['Collision'],k_field,'FLOAT',p,island)
    k = g.math('MAXIMUM',k_sample,0)
    x = g.math('MULTIPLY',k,h)
    x2 = g.math('MULTIPLY',x,x)
    decay = g.math('EXPONENT',g.math('MULTIPLY',-1,x))
    denominator = g.math('MAXIMUM',k,1e-8)
    A_exact = g.math('DIVIDE',g.math('SUBTRACT',1,decay),denominator)
    B_exact = g.math('DIVIDE',g.math('SUBTRACT',h,A_exact),denominator)
    A_small = g.math('MULTIPLY',h,g.math('ADD',g.math('SUBTRACT',1,g.math('MULTIPLY',.5,x)),g.math('DIVIDE',x2,6)))
    B_small = g.math('MULTIPLY',g.math('MULTIPLY',h,h),
                     g.math('ADD',g.math('SUBTRACT',.5,g.math('DIVIDE',x,6)),g.math('DIVIDE',x2,24)))
    small = g.math('LESS_THAN',x,.001)
    A = g.switch('FLOAT',small,A_exact,A_small)
    B = g.switch('FLOAT',small,B_exact,B_small)
    velocity = g.vector('ADD',g.vector('SCALE',u,decay),g.vector('SCALE',a,A))
    displacement = g.vector('ADD',g.vector('SCALE',u,A),g.vector('SCALE',a,B))
    travel = g.vector('LENGTH',displacement)
    limit = g.math('MAXIMUM',i['Max Travel'],1e-5)
    factor = g.math('MINIMUM',1,g.math('DIVIDE',limit,g.math('MAXIMUM',travel,1e-12)))
    displacement = g.vector('SCALE',displacement,factor)
    velocity = g.vector('SCALE',velocity,factor)
    candidate = g.vector('ADD',p,displacement)
    q, valid_q = g.sample(i['Collision'],p,'FLOAT_VECTOR',candidate,island)
    next_normal, _ = g.sample(i['Collision'],g.node('GeometryNodeInputNormal').outputs[0],'FLOAT_VECTOR',q,island)
    next_normal = g.vector('NORMALIZE',next_normal)
    residual = g.vector('LENGTH',tangent(g,g.vector('SUBTRACT',candidate,q),n))
    lost_tangent = g.boolean('AND',g.math('GREATER_THAN',travel,1e-5),
                             g.math('GREATER_THAN',residual,g.math('MULTIPLY',.5,g.vector('LENGTH',displacement))))
    lost_distance = g.math('GREATER_THAN',g.vector('DISTANCE',candidate,q),i['Capture Distance'])
    angle_cos = g.math('COSINE',g.math('MULTIPLY',i['Normal Turn Limit'],pi/180))
    turn_ok = g.math('GREATER_THAN',g.vector('DOT_PRODUCT',n,next_normal),angle_cos)
    valid = g.boolean('AND',valid_old,valid_q)
    valid = g.boolean('AND',valid,g.math('GREATER_THAN',g.vector('LENGTH',n),.5))
    outward = g.math('GREATER_THAN',g.vector('DOT_PRODUCT',i['Gravity'],n),i['Adhesion Acceleration'])
    detach = g.boolean('AND',valid,g.boolean('OR',outward,g.boolean('OR',lost_tangent,lost_distance)))
    accepted = g.boolean('AND',valid,g.boolean('AND',turn_ok,g.boolean('NOT',g.boolean('OR',lost_tangent,lost_distance))))
    active = g.boolean('AND',g.math('LESS_THAN',g.named('sf_state','INT'),.5),g.math('GREATER_THAN',h,0))
    radius = g.math('POWER',g.math('MULTIPLY',g.named('sf_volume'),3/(4*pi)),1/3)
    released = g.vector('ADD',candidate,g.vector('SCALE',n,g.math('ADD',radius,1e-5)))
    next_position = g.switch('VECTOR',accepted,p,q)
    next_position = g.switch('VECTOR',detach,next_position,released)
    next_velocity = g.switch('VECTOR',accepted,(0,0,0),tangent(g,velocity,next_normal))
    next_velocity = g.switch('VECTOR',detach,next_velocity,velocity)
    changed = g.boolean('AND',active,detach)
    values = {
        'sf_velocity': ('FLOAT_VECTOR',g.switch('VECTOR',active,old_velocity,next_velocity)),
        'sf_age': ('FLOAT',g.math('ADD',g.named('sf_age'),g.switch('FLOAT',active,0,h))),
        'sf_normal': ('FLOAT_VECTOR',g.switch('VECTOR',g.boolean('AND',active,accepted),old_normal,next_normal)),
        'sf_anchor': ('FLOAT_VECTOR',g.switch('VECTOR',g.boolean('AND',active,accepted),g.named('sf_anchor','FLOAT_VECTOR'),q)),
        'sf_state': ('INT',g.switch('INT',changed,g.named('sf_state','INT'),1)),
        'sf_blocked': ('BOOLEAN',g.switch('BOOLEAN',active,g.named('sf_blocked','BOOLEAN'),g.boolean('NOT',accepted))),
        'sf_step_limited': ('BOOLEAN',g.boolean('OR',g.named('sf_step_limited','BOOLEAN'),g.boolean('AND',active,g.math('GREATER_THAN',travel,limit)))),
        '_sf_transition': ('BOOLEAN',changed),
    }
    particles = g.commit(i['Particles'],values,g.switch('VECTOR',active,p,next_position))
    return g.finish({'Particles':particles})
