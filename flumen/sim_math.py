"""Independent numerical reference for the procedural particle model (SI units)."""
from math import exp, expm1, isfinite, sqrt


def _scalar(value, name, minimum=None):
    value = float(value)
    if not isfinite(value) or minimum is not None and value < minimum:
        raise ValueError(f'{name} must be finite' + (f' and >= {minimum}' if minimum is not None else ''))
    return value


def _vector(value):
    result = tuple(_scalar(x, 'vector component') for x in value)
    if len(result) != 3:
        raise ValueError('Expected three vector components')
    return result


def free_step(position, velocity, gravity, dt):
    p, v, g = map(_vector, (position, velocity, gravity))
    h = _scalar(dt, 'dt', 0)
    return (tuple(p[i] + h*v[i] + 0.5*h*h*g[i] for i in range(3)),
            tuple(v[i] + h*g[i] for i in range(3)))


def attached_step(anchor, relative_velocity, normal, gravity, drag_rate, dt):
    p, v, n, g = map(_vector, (anchor, relative_velocity, normal, gravity))
    k, h = _scalar(drag_rate, 'drag rate', 0), _scalar(dt, 'dt', 0)
    length = sqrt(sum(x*x for x in n))
    if length < 1e-12:
        raise ValueError('Normal must be nonzero')
    if h == 0:
        return p, v
    n = tuple(x/length for x in n)
    def tangent(vector):
        dot = sum(vector[i]*n[i] for i in range(3))
        return tuple(vector[i]-dot*n[i] for i in range(3))
    u, a = tangent(v), tangent(g)
    x = k*h
    decay = exp(-x)
    if x < 1e-3:
        A = h*(1-x/2+x*x/6-x*x*x/24)
        B = h*h*(0.5-x/6+x*x/24-x*x*x/120)
    else:
        A = -expm1(-x)/k
        B = (h-A)/k
    return (tuple(p[i]+A*u[i]+B*a[i] for i in range(3)),
            tuple(decay*u[i]+A*a[i] for i in range(3)))
