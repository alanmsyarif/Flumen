"""Pure-python reference math used by tests and documentation.

The Blender Geometry Nodes implementation mirrors these formulas.
This module intentionally has no bpy dependency.
"""
from __future__ import annotations

from math import sqrt
from typing import Iterable, Tuple

Vec3 = Tuple[float, float, float]


def _vec3(v: Iterable[float]) -> Vec3:
    a = tuple(float(x) for x in v)
    if len(a) != 3:
        raise ValueError("expected 3 components")
    return a  # type: ignore[return-value]


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    x = _vec3(a)
    y = _vec3(b)
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def length(v: Iterable[float]) -> float:
    x = _vec3(v)
    return sqrt(dot(x, x))


def normalize(v: Iterable[float], eps: float = 1e-12) -> Vec3:
    x = _vec3(v)
    l = length(x)
    if l <= eps:
        return (0.0, 0.0, 0.0)
    return (x[0] / l, x[1] / l, x[2] / l)


def tangent_gravity(gravity: Iterable[float], normal: Iterable[float], eps: float = 1e-12) -> Vec3:
    """Project gravity onto the tangent plane of a surface.

    G_t = G - dot(G, N) * N
    """
    g = normalize(gravity, eps)
    n = normalize(normal, eps)
    d = dot(g, n)
    t = (g[0] - d * n[0], g[1] - d * n[1], g[2] - d * n[2])
    return normalize(t, eps)


def normalized_height(value: float, minimum: float, maximum: float, eps: float = 1e-12) -> float:
    if abs(maximum - minimum) <= eps:
        return 0.0
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
