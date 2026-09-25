import math
from flumen.math_core import tangent_gravity, normalized_height


def close(a, b, eps=1e-6):
    return all(abs(x-y) < eps for x, y in zip(a, b))


def test_vertical_wall():
    # Gravity is already tangent to a vertical wall with X normal.
    t = tangent_gravity((0, 0, -1), (1, 0, 0))
    assert close(t, (0, 0, -1))


def test_horizontal_top_has_no_tangent_gravity():
    t = tangent_gravity((0, 0, -1), (0, 0, 1))
    assert close(t, (0, 0, 0))


def test_sloped_surface_points_downhill():
    n = (0, math.sqrt(0.5), math.sqrt(0.5))
    t = tangent_gravity((0, 0, -1), n)
    assert t[2] < 0
    assert t[1] > 0


def test_normalized_height():
    assert normalized_height(5, 0, 10) == 0.5
    assert normalized_height(-1, 0, 10) == 0.0
    assert normalized_height(20, 0, 10) == 1.0
