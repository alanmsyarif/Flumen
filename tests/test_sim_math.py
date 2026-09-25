import importlib.util
import math
import pytest


def api():
    assert importlib.util.find_spec('flumen.sim_math'), 'Animated numerical reference missing'
    from flumen import sim_math
    return sim_math


def test_free_fall_exact_step():
    p, v = api().free_step((0,0,1), (0,0,0), (0,0,-9.81), 0.1)
    assert p == pytest.approx((0,0,0.95095))
    assert v == pytest.approx((0,0,-0.981))


@pytest.mark.parametrize('fps', [24, 48])
@pytest.mark.parametrize('drag', [0, 1e-6, 1e-3, 5])
def test_plane_drag_matches_analytic_solution(fps, drag):
    p, v = (0,0,0), (0,0,0)
    for _ in range(fps):
        p,v = api().attached_step(p,v,(0,1,0),(0,0,-1),drag,1/fps)
    distance = 0.5 if drag == 0 else (drag + math.expm1(-drag)) / (drag*drag)
    assert p == pytest.approx((0,0,-distance), abs=1e-5)
    assert v[1] == 0


def test_shallow_slope_preserves_acceleration_magnitude():
    p,v=api().attached_step((0,0,0),(0,0,0),(0,0.1,math.sqrt(.99)),(0,0,-9.81),0,0.1)
    assert math.sqrt(sum(x*x for x in v)) == pytest.approx(0.0981)


def test_zero_dt_does_not_change_state():
    args=((1,2,3),(4,5,6),(0,0,-9.81),0)
    assert api().free_step(*args) == (args[0], args[1])


@pytest.mark.parametrize('dt', [-1, math.nan, math.inf])
def test_invalid_time_rejected(dt):
    with pytest.raises(ValueError):
        api().free_step((0,0,0),(0,0,0),(0,0,-9.81),dt)


def test_invalid_normal_and_drag_rejected():
    for normal,drag in [((0,0,0),1),((0,0,1),-1),((0,0,math.nan),1)]:
        with pytest.raises(ValueError):
            api().attached_step((0,0,0),(0,0,0),normal,(0,0,-9.81),drag,.1)
