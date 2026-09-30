from dataclasses import replace
import importlib
import pytest


def config_type():
    assert importlib.util.find_spec('flumen.gpu'), 'GPU package missing'
    return importlib.import_module('flumen.gpu.config').FlowConfig


def test_defaults_are_valid_and_zero_rate_is_allowed():
    cls = config_type()
    cfg = cls()
    cfg.validate()
    assert (cfg.particles_per_frame, cfg.capacity, cfg.lifetime) == (64, 8192, 4.0)
    replace(cfg, particles_per_frame=0).validate()


@pytest.mark.parametrize('kwargs', [
    {'radius': 0}, {'radius': float('nan')}, {'gravity': (0, 0, 0)},
    {'gravity': (0, 0, float('inf'))}, {'particles_per_frame': -1},
    {'particles_per_frame': 1.2}, {'capacity': 0}, {'emission_end': 0},
    {'minimum_substeps': 65}, {'lifetime': -1}, {'mode': 'invalid'},
])
def test_rejects_invalid_settings(kwargs):
    cls = config_type()
    with pytest.raises(ValueError):
        replace(cls(), **kwargs).validate()


def test_fps_base_is_used_and_invalid_timing_rejected():
    cls = config_type()
    module = importlib.import_module('flumen.gpu.config')
    assert module.frame_dt(30, 1.001) == pytest.approx(1.001 / 30)
    for fps, base in [(0,1), (30,0), (float('nan'),1), (30,float('inf'))]:
        with pytest.raises(ValueError):
            module.frame_dt(fps, base)
