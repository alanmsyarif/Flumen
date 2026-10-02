from dataclasses import replace
import pytest
from flumen.gpu.config import FlowConfig


def test_legacy_config_defaults():
    cfg = FlowConfig()
    assert getattr(cfg, 'solver_backend', None) == 'LEGACY'
    assert cfg.display_mode == 'DROPS'
    cfg.validate()


def test_field_config_bounds():
    cfg = FlowConfig(solver_backend='FIELD', display_mode='POINTS', capacity=100)
    cfg.validate()
    bounds = {'field_spacing': (.00005, .02), 'contact_spacing': (.00005, .05),
              'field_viscosity': (0., .01), 'surface_tension': (0., 1.)}
    for name, (low, high) in bounds.items():
        replace(cfg, **{name: low}).validate()
        replace(cfg, **{name: high}).validate()
        for invalid in (low-1e-7, high+1e-7, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                replace(cfg, **{name: invalid}).validate()
    for value in (True, -1, 101, 1.5):
        with pytest.raises(ValueError):
            replace(cfg, resample_target=value).validate()
    replace(cfg, resample_target=100).validate()
    for backend, mode in [('NO', 'POINTS'), ('FIELD', 'CONNECTED'), ('FIELD', 'DROPS')]:
        with pytest.raises(ValueError):
            replace(cfg, solver_backend=backend, display_mode=mode).validate()


def test_physical_and_preparation_keys_separate_static_and_cosmetic_settings():
    from dataclasses import replace
    from flumen.gpu.config import physical_key, preparation_key
    field = FlowConfig(solver_backend='FIELD', display_mode='POINTS')
    assert physical_key(replace(field, reconstruction_scale=2.)) == physical_key(field)
    assert physical_key(replace(field, interactions_enabled=True)) == physical_key(field)   # LEGACY-only toggle
    assert physical_key(replace(field, gravity=(0., 0., -1.))) != physical_key(field)
    legacy = FlowConfig()
    assert physical_key(replace(legacy, reconstruction_scale=2.)) != physical_key(legacy)
    assert preparation_key('g', replace(field, gravity=(0., 0., -1.))) == preparation_key('g', field)
    assert preparation_key('g', replace(field, field_spacing=.002)) != preparation_key('g', field)
    assert preparation_key('h', field) != preparation_key('g', field)
