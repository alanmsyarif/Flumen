from dataclasses import replace
import pytest
from flumen.gpu.config import FlowConfig


def test_legacy_defaults():
    cfg=FlowConfig()
    assert (cfg.display_mode,cfg.interactions_enabled,cfg.time_scale,cfg.initial_coating_count)==('DROPS',False,1.,0)
    cfg.validate()


@pytest.mark.parametrize('values',[
    {'display_mode':'UNKNOWN'}, {'interactions_enabled':1},
    {'time_scale':0}, {'time_scale':float('nan')}, {'initial_coating_count':True},
    {'initial_coating_count':-1}, {'interaction_radius_scale':.9},
    {'cohesion_acceleration':float('inf')}, {'repulsion_acceleration':-1},
    {'surface_damping':-1}, {'merge_distance_scale':1.1},
    {'maximum_merged_radius_scale':.9}, {'reconstruction_scale':0},
    {'wetness_deposit_rate':-1}, {'wetness_drying_rate':float('nan')},
])
def test_connected_control_validation(values):
    with pytest.raises(ValueError):
        replace(FlowConfig(),**values).validate()
