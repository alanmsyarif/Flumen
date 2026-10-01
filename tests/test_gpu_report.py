import pytest
from scripts.gpu_report import summarize


def connected_report(**changes):
    values=dict(samples_ms=[10.]*600,simulated_frames=600,completed_draws=600,
        wall_seconds=6.,width=1920,height=1080,display_mode='CONNECTED',
        engine='BLENDER_EEVEE',fixture='SUZANNE_0P3M',source_triangles=20000,
        dropped_simulation_frames=0,geometry_errors=0)
    values.update(changes)
    return summarize('viewport_frame',**values)


def test_correct_connected_eevee_report_passes():
    assert connected_report()['realtime_connected_pass']


@pytest.mark.parametrize('changes',[
    dict(display_mode='DROPS'),dict(engine='BLENDER_WORKBENCH'),dict(fixture='SPHERE'),
    dict(source_triangles=512),dict(completed_draws=599),dict(simulated_frames=599),
    dict(width=960),dict(dropped_simulation_frames=1),dict(geometry_errors=1),
    dict(wall_seconds=21.),dict(samples_ms=[34.]*600),
])
def test_wrong_fixture_or_incomplete_measurement_fails(changes):
    assert not connected_report(**changes)['realtime_connected_pass']
