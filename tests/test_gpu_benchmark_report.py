import importlib.util
from pathlib import Path


def report_api():
    path=Path('scripts/gpu_report.py')
    assert path.exists(), 'Benchmark report implementation missing'
    spec=importlib.util.spec_from_file_location('gpu_report',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_only_complete_viewport_measurement_can_pass():
    api=report_api()
    common=dict(samples_ms=[10.0]*600,simulated_frames=600,completed_draws=600,
                wall_seconds=6.0,width=1920,height=1080)
    assert not api.summarize('background_evaluation',**common)['realtime_viewport_pass']
    assert api.summarize('viewport_frame',**common)['realtime_viewport_pass']
    assert not api.summarize('viewport_frame',**dict(common,completed_draws=599))['realtime_viewport_pass']
    assert not api.summarize('viewport_frame',**dict(common,wall_seconds=30))['realtime_viewport_pass']


def test_slow_frames_and_wrong_resolution_fail():
    api=report_api()
    values=dict(samples_ms=[50.0]*600,simulated_frames=600,completed_draws=600,wall_seconds=30,width=1920,height=1080)
    assert not api.summarize('viewport_frame',**values)['realtime_viewport_pass']
    assert not api.summarize('viewport_frame',**dict(values,samples_ms=[10]*600,wall_seconds=6,width=960))['realtime_viewport_pass']
