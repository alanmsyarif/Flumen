import copy
import pytest
from flumen.gpu.preview_report import validate_particle_preview_report

STAGES = ('solver_ms','field_ms','contact_ms','aggregation_ms','resample_ms','deposit_ms',
          'readback_ms','upload_ms','draw_ms')


def valid_report():
    samples = []
    for index in range(600):
        sample = dict(frame=122+index, intervals=1, frame_ms=30., width=1920, height=1080,
                      live_count=1_000_000, displayed_count=1_000_000, contact_unresolved_count=0)
        sample.update({name: 1. for name in STAGES})
        samples.append(sample)
    return dict(measurement_kind='particle_viewport', status='passed', warmup_frames=120,
                capacity=1_000_000, wall_seconds=18., samples=samples, finite_state=True,
                ledger_relative_error=1e-9,
                memory=dict(owned_array_bytes=600*2**20, whole_device_used_mib=4000))


def altered(change):
    report = copy.deepcopy(valid_report()); change(report); return report


def test_valid_synthetic_report_passes():
    assert validate_particle_preview_report(valid_report()) == []


@pytest.mark.parametrize('change', [
    lambda r: [s.update(live_count=None) for s in r['samples']],
    lambda r: r['samples'][300].update(displayed_count=999_999),
    lambda r: r['samples'][5].update(live_count=999_999, displayed_count=999_999),
])
def test_rejects_capacity_only_or_subset(change):
    assert validate_particle_preview_report(altered(change))


@pytest.mark.parametrize('change', [
    lambda r: r.update(warmup_frames=119),
    lambda r: r['samples'].pop(),
    lambda r: r['samples'][10].update(width=1919),
    lambda r: r['samples'][10].update(height=1079),
    lambda r: [s.update(frame_ms=34.) for s in r['samples'][:40]],
    lambda r: r.update(wall_seconds=20.1),
    lambda r: r['samples'][7].update(intervals=2),
    lambda r: r['samples'][7].update(frame=500),
    lambda r: r['samples'][9].update(frame_ms=float('nan')),
])
def test_rejects_short_wrong_size_and_tail_latency(change):
    assert validate_particle_preview_report(altered(change))


@pytest.mark.parametrize('change', [
    lambda r: r['samples'][3].update(contact_unresolved_count=1),
    lambda r: r.update(status='failed'),
    lambda r: r.update(ledger_relative_error=2e-5),
    lambda r: r.update(finite_state=False),
])
def test_rejects_contact_overflow_or_ledger_error(change):
    assert validate_particle_preview_report(altered(change))


@pytest.mark.parametrize('change', [
    lambda r: r['samples'][0].pop('upload_ms'),
    lambda r: r['samples'][0].update(draw_ms=-1.),
    lambda r: r['memory'].pop('owned_array_bytes'),
    lambda r: r['memory'].update(owned_array_bytes=4000*2**20),
])
def test_stages_and_memory_are_distinct(change):
    assert validate_particle_preview_report(altered(change))
