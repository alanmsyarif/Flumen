"""Full-count particle preview acceptance; synthetic reports never prove performance."""
import math

TARGET_COUNT = 1_000_000
STAGES = ('solver_ms','field_ms','contact_ms','aggregation_ms','resample_ms','deposit_ms',
          'readback_ms','upload_ms','draw_ms')


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate_particle_preview_report(report: dict) -> list[str]:
    """Return every reason the report fails the 1080p million-particle viewport gate."""
    errors = []
    if report.get('measurement_kind') != 'particle_viewport': errors.append('not a particle viewport measurement')
    if report.get('status') != 'passed': errors.append(f"run status {report.get('status')!r}")
    if not isinstance(report.get('warmup_frames'),int) or report['warmup_frames'] < 120:
        errors.append('fewer than 120 warmup frames')
    samples = report.get('samples') or []
    if len(samples) != 600: errors.append(f'{len(samples)} measured draws, need 600')
    if report.get('finite_state') is not True: errors.append('nonfinite particle state')
    ledger = report.get('ledger_relative_error')
    if not _number(ledger) or ledger > 1e-5: errors.append(f'ledger relative error {ledger}')
    frame_ms = []
    for index,sample in enumerate(samples):
        where = f'sample {index}'
        if (sample.get('width'),sample.get('height')) != (1920,1080): errors.append(f'{where}: viewport not 1920x1080')
        live, shown = sample.get('live_count'), sample.get('displayed_count')
        if not isinstance(live,int) or live < TARGET_COUNT: errors.append(f'{where}: live count {live}')
        if not isinstance(shown,int) or shown < TARGET_COUNT or shown != live:
            errors.append(f'{where}: displayed {shown} of {live} live')
        if sample.get('intervals') != 1: errors.append(f'{where}: advanced {sample.get("intervals")} intervals')
        if index and sample.get('frame') != samples[index-1].get('frame',0)+1: errors.append(f'{where}: frame skipped')
        if sample.get('contact_unresolved_count') != 0: errors.append(f'{where}: unresolved contacts')
        for name in STAGES:
            if not _number(sample.get(name)) or sample[name] < 0: errors.append(f'{where}: missing stage {name}')
        if not _number(sample.get('frame_ms')) or sample['frame_ms'] < 0:
            errors.append(f'{where}: invalid frame time'); continue
        frame_ms.append(sample['frame_ms'])
    if frame_ms and len(frame_ms) == len(samples):
        ordered = sorted(frame_ms)
        p95 = ordered[math.ceil(len(ordered)*.95)-1]
        if p95 > 33.3: errors.append(f'p95 {p95:.2f} ms exceeds 33.3')
        wall = report.get('wall_seconds')
        # Average FPS uses elapsed wall time, so event-loop gaps between draws count.
        if not _number(wall) or wall <= 0 or len(samples)/wall < 30:
            errors.append(f'average FPS below 30 (wall {wall} s)')
    memory = report.get('memory') or {}
    owned = memory.get('owned_array_bytes')
    if not isinstance(owned,int) or owned <= 0: errors.append('owned array bytes missing')
    else:
        device = memory.get('whole_device_used_mib')
        if _number(device) and owned >= device*2**20:
            errors.append('owned array bytes are not distinct from whole-device VRAM')
    return errors
