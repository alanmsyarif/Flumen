"""Performance acceptance cannot be inferred from solver-only timing."""
import math
import statistics


def summarize(measurement_kind, samples_ms, simulated_frames, completed_draws,
              wall_seconds, width, height):
    if not samples_ms or any(not math.isfinite(x) or x<0 for x in samples_ms):
        raise ValueError('Timing samples must be finite and nonempty')
    ordered=sorted(samples_ms)
    p95=ordered[math.ceil(len(ordered)*.95)-1]
    fps=len(samples_ms)/wall_seconds if wall_seconds>0 else 0
    passed=(measurement_kind=='viewport_frame' and len(samples_ms)==600 and
            simulated_frames==600 and completed_draws==600 and width==1920 and height==1080
            and p95<=33.3 and fps>=30)
    return dict(measurement_kind=measurement_kind,median_ms=statistics.median(ordered),
                p95_ms=p95,max_ms=max(ordered),average_fps=fps,sample_count=len(ordered),
                simulated_frames=simulated_frames,completed_draws=completed_draws,
                width=width,height=height,wall_seconds=wall_seconds,realtime_viewport_pass=passed)
