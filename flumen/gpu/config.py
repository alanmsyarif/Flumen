"""Immutable settings shared by CUDA and Blender adapters."""
from dataclasses import dataclass, fields
from math import isfinite, sqrt

# Bump when chart/contact construction changes, so retained preparation is rebuilt.
PREPARATION_VERSION = 1
# FIELD meshes cached particles offline; mesh quality never changes particle state.
# FIELD ignores these: offline-only reconstruction, and the LEGACY pairwise interaction toggle.
_FIELD_OFFLINE_ONLY = frozenset({'reconstruction_scale', 'interactions_enabled'})


def physical_key(config) -> tuple:
    """Settings whose change requires replaying particle state."""
    skip = _FIELD_OFFLINE_ONLY if config.solver_backend == 'FIELD' else frozenset()
    return tuple((f.name, getattr(config, f.name)) for f in fields(config) if f.name not in skip)


def preparation_key(geometry_fingerprint: str, config) -> str:
    """Identity of static chart/contact data: evaluated world geometry plus spacing."""
    return f'{PREPARATION_VERSION}:{geometry_fingerprint}:{config.field_spacing!r}:{config.contact_spacing!r}'


def frame_dt(fps: float, fps_base: float = 1.0) -> float:
    if any(not isfinite(v) or v <= 0 for v in (fps, fps_base)):
        raise ValueError('FPS and FPS base must be positive and finite')
    return fps_base / fps


@dataclass(frozen=True)
class FlowConfig:
    mode: str = 'CONTINUOUS'
    particles_per_frame: int = 64
    burst_count: int = 512
    emission_start: int = 1
    emission_end: int = 250
    capacity: int = 8192
    seed: int = 0
    radius: float = .001
    lifetime: float = 4.0
    kill_height: float = -10.0
    gravity: tuple = (0.0, 0.0, -9.81)
    source_start: float = .72
    source_softness: float = .08
    resistance: float = 5.0
    adhesion: float = 15.0
    capture_distance: float = .002
    capture_speed: float = .5
    minimum_substeps: int = 8
    max_travel: float = .002
    normal_turn_limit: float = 60.0
    display_mode: str = 'DROPS'
    interactions_enabled: bool = False
    time_scale: float = 1.0
    initial_coating_count: int = 0
    interaction_radius_scale: float = 4.0
    cohesion_acceleration: float = 2.0
    repulsion_acceleration: float = 20.0
    surface_damping: float = 10.0
    merge_distance_scale: float = .25
    maximum_merged_radius_scale: float = 4.0
    reconstruction_scale: float = 1.0
    wetness_deposit_rate: float = 5.0
    wetness_drying_rate: float = .1
    solver_backend: str = 'LEGACY'
    field_spacing: float = .001
    contact_spacing: float = .002
    field_viscosity: float = 1.e-6
    surface_tension: float = .072
    contact_hysteresis: float = 0.   # cos(receding)-cos(advancing): dry surface pins thin film fronts
    resample_target: int = 0

    def validate(self) -> None:
        if self.mode not in ('BURST', 'CONTINUOUS'):
            raise ValueError('Emission mode must be BURST or CONTINUOUS')
        if self.display_mode not in ('DROPS','CONNECTED','POINTS'):
            raise ValueError('Unknown particle display mode')
        if self.solver_backend not in ('LEGACY','FIELD'):
            raise ValueError('Unknown solver backend')
        if self.solver_backend == 'FIELD' and self.display_mode != 'POINTS':
            raise ValueError('FIELD requires the POINTS preview; mesh cached frames offline')
        if not isinstance(self.interactions_enabled,bool):
            raise ValueError('interactions_enabled must be a Boolean')
        for name, low, high in (
            ('particles_per_frame', 0, 2**31-1), ('burst_count', 0, 2**31-1),
            ('capacity', 1, 1_000_000), ('seed', 0, 2**31-1),
            ('emission_start', -1_000_000, 1_000_000),
            ('emission_end', -1_000_000, 1_000_000), ('minimum_substeps', 1, 64),
            ('initial_coating_count', 0, 2**31-1),
            ('resample_target', 0, self.capacity),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        if self.emission_end < self.emission_start:
            raise ValueError('Emission end must be at or after start')
        for name, low, high in (
            ('radius', .00001, .1), ('lifetime', .01, 1000.),
            ('kill_height', -10000., 10000.), ('source_start', 0., 1.),
            ('source_softness', 0., .5), ('resistance', 0., 1000.),
            ('adhesion', 0., 1000.), ('capture_distance', .00001, .1),
            ('capture_speed', 0., 100.), ('max_travel', .00001, .1),
            ('normal_turn_limit', 1., 89.),
            ('time_scale', .01, 2.), ('interaction_radius_scale',1.,16.),
            ('cohesion_acceleration',0.,100.), ('repulsion_acceleration',0.,100.),
            ('surface_damping',0.,1000.), ('merge_distance_scale',0.,1.),
            ('maximum_merged_radius_scale',1.,8.), ('reconstruction_scale',.5,4.),
            ('wetness_deposit_rate',0.,1000.), ('wetness_drying_rate',0.,1000.),
            ('field_spacing',.00005,.02), ('contact_spacing',.00005,.05),
            ('field_viscosity',0.,.01), ('surface_tension',0.,1.), ('contact_hysteresis',0.,2.),
        ):
            value = getattr(self, name)
            if not isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be finite in [{low}, {high}]')
        if len(self.gravity) != 3 or any(not isfinite(v) for v in self.gravity):
            raise ValueError('Gravity must have three finite components')
        if not 1e-8 < sqrt(sum(v*v for v in self.gravity)) <= 10000:
            raise ValueError('Gravity magnitude must be nonzero and <= 10000')
