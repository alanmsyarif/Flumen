"""Stable animated interface and particle-state definitions."""
ATTACHED = 0
FREE = 1
SIM_SCHEMA_VERSION = 1
SIM_NODE_GROUP_NAME = 'FL_SurfaceFlowSim'
PARTICLE_ATTRIBUTES = {
    'sf_id': 'INT', 'sf_age': 'FLOAT', 'sf_velocity': 'FLOAT_VECTOR',
    'sf_volume': 'FLOAT', 'sf_state': 'INT', 'sf_normal': 'FLOAT_VECTOR',
    'sf_anchor': 'FLOAT_VECTOR', 'sf_island': 'INT', 'sf_path': 'INT',
    'sf_blocked': 'BOOLEAN', 'sf_step_limited': 'BOOLEAN',
}

# name, socket type, default, minimum, maximum
CONTROLS = (
    ('Gravity', 'NodeSocketVector', (0.0, 0.0, -9.81), None, None),
    ('Source Start', 'NodeSocketFloat', .72, 0.0, 1.0),
    ('Source Softness', 'NodeSocketFloat', .08, 0.0, .5),
    ('Seed Density', 'NodeSocketFloat', 35.0, 0.0, 10000.0),
    ('Seed', 'NodeSocketInt', 0, 0, 1000000),
    ('Particle Budget', 'NodeSocketInt', 512, 1, 2048),
    ('Drop Radius', 'NodeSocketFloat', .001, .00001, .1),
    ('Resistance', 'NodeSocketFloat', 5.0, 0.0, 1000.0),
    ('Adhesion Acceleration', 'NodeSocketFloat', 15.0, 0.0, 1000.0),
    ('Capture Distance', 'NodeSocketFloat', .002, .00001, .1),
    ('Capture Speed', 'NodeSocketFloat', .5, 0.0, 100.0),
    ('Minimum Substeps', 'NodeSocketInt', 8, 1, 64),
    ('Max Travel', 'NodeSocketFloat', .002, .00001, .1),
    ('Normal Turn Limit', 'NodeSocketFloat', 60.0, 1.0, 89.0),
    ('Lifetime', 'NodeSocketFloat', 10.0, .01, 1000.0),
    ('Kill Height', 'NodeSocketFloat', -10.0, -10000.0, 10000.0),
)
