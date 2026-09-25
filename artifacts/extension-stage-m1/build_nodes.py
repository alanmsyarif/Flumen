"""Geometry Nodes builder for the Flumen MVP.

Target: Blender 5.2 LTS.

MVP implemented here:
- automatic source mask at the top of the mesh relative to gravity
- face seed distribution
- surface tangent gravity
- Repeat Zone path integration
- closest-surface re-projection with Geometry Proximity
- stable sf_id / sf_age / sf_normal attributes
- Points to Curves reconstruction
- preview tube output

This is deliberately a procedural drainage approximation, not a fluid solver.
"""
from __future__ import annotations

import bpy

from .constants import (
    ATTR_AGE,
    ATTR_ID,
    ATTR_NORMAL,
    MATERIAL_NAME,
    NODE_GROUP_NAME,
)
from .node_utils import (
    add_interface_socket,
    clear_nodes,
    link,
    new_node,
    set_input,
    socket_by_name,
)


def ensure_flow_material():
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(MATERIAL_NAME)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            base = bsdf.inputs.get("Base Color")
            if base:
                base.default_value = (0.12, 0.32, 0.55, 1.0)
            rough = bsdf.inputs.get("Roughness")
            if rough:
                rough.default_value = 0.18
            metallic = bsdf.inputs.get("Metallic")
            if metallic:
                metallic.default_value = 0.0
    return mat


def _build_interface(tree, material):
    # Output first so Geometry is the modifier result.
    add_interface_socket(tree, "Geometry", "OUTPUT", "NodeSocketGeometry")
    add_interface_socket(tree, "Geometry", "INPUT", "NodeSocketGeometry")
    add_interface_socket(tree, "Seeds", "OUTPUT", "NodeSocketGeometry")
    add_interface_socket(tree, "Trails", "OUTPUT", "NodeSocketGeometry")

    add_interface_socket(
        tree, "Gravity", "INPUT", "NodeSocketVector",
        default=(0.0, 0.0, -1.0),
        description="World-like gravity direction in object space. Magnitude is ignored.",
    )
    add_interface_socket(
        tree, "Source Start", "INPUT", "NodeSocketFloat",
        default=0.72, min_value=0.0, max_value=1.0,
        description="Normalized height where automatic coating/source begins.",
    )
    add_interface_socket(
        tree, "Source Softness", "INPUT", "NodeSocketFloat",
        default=0.08, min_value=0.0, max_value=0.5,
    )
    add_interface_socket(
        tree, "Seed Density", "INPUT", "NodeSocketFloat",
        default=35.0, min_value=0.0, max_value=10000.0,
    )
    add_interface_socket(
        tree, "Steps", "INPUT", "NodeSocketInt",
        default=48, min_value=1, max_value=512,
    )
    add_interface_socket(
        tree, "Step Length", "INPUT", "NodeSocketFloat",
        default=0.025, min_value=0.00001, max_value=10.0,
    )
    add_interface_socket(
        tree, "Randomness", "INPUT", "NodeSocketFloat",
        default=0.10, min_value=0.0, max_value=1.0,
    )
    add_interface_socket(
        tree, "Seed", "INPUT", "NodeSocketInt",
        default=0, min_value=0, max_value=1000000,
    )
    add_interface_socket(
        tree, "Flow Radius", "INPUT", "NodeSocketFloat",
        default=0.006, min_value=0.00001, max_value=1.0,
    )
    add_interface_socket(
        tree, "Surface Offset", "INPUT", "NodeSocketFloat",
        default=0.002, min_value=0.0, max_value=1.0,
    )
    mat_sock = add_interface_socket(tree, "Flow Material", "INPUT", "NodeSocketMaterial")
    if hasattr(mat_sock, "default_value"):
        mat_sock.default_value = material


def _store_named(tree, geometry_node, value_node, value_names, attr_name, data_type, x, y):
    store = new_node(tree, "GeometryNodeStoreNamedAttribute", f"Store {attr_name}", x, y)
    store.data_type = data_type
    store.domain = 'POINT'
    set_input(store, ("Name",), attr_name)
    link(tree, geometry_node, ("Geometry", "Points"), store, ("Geometry",), from_fallback=0, to_fallback=0)
    link(tree, value_node, value_names, store, ("Value",), from_fallback=0)
    return store


def _populate_static(tree, material):
    _build_interface(tree, material)

    nodes = tree.nodes
    links = tree.links

    # Group IO -----------------------------------------------------------------
    group_in = new_node(tree, "NodeGroupInput", "GROUP INPUT", -1800, 0)
    group_out = new_node(tree, "NodeGroupOutput", "GROUP OUTPUT", 1900, 0)
    group_out.is_active_output = True

    # Normalize gravity and derive 'up' (-gravity) -----------------------------
    g_norm = new_node(tree, "ShaderNodeVectorMath", "Normalize Gravity", -1580, 500)
    g_norm.operation = 'NORMALIZE'
    link(tree, group_in, ("Gravity",), g_norm, ("Vector",), to_index=0)

    g_neg = new_node(tree, "ShaderNodeVectorMath", "Up Direction", -1380, 500)
    g_neg.operation = 'SCALE'
    link(tree, g_norm, ("Vector",), g_neg, ("Vector",), from_fallback=0, to_index=0)
    set_input(g_neg, ("Scale",), -1.0, fallback=3)

    # Automatic source region relative to gravity ------------------------------
    pos_src = new_node(tree, "GeometryNodeInputPosition", "Surface Position", -1580, 220)
    height_dot = new_node(tree, "ShaderNodeVectorMath", "Height Along Up", -1370, 220)
    height_dot.operation = 'DOT_PRODUCT'
    link(tree, pos_src, ("Position",), height_dot, ("Vector",), to_index=0)
    link(tree, g_neg, ("Vector",), height_dot, ("Vector",), from_fallback=0, to_index=1)

    stats = new_node(tree, "GeometryNodeAttributeStatistic", "Height Range", -1140, 240)
    stats.data_type = 'FLOAT'
    stats.domain = 'POINT'
    link(tree, group_in, ("Geometry",), stats, ("Geometry",), to_fallback=0)
    link(tree, height_dot, ("Value",), stats, ("Attribute", "Value"), from_fallback=1)

    norm_height = new_node(tree, "ShaderNodeMapRange", "Normalized Height", -900, 250)
    norm_height.data_type = 'FLOAT'
    norm_height.interpolation_type = 'LINEAR'
    norm_height.clamp = True
    link(tree, height_dot, ("Value",), norm_height, ("Value",), from_fallback=1)
    link(tree, stats, ("Min",), norm_height, ("From Min",))
    link(tree, stats, ("Max",), norm_height, ("From Max",))
    set_input(norm_height, ("To Min",), 0.0)
    set_input(norm_height, ("To Max",), 1.0)

    lower = new_node(tree, "ShaderNodeMath", "Source Lower", -900, 40)
    lower.operation = 'SUBTRACT'
    link(tree, group_in, ("Source Start",), lower, ("Value",), to_index=0)
    link(tree, group_in, ("Source Softness",), lower, ("Value",), to_index=1)

    upper = new_node(tree, "ShaderNodeMath", "Source Upper", -900, -70)
    upper.operation = 'ADD'
    link(tree, group_in, ("Source Start",), upper, ("Value",), to_index=0)
    link(tree, group_in, ("Source Softness",), upper, ("Value",), to_index=1)

    source_mask = new_node(tree, "ShaderNodeMapRange", "Auto Source Mask", -650, 180)
    source_mask.data_type = 'FLOAT'
    source_mask.interpolation_type = 'SMOOTHERSTEP'
    source_mask.clamp = True
    link(tree, norm_height, ("Result",), source_mask, ("Value",))
    link(tree, lower, ("Value",), source_mask, ("From Min",))
    link(tree, upper, ("Value",), source_mask, ("From Max",))
    set_input(source_mask, ("To Min",), 0.0)
    set_input(source_mask, ("To Max",), 1.0)

    hard_mask = new_node(tree, 'ShaderNodeMath', 'Hard Source Threshold', -650, -100)
    hard_mask.operation = 'GREATER_THAN'
    link(tree, norm_height, ('Result',), hard_mask, ('Value',), to_index=0)
    link(tree, group_in, ('Source Start',), hard_mask, ('Value',), to_index=1)
    hard_enabled = new_node(tree, 'ShaderNodeMath', 'Zero Softness', -650, -240)
    hard_enabled.operation = 'LESS_THAN'
    link(tree, group_in, ('Source Softness',), hard_enabled, ('Value',), to_index=0)
    set_input(hard_enabled, ('Value',), 1e-8, index=1)
    choose_mask = new_node(tree, 'GeometryNodeSwitch', 'Choose Source Mask', -430, -100)
    choose_mask.input_type = 'FLOAT'
    link(tree, hard_enabled, ('Value',), choose_mask, ('Switch',))
    link(tree, source_mask, ('Result',), choose_mask, ('False',))
    link(tree, hard_mask, ('Value',), choose_mask, ('True',))

    density = new_node(tree, "ShaderNodeMath", "Source Density", -430, 150)
    density.operation = 'MULTIPLY'
    link(tree, choose_mask, ("Output",), density, ("Value",), to_index=0)
    link(tree, group_in, ("Seed Density",), density, ("Value",), to_index=1)

    distribute = new_node(tree, "GeometryNodeDistributePointsOnFaces", "Seed Flow", -190, 80)
    distribute.distribute_method = 'RANDOM'
    link(tree, group_in, ("Geometry",), distribute, ("Mesh", "Geometry"), to_fallback=0)
    link(tree, density, ("Value",), distribute, ("Density",))
    height_span = new_node(tree, 'ShaderNodeMath', 'Height Span', -1140, -100)
    height_span.operation = 'SUBTRACT'
    link(tree, stats, ('Max',), height_span, ('Value',), to_index=0)
    link(tree, stats, ('Min',), height_span, ('Value',), to_index=1)
    valid_height = new_node(tree, 'ShaderNodeMath', 'Nonflat Source', -900, -300)
    valid_height.operation = 'GREATER_THAN'
    link(tree, height_span, ('Value',), valid_height, ('Value',), to_index=0)
    set_input(valid_height, ('Value',), 1e-8, index=1)
    link(tree, valid_height, ('Value',), distribute, ('Selection',))
    # Random seed is named Seed in current Blender, fallback is defensive.
    try:
        link(tree, group_in, ("Seed",), distribute, ("Seed",))
    except Exception:
        pass

    # Initial stable attributes -------------------------------------------------
    id_input = new_node(tree, "GeometryNodeInputID", "Seed ID", 30, -100)
    store_id = _store_named(tree, distribute, id_input, ("ID",), ATTR_ID, 'INT', 240, 70)

    value_zero = new_node(tree, "ShaderNodeValue", "Age 0", 40, -260)
    value_zero.outputs[0].default_value = 0.0
    store_age0 = _store_named(tree, store_id, value_zero, ("Value",), ATTR_AGE, 'FLOAT', 470, 70)

    # Initial normal comes directly from Distribute Points on Faces and is
    # evaluated on the generated points, which keeps it attached to the seeds.
    store_n0 = new_node(tree, "GeometryNodeStoreNamedAttribute", f"Store {ATTR_NORMAL}", 700, 70)
    store_n0.data_type = 'FLOAT_VECTOR'
    store_n0.domain = 'POINT'
    set_input(store_n0, ("Name",), ATTR_NORMAL)
    link(tree, store_age0, ("Geometry",), store_n0, ("Geometry",), to_fallback=0)
    link(tree, distribute, ("Normal",), store_n0, ("Value",))

    # Repeat Zone ---------------------------------------------------------------
    repeat_in = new_node(tree, "GeometryNodeRepeatInput", "FLOW REPEAT INPUT", 930, 80)
    repeat_out = new_node(tree, "GeometryNodeRepeatOutput", "FLOW REPEAT OUTPUT", 2700, 80)
    repeat_in.pair_with_output(repeat_out)
    repeat_out.repeat_items.clear()
    repeat_out.repeat_items.new('GEOMETRY', 'Tips')
    repeat_out.repeat_items.new('GEOMETRY', 'Trails')

    # after changing repeat items, sockets are generated dynamically
    link(tree, group_in, ("Steps",), repeat_in, ("Iterations",), to_fallback=0)
    link(tree, store_n0, ("Geometry",), repeat_in, ("Tips",), to_fallback=1)
    link(tree, store_n0, ("Geometry",), repeat_in, ("Trails",), to_fallback=2)

    # Sample nearest surface normal at every active tip -------------------------
    tip_pos = new_node(tree, "GeometryNodeInputPosition", "Tip Position", 1160, 430)
    target_normal = new_node(tree, "GeometryNodeInputNormal", "Target Normal", 1130, 640)
    sample_normal = new_node(tree, "GeometryNodeSampleNearestSurface", "Nearest Surface Normal", 1370, 450)
    sample_normal.data_type = 'FLOAT_VECTOR'
    link(tree, group_in, ("Geometry",), sample_normal, ("Mesh", "Geometry"), to_fallback=0)
    link(tree, target_normal, ("Normal",), sample_normal, ("Value",))
    link(tree, tip_pos, ("Position",), sample_normal, ("Sample Position",))
    unit_normal = new_node(tree, 'ShaderNodeVectorMath', 'Unit Surface Normal', 1480, 800)
    unit_normal.operation = 'NORMALIZE'
    link(tree, sample_normal, ('Value',), unit_normal, ('Vector',), to_index=0)

    # Tangent gravity: G - dot(G,N)N -------------------------------------------
    dot_gn = new_node(tree, "ShaderNodeVectorMath", "G dot N", 1590, 610)
    dot_gn.operation = 'DOT_PRODUCT'
    link(tree, g_norm, ("Vector",), dot_gn, ("Vector",), from_fallback=0, to_index=0)
    link(tree, unit_normal, ("Vector",), dot_gn, ("Vector",), to_index=1)

    n_scaled = new_node(tree, "ShaderNodeVectorMath", "Normal Component", 1800, 600)
    n_scaled.operation = 'SCALE'
    link(tree, unit_normal, ("Vector",), n_scaled, ("Vector",), to_index=0)
    link(tree, dot_gn, ("Value",), n_scaled, ("Scale",), from_fallback=1, to_fallback=3)

    tangent = new_node(tree, "ShaderNodeVectorMath", "Tangent Gravity", 2010, 580)
    tangent.operation = 'SUBTRACT'
    link(tree, g_norm, ("Vector",), tangent, ("Vector",), from_fallback=0, to_index=0)
    link(tree, n_scaled, ("Vector",), tangent, ("Vector",), from_fallback=0, to_index=1)

    tangent_norm = new_node(tree, "ShaderNodeVectorMath", "Normalize Tangent", 2210, 580)
    tangent_norm.operation = 'NORMALIZE'
    link(tree, tangent, ("Vector",), tangent_norm, ("Vector",), to_index=0)

    # Deterministic tangent noise ----------------------------------------------
    rand = new_node(tree, "FunctionNodeRandomValue", "Stable Direction Noise", 1590, 310)
    rand.data_type = 'FLOAT_VECTOR'
    set_input(rand, ("Min",), (-1.0, -1.0, -1.0))
    set_input(rand, ("Max",), (1.0, 1.0, 1.0))
    id_inside = new_node(tree, "GeometryNodeInputID", "Tip ID", 1370, 250)
    link(tree, id_inside, ("ID",), rand, ("ID",))

    seed_add = new_node(tree, "ShaderNodeMath", "Seed + Iteration", 1370, 120)
    seed_add.operation = 'ADD'
    link(tree, group_in, ("Seed",), seed_add, ("Value",), to_index=0)
    link(tree, repeat_in, ("Iteration",), seed_add, ("Value",), to_index=1)
    try:
        link(tree, seed_add, ("Value",), rand, ("Seed",))
    except Exception:
        pass

    noise_dot = new_node(tree, "ShaderNodeVectorMath", "Noise dot N", 1800, 330)
    noise_dot.operation = 'DOT_PRODUCT'
    link(tree, rand, ("Value",), noise_dot, ("Vector",), to_index=0)
    link(tree, unit_normal, ("Vector",), noise_dot, ("Vector",), to_index=1)

    noise_n = new_node(tree, "ShaderNodeVectorMath", "Noise Normal Component", 2000, 330)
    noise_n.operation = 'SCALE'
    link(tree, unit_normal, ("Vector",), noise_n, ("Vector",), to_index=0)
    link(tree, noise_dot, ("Value",), noise_n, ("Scale",), from_fallback=1, to_fallback=3)

    noise_tan = new_node(tree, "ShaderNodeVectorMath", "Tangent Noise", 2200, 330)
    noise_tan.operation = 'SUBTRACT'
    link(tree, rand, ("Value",), noise_tan, ("Vector",), to_index=0)
    link(tree, noise_n, ("Vector",), noise_tan, ("Vector",), from_fallback=0, to_index=1)

    noise_scale = new_node(tree, "ShaderNodeVectorMath", "Noise Strength", 2400, 330)
    noise_scale.operation = 'SCALE'
    link(tree, noise_tan, ("Vector",), noise_scale, ("Vector",), to_index=0)
    link(tree, group_in, ("Randomness",), noise_scale, ("Scale",), to_fallback=3)

    direction_add = new_node(tree, "ShaderNodeVectorMath", "Flow Direction", 2430, 560)
    direction_add.operation = 'ADD'
    link(tree, tangent_norm, ("Vector",), direction_add, ("Vector",), to_index=0)
    link(tree, noise_scale, ("Vector",), direction_add, ("Vector",), to_index=1)

    direction_norm = new_node(tree, "ShaderNodeVectorMath", "Normalize Flow", 2640, 560)
    direction_norm.operation = 'NORMALIZE'
    link(tree, direction_add, ("Vector",), direction_norm, ("Vector",), to_index=0)

    step_vec = new_node(tree, "ShaderNodeVectorMath", "Step Vector", 2850, 540)
    step_vec.operation = 'SCALE'
    link(tree, direction_norm, ("Vector",), step_vec, ("Vector",), to_index=0)
    link(tree, group_in, ("Step Length",), step_vec, ("Scale",), to_fallback=3)

    candidate = new_node(tree, "GeometryNodeSetPosition", "Advance Tips", 3060, 100)
    link(tree, repeat_in, ("Tips",), candidate, ("Geometry",), from_fallback=1, to_fallback=0)
    link(tree, step_vec, ("Vector",), candidate, ("Offset",))

    # Closest-surface projection; this is more robust than raycast-only
    # attachment around noses, rims and concavities.
    candidate_pos = new_node(tree, "GeometryNodeInputPosition", "Candidate Position", 3060, 420)
    proximity = new_node(tree, "GeometryNodeProximity", "Project to Surface", 3270, 360)
    proximity.target_element = 'FACES'
    link(tree, group_in, ("Geometry",), proximity, ("Geometry",), to_fallback=0)
    link(tree, candidate_pos, ("Position",), proximity, ("Sample Position",))

    projected = new_node(tree, "GeometryNodeSetPosition", "Snap Tips", 3490, 100)
    link(tree, candidate, ("Geometry",), projected, ("Geometry",), to_fallback=0)
    link(tree, proximity, ("Position",), projected, ("Position",))

    # Resample normal at the projected point so the path can later be offset
    # from the mesh without losing surface orientation.
    projected_pos = new_node(tree, "GeometryNodeInputPosition", "Projected Position", 3490, 420)
    sample_n2 = new_node(tree, "GeometryNodeSampleNearestSurface", "Projected Normal", 3700, 430)
    sample_n2.data_type = 'FLOAT_VECTOR'
    link(tree, group_in, ("Geometry",), sample_n2, ("Mesh", "Geometry"), to_fallback=0)
    link(tree, target_normal, ("Normal",), sample_n2, ("Value",))
    link(tree, projected_pos, ("Position",), sample_n2, ("Sample Position",))

    store_normal = new_node(tree, "GeometryNodeStoreNamedAttribute", f"Store {ATTR_NORMAL} Step", 3910, 80)
    store_normal.data_type = 'FLOAT_VECTOR'
    store_normal.domain = 'POINT'
    set_input(store_normal, ("Name",), ATTR_NORMAL)
    link(tree, projected, ("Geometry",), store_normal, ("Geometry",), to_fallback=0)
    link(tree, sample_n2, ("Value",), store_normal, ("Value",))

    iter_plus_one = new_node(tree, "ShaderNodeMath", "Age", 3700, -160)
    iter_plus_one.operation = 'ADD'
    link(tree, repeat_in, ("Iteration",), iter_plus_one, ("Value",), to_index=0)
    set_input(iter_plus_one, ("Value",), 1.0, index=1)

    store_age = new_node(tree, "GeometryNodeStoreNamedAttribute", f"Store {ATTR_AGE} Step", 4130, 80)
    store_age.data_type = 'FLOAT'
    store_age.domain = 'POINT'
    set_input(store_age, ("Name",), ATTR_AGE)
    link(tree, store_normal, ("Geometry",), store_age, ("Geometry",), to_fallback=0)
    link(tree, iter_plus_one, ("Value",), store_age, ("Value",))

    join_trails = new_node(tree, "GeometryNodeJoinGeometry", "Append Trail Point", 4360, -40)
    link(tree, repeat_in, ("Trails",), join_trails, ("Geometry",), from_fallback=2, to_fallback=0)
    link(tree, store_age, ("Geometry",), join_trails, ("Geometry",), to_fallback=0)

    link(tree, store_age, ("Geometry",), repeat_out, ("Tips",), to_fallback=0)
    link(tree, join_trails, ("Geometry",), repeat_out, ("Trails",), to_fallback=1)

    # Rebuild one poly spline per sf_id ----------------------------------------
    id_attr = new_node(tree, "GeometryNodeInputNamedAttribute", "Read sf_id", 2920, -330)
    id_attr.data_type = 'INT'
    set_input(id_attr, ("Name",), ATTR_ID)

    age_attr = new_node(tree, "GeometryNodeInputNamedAttribute", "Read sf_age", 2920, -470)
    age_attr.data_type = 'FLOAT'
    set_input(age_attr, ("Name",), ATTR_AGE)

    points_to_curves = new_node(tree, "GeometryNodePointsToCurves", "Build Flow Curves", 3140, -280)
    link(tree, repeat_out, ("Trails",), points_to_curves, ("Points",), from_fallback=1, to_fallback=0)
    link(tree, id_attr, ("Attribute",), points_to_curves, ("Curve Group ID", "Group ID"))
    link(tree, age_attr, ("Attribute",), points_to_curves, ("Weight",))

    normal_attr = new_node(tree, "GeometryNodeInputNamedAttribute", "Read sf_normal", 3140, -560)
    normal_attr.data_type = 'FLOAT_VECTOR'
    set_input(normal_attr, ("Name",), ATTR_NORMAL)

    normal_offset = new_node(tree, "ShaderNodeVectorMath", "Surface Offset Vector", 3370, -560)
    normal_offset.operation = 'SCALE'
    link(tree, normal_attr, ("Attribute",), normal_offset, ("Vector",), to_index=0)
    link(tree, group_in, ("Surface Offset",), normal_offset, ("Scale",), to_fallback=3)

    offset_curves = new_node(tree, "GeometryNodeSetPosition", "Offset Curves", 3570, -280)
    link(tree, points_to_curves, ("Curves",), offset_curves, ("Geometry",), to_fallback=0)
    link(tree, normal_offset, ("Vector",), offset_curves, ("Offset",))

    set_radius = new_node(tree, "GeometryNodeSetCurveRadius", "Flow Radius", 3780, -280)
    link(tree, offset_curves, ("Geometry",), set_radius, ("Curve", "Geometry"), to_fallback=0)
    link(tree, group_in, ("Flow Radius",), set_radius, ("Radius",))

    profile = new_node(tree, "GeometryNodeCurvePrimitiveCircle", "Preview Profile", 3780, -500)
    profile.mode = 'RADIUS'
    set_input(profile, ("Resolution",), 6)
    set_input(profile, ("Radius",), 1.0)

    curve_mesh = new_node(tree, "GeometryNodeCurveToMesh", "Curve to Preview Mesh", 4010, -250)
    link(tree, set_radius, ("Curve", "Geometry"), curve_mesh, ("Curve",), from_fallback=0, to_fallback=0)
    link(tree, profile, ("Curve",), curve_mesh, ("Profile Curve",), from_fallback=0, to_fallback=1)

    set_mat = new_node(tree, "GeometryNodeSetMaterial", "Set Flow Material", 4220, -250)
    link(tree, curve_mesh, ("Mesh", "Geometry"), set_mat, ("Geometry",), from_fallback=0, to_fallback=0)
    link(tree, group_in, ("Flow Material",), set_mat, ("Material",))

    result = new_node(tree, "GeometryNodeJoinGeometry", "Result", 4480, 0)
    link(tree, group_in, ("Geometry",), result, ("Geometry",), to_fallback=0)
    link(tree, set_mat, ("Geometry",), result, ("Geometry",), to_fallback=0)
    link(tree, result, ("Geometry",), group_out, ("Geometry",), to_fallback=0)
    link(tree, store_n0, ("Geometry",), group_out, ("Seeds",))
    link(tree, repeat_out, ("Trails",), group_out, ("Trails",))

    # Frames improve readability in the generated graph.
    for title, members in [
        ("01 — AUTO SOURCE", [pos_src, height_dot, stats, norm_height, lower, upper, source_mask, density, distribute, store_id, store_age0, store_n0]),
        ("02 — FLOW INTEGRATION", [repeat_in, repeat_out, tip_pos, target_normal, sample_normal, dot_gn, n_scaled, tangent, tangent_norm, rand, id_inside, seed_add, noise_dot, noise_n, noise_tan, noise_scale, direction_add, direction_norm, step_vec, candidate, candidate_pos, proximity, projected, projected_pos, sample_n2, store_normal, iter_plus_one, store_age, join_trails]),
        ("03 — CURVE OUTPUT", [id_attr, age_attr, points_to_curves, normal_attr, normal_offset, offset_curves, set_radius, profile, curve_mesh, set_mat]),
    ]:
        frame = nodes.new("NodeFrame")
        frame.label = title
        frame.name = title
        for node in members:
            node.parent = frame

    return tree


def build_static_implementation():
    """Construct a replacement without touching any live modifier or group."""
    tree = bpy.data.node_groups.new(NODE_GROUP_NAME + '.Implementation', 'GeometryNodeTree')
    tree['sf_owned'] = True
    try:
        return _populate_static(tree, ensure_flow_material())
    except Exception:
        bpy.data.node_groups.remove(tree)
        raise


def validate_static_implementation(tree):
    if any(not item.is_valid for item in tree.links):
        raise RuntimeError('Flumen candidate contains an invalid node link')
    output = next((n for n in tree.nodes if n.bl_idname == 'NodeGroupOutput' and n.is_active_output), None)
    if output is None or any(not output.inputs[n].is_linked for n in ('Geometry', 'Seeds', 'Trails')):
        raise RuntimeError('Flumen candidate is missing a required output')
    _validate_static_behavior(tree)


def _validate_static_behavior(tree):
    """Evaluate an isolated slope without changing selection or any live modifier."""
    mesh=bpy.data.meshes.new('SF validation slope')
    probe=bpy.data.objects.new('SF validation probe',mesh)
    wrapper=bpy.data.node_groups.new('SF validation wrapper','GeometryNodeTree')
    try:
        mesh.from_pydata([(-1,0,-1),(-1,0,1),(1,0,1),(1,0,-1)],[],[(0,1,2,3)])
        mesh.update(); bpy.context.scene.collection.objects.link(probe)
        add_interface_socket(wrapper,'Geometry','INPUT','NodeSocketGeometry')
        add_interface_socket(wrapper,'Geometry','OUTPUT','NodeSocketGeometry')
        gi=wrapper.nodes.new('NodeGroupInput'); go=wrapper.nodes.new('NodeGroupOutput')
        group=wrapper.nodes.new('GeometryNodeGroup'); group.node_tree=tree
        group.inputs['Steps'].default_value=8
        group.inputs['Seed Density'].default_value=64
        group.inputs['Randomness'].default_value=0
        wrapper.links.new(gi.outputs['Geometry'],group.inputs['Geometry'])
        points=wrapper.nodes.new('GeometryNodePointsToVertices')
        mod=probe.modifiers.new('Validate','NODES'); mod.node_group=wrapper
        for name in ('Seeds','Trails','Geometry'):
            if name=='Geometry':
                wrapper.links.new(group.outputs[name],go.inputs['Geometry'])
            else:
                wrapper.links.new(group.outputs[name],points.inputs['Points'])
                wrapper.links.new(points.outputs[0],go.inputs['Geometry'])
            probe.update_tag(); bpy.context.view_layer.update()
            evaluated=probe.evaluated_get(bpy.context.evaluated_depsgraph_get())
            result=evaluated.to_mesh()
            try:
                age=result.attributes.get(ATTR_AGE)
                if not len(result.vertices) or age is None:
                    raise RuntimeError(f'Flumen candidate has no valid {name.lower()}')
                if name=='Trails':
                    ids=result.attributes.get(ATTR_ID)
                    if ids is None or max(v.value for v in age.data)<8:
                        raise RuntimeError('Flumen candidate has no propagated trail history')
                    starts={ids.data[j].value:v.co.z for j,v in enumerate(result.vertices) if age.data[j].value==0}
                    if not any(v.co.z<starts.get(ids.data[j].value,v.co.z)-.001 for j,v in enumerate(result.vertices)):
                        raise RuntimeError('Flumen candidate does not move downhill')
                if name=='Geometry' and len(result.vertices)<=4:
                    raise RuntimeError('Flumen candidate has no rendered flow')
            finally:
                evaluated.to_mesh_clear()
    finally:
        bpy.data.objects.remove(probe,do_unlink=True)
        bpy.data.node_groups.remove(wrapper)
        bpy.data.meshes.remove(mesh)


def build_flumen_group(force_rebuild: bool = False):
    """Keep the public interface stable while swapping validated child graphs."""
    tree = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if tree is not None and not force_rebuild:
        return tree
    candidate = build_static_implementation()
    try:
        validate_static_implementation(candidate)
    except Exception:
        bpy.data.node_groups.remove(candidate)
        raise
    created = tree is None
    if created:
        tree = bpy.data.node_groups.new(NODE_GROUP_NAME, 'GeometryNodeTree')
        _build_interface(tree, ensure_flow_material())
    else:
        existing = {(s.in_out,s.name) for s in tree.interface.items_tree if s.item_type == 'SOCKET'}
        for name in ('Seeds','Trails'):
            if ('OUTPUT',name) not in existing:
                add_interface_socket(tree, name, 'OUTPUT', 'NodeSocketGeometry')
    old_nodes = list(tree.nodes)
    old_output = next((n for n in old_nodes if n.bl_idname == 'NodeGroupOutput' and n.is_active_output), None)
    old_children = [n.node_tree for n in old_nodes if n.bl_idname == 'GeometryNodeGroup' and n.node_tree]
    new_nodes = []
    try:
        gi = tree.nodes.new('NodeGroupInput'); new_nodes.append(gi)
        group = tree.nodes.new('GeometryNodeGroup'); new_nodes.append(group)
        group.node_tree = candidate
        group.label = 'Flumen implementation'
        go = tree.nodes.new('NodeGroupOutput'); new_nodes.append(go)
        for s in candidate.interface.items_tree:
            if s.item_type != 'SOCKET':
                continue
            if s.in_out == 'INPUT':
                tree.links.new(gi.outputs[s.name], group.inputs[s.name])
            else:
                tree.links.new(group.outputs[s.name], go.inputs[s.name])
        go.is_active_output = True
        bpy.context.view_layer.update()
        for obj in bpy.data.objects:
            if any(m.type == 'NODES' and m.node_group == tree for m in obj.modifiers):
                obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        tree['sf_schema'] = 1
        tree['sf_owned'] = True
    except Exception:
        for node in new_nodes:
            tree.nodes.remove(node)
        if old_output:
            old_output.is_active_output = True
        if created:
            bpy.data.node_groups.remove(tree)
        bpy.data.node_groups.remove(candidate)
        raise
    for node in old_nodes:
        tree.nodes.remove(node)
    for child in set(old_children):
        if child.get('sf_owned') and child.users == 0:
            bpy.data.node_groups.remove(child)
    gi.location = (-250,0); group.location = (0,0); go.location = (250,0)
    return tree
