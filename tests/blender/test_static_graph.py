import unittest
import bpy
from flumen import node_utils
from flumen.build_nodes import build_flumen_group
from helpers import reset_scene, debug_modifier, read_points, mesh_object


class StaticGraphTests(unittest.TestCase):
    def setUp(self):
        reset_scene()

    def test_duplicate_operands_are_explicit(self):
        self.assertTrue(callable(getattr(node_utils, 'resolve_socket', None)), 'Explicit socket resolver missing')
        tree = bpy.data.node_groups.new('Socket test', 'GeometryNodeTree')
        node = tree.nodes.new('ShaderNodeMath')
        resolve = node_utils.resolve_socket
        self.assertNotEqual(resolve(node.inputs, index=0), resolve(node.inputs, index=1))
        with self.assertRaises(ValueError):
            resolve(node.inputs, name='Value')
        with self.assertRaises(ValueError):
            resolve(node.inputs, identifier='missing')

    def sphere(self, output, **settings):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
        obj = bpy.context.object
        tree = build_flumen_group(force_rebuild=True)
        self.assertIn(output, [s.name for s in tree.interface.items_tree if s.in_out == 'OUTPUT'])
        debug_modifier(obj, tree, output, settings)
        return obj

    def test_zero_noise_paths_move_downhill_and_have_integer_ages(self):
        points = read_points(self.sphere('Trails', Steps=8, Randomness=0.0))
        self.assertGreater(len(points), 20)
        paths = {}
        for p in points:
            paths.setdefault(p['sf_id'], []).append(p)
        moved = 0
        for path in paths.values():
            path.sort(key=lambda p: p['sf_age'])
            self.assertEqual([p['sf_age'] for p in path], list(range(9)))
            z0, z1 = path[0]['position'][2], path[-1]['position'][2]
            self.assertLessEqual(z1, z0 + 1e-5)
            moved += z1 < z0 - 1e-3
        self.assertGreater(moved, 0.9 * len(paths))

    def test_source_threshold_changes_distribution(self):
        low = read_points(self.sphere('Seeds', **{'Source Start': 0.3, 'Seed Density': 60.0}))
        high = read_points(self.sphere('Seeds', **{'Source Start': 0.9, 'Seed Density': 60.0}))
        self.assertGreater(len(low), len(high))
        self.assertGreater(len(high), 0)
        self.assertGreater(min(p['position'][2] for p in high), 0.55)

    def test_hard_threshold_and_flat_surface(self):
        points = read_points(self.sphere('Seeds', **{'Source Start': 0.8, 'Source Softness': 0.0}))
        self.assertGreater(len(points), 0)
        self.assertGreater(min(p['position'][2] for p in points), 0.45)
        plane = mesh_object([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [(0,1,2,3)])
        debug_modifier(plane, build_flumen_group(), 'Seeds')
        self.assertEqual(read_points(plane), [])
