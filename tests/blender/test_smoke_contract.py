import unittest
import bpy
from scripts import smoke_test_blender
from flumen.build_nodes import build_flumen_group
from helpers import reset_scene


class SmokeTests(unittest.TestCase):
    def test_bare_surface_does_not_pass_smoke(self):
        reset_scene()
        self.assertTrue(callable(getattr(smoke_test_blender, 'validate_output', None)))
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
        obj = bpy.context.object
        tree = build_flumen_group()
        mod = obj.modifiers.new('Flow', 'NODES'); mod.node_group = tree
        implementation = next(n.node_tree for n in tree.nodes if n.type == 'GROUP')
        result = implementation.nodes['Result']
        for item in list(result.inputs['Geometry'].links):
            if item.from_node.name == 'Set Flow Material':
                implementation.links.remove(item)
        with self.assertRaises(AssertionError):
            smoke_test_blender.validate_output(obj)
