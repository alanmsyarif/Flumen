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

    def test_empty_field_and_baked_output_rejected(self):
        from types import SimpleNamespace
        self.assertTrue(callable(getattr(smoke_test_blender, 'validate_field_output', None)))
        self.assertTrue(callable(getattr(smoke_test_blender, 'validate_baked_output', None)))
        stats = lambda **k: SimpleNamespace(stats=SimpleNamespace(**dict(dict(
            live_count=10, emitted_volume=1., live_volume=.9, removed_volume=.1), **k)))
        with self.assertRaises(AssertionError): smoke_test_blender.validate_field_output(stats(live_count=0))
        with self.assertRaises(AssertionError): smoke_test_blender.validate_field_output(stats(live_volume=.5))
        reset_scene()
        empty = bpy.data.objects.new('Baked', bpy.data.meshes.new('Baked'))
        bpy.context.scene.collection.objects.link(empty); empty['sf_baked_frame'] = 1
        with self.assertRaises(AssertionError): smoke_test_blender.validate_baked_output(empty)
        bpy.ops.mesh.primitive_cube_add(); cube = bpy.context.object
        with self.assertRaises(AssertionError): smoke_test_blender.validate_baked_output(cube)   # not a baked host
        cube['sf_baked_frame'] = 1
        smoke_test_blender.validate_baked_output(cube)
