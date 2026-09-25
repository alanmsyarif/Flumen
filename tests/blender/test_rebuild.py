from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import bpy
from flumen import build_nodes
from helpers import reset_scene, set_input_value, read_input_value, interface_ids


class RebuildTests(unittest.TestCase):
    def setUp(self):
        reset_scene()
        self.directory=tempfile.TemporaryDirectory(prefix='flumen-rebuild-')
        self.addCleanup(self.directory.cleanup)

    def fixture(self):
        tree = build_nodes.build_flumen_group()
        mods = []
        for name in ('A', 'B'):
            bpy.ops.mesh.primitive_cube_add()
            bpy.context.object.name = name
            mod = bpy.context.object.modifiers.new('Flumen', 'NODES')
            mod.node_group = tree
            mods.append(mod)
        return tree, mods

    def test_rebuild_preserves_identifiers_values_and_external_links(self):
        tree, (a,b) = self.fixture()
        set_input_value(a, 'Steps', 7)
        set_input_value(b, 'Steps', 13)
        identifiers = interface_ids(tree)
        parent = bpy.data.node_groups.new('Parent', 'GeometryNodeTree')
        parent.use_fake_user = True
        group = parent.nodes.new('GeometryNodeGroup'); group.node_tree = tree
        value = parent.nodes.new('ShaderNodeValue')
        parent.links.new(value.outputs[0], group.inputs['Steps'])
        build_nodes.build_flumen_group(force_rebuild=True)
        self.assertEqual(read_input_value(a, 'Steps'), 7)
        self.assertEqual(read_input_value(b, 'Steps'), 13)
        self.assertEqual(interface_ids(tree), identifiers)
        self.assertTrue(group.inputs['Steps'].is_linked)

    def test_reuse_does_not_add_nodes(self):
        tree, _ = self.fixture()
        count = len(tree.nodes)
        result = build_nodes.build_flumen_group(force_rebuild=False)
        self.assertEqual(result, tree)
        self.assertEqual(len(tree.nodes), count)

    def test_candidate_failure_keeps_working_group(self):
        tree, (mod, _) = self.fixture()
        set_input_value(mod, 'Steps', 7)
        before = [(n.name, n.bl_idname) for n in tree.nodes]
        self.assertTrue(callable(getattr(build_nodes, 'build_static_implementation', None)))
        with patch.object(build_nodes, 'build_static_implementation', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                build_nodes.build_flumen_group(force_rebuild=True)
        self.assertEqual([(n.name,n.bl_idname) for n in tree.nodes], before)
        self.assertEqual(read_input_value(mod, 'Steps'), 7)

    def test_driver_survives_rebuild_and_reload(self):
        tree, (a, _) = self.fixture()
        item = next(s for s in tree.interface.items_tree if s.name=='Step Length')
        prop = getattr(a.properties.inputs, item.identifier)
        curve = prop.driver_add('value')
        curve.driver.expression = '0.0125'
        path = curve.data_path
        build_nodes.build_flumen_group(force_rebuild=True)
        bpy.context.scene.frame_set(2)
        self.assertEqual(a.id_data.animation_data.drivers[0].data_path, path)
        target = Path(self.directory.name)/'roundtrip.blend'
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(target))
        bpy.ops.wm.open_mainfile(filepath=str(target))
        mod = bpy.data.objects['A'].modifiers['Flumen']
        self.assertEqual(mod.id_data.animation_data.drivers[0].data_path, path)
        self.assertAlmostEqual(read_input_value(mod, 'Step Length'), 0.0125)

    def test_validation_failure_keeps_values_and_active_output(self):
        tree,(a,_)=self.fixture()
        set_input_value(a,'Steps',7)
        output=next(n for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output)
        with patch.object(build_nodes,'validate_static_implementation',side_effect=RuntimeError('invalid graph')):
            with self.assertRaises(RuntimeError):
                build_nodes.build_flumen_group(True)
        self.assertTrue(output.is_active_output)
        self.assertEqual(read_input_value(a,'Steps'),7)

    def test_behaviorally_broken_candidate_does_not_replace_working_graph(self):
        tree,(a,_)=self.fixture()
        set_input_value(a,'Steps',7)
        original_nodes=list(tree.nodes)
        candidate=build_nodes.build_static_implementation()
        output=next(n for n in candidate.nodes if n.type=='GROUP_OUTPUT')
        gi=next(n for n in candidate.nodes if n.type=='GROUP_INPUT')
        # All required outputs stay linked, but there is no propagation.
        for name in ('Seeds','Trails'):
            candidate.links.new(gi.outputs['Geometry'],output.inputs[name])
        with patch.object(build_nodes,'build_static_implementation',return_value=candidate):
            with self.assertRaises(RuntimeError): build_nodes.build_flumen_group(True)
        self.assertEqual(list(tree.nodes),original_nodes)
        self.assertEqual(read_input_value(a,'Steps'),7)

    def test_build_second_object_preserves_first_and_unrelated_modifier(self):
        from flumen.operators import SF_OT_build
        tree,(a,b)=self.fixture()
        set_input_value(a,'Steps',7)
        obj=b.id_data
        obj.modifiers.remove(b)
        other=bpy.data.node_groups.new('Unrelated','GeometryNodeTree')
        other.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        other.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        gi=other.nodes.new('NodeGroupInput'); go=other.nodes.new('NodeGroupOutput')
        other.links.new(gi.outputs[0],go.inputs[0])
        unrelated=obj.modifiers.new('Flumen','NODES'); unrelated.node_group=other
        bpy.context.view_layer.objects.active=obj
        bpy.utils.register_class(SF_OT_build)
        try:
            self.assertEqual(bpy.ops.flumen.build(),{'FINISHED'})
            self.assertEqual(read_input_value(a,'Steps'),7)
            self.assertEqual(unrelated.node_group,other)
            self.assertEqual(sum(m.node_group==tree for m in obj.modifiers if m.type=='NODES'),1)
        finally:
            bpy.utils.unregister_class(SF_OT_build)
