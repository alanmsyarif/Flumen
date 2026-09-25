import unittest
from unittest.mock import patch
import bpy
from flumen import build_simulation
from helpers import reset_scene


class SetupTests(unittest.TestCase):
    def setUp(self):
        reset_scene()

    def create(self, source, **settings):
        self.assertTrue(callable(getattr(build_simulation,'create_simulation_host',None)), 'Animated setup missing')
        return build_simulation.create_simulation_host(source,**settings)

    def test_creates_independent_identity_host_and_render_geometry(self):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12)
        source=bpy.context.object
        source.location=(2,3,1)
        host=self.create(source)
        self.assertIsNone(host.parent)
        self.assertEqual(tuple(host.location),(0,0,0))
        self.assertEqual(tuple(host.scale),(1,1,1))
        self.assertEqual(tuple(source.location),(2,3,1))
        self.assertFalse(source.hide_render)
        evaluated=host.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh=evaluated.to_mesh()
        self.assertGreater(len(mesh.vertices),0)
        evaluated.to_mesh_clear()

    def test_rejects_invalid_units_empty_source_and_nonfinite_settings(self):
        bpy.ops.mesh.primitive_uv_sphere_add()
        source=bpy.context.object
        bpy.context.scene.unit_settings.scale_length=.01
        try:
            with self.assertRaises(ValueError): self.create(source)
        finally:
            bpy.context.scene.unit_settings.scale_length=1
        with self.assertRaises(ValueError): self.create(source,Resistance=float('nan'))
        with self.assertRaises(ValueError): self.create(source,**{'Drop Radius':-1})
        with self.assertRaises(ValueError): self.create(source,Unknown=1)
        empty=bpy.data.objects.new('Empty',bpy.data.meshes.new('Empty'))
        bpy.context.collection.objects.link(empty)
        with self.assertRaises(ValueError): self.create(empty)

    def test_failed_construction_can_be_retried(self):
        with patch.object(build_simulation,'build_free_group',side_effect=RuntimeError('probe')):
            with self.assertRaises(RuntimeError): build_simulation.build_simulation_group()
        self.assertIsNone(bpy.data.node_groups.get('FL_SurfaceFlowSim'))
        tree=build_simulation.build_simulation_group()
        self.assertEqual(tree['sf_schema'],1)
        self.assertTrue(tree.nodes.get('Simulation State'))

    def test_flat_source_reports_actionable_error(self):
        bpy.ops.mesh.primitive_plane_add()
        with self.assertRaisesRegex(ValueError,'height'):
            self.create(bpy.context.object)

    def test_unowned_helper_group_is_not_reused(self):
        other=bpy.data.node_groups.new('SF_Source','GeometryNodeTree')
        with self.assertRaisesRegex(ValueError,'incompatible'):
            build_simulation.build_simulation_group()
        self.assertEqual(bpy.data.node_groups.get('SF_Source'),other)
        self.assertIsNone(bpy.data.node_groups.get('FL_SurfaceFlowSim'))

    def test_drop_radius_scales_with_volume(self):
        self.assertTrue(callable(getattr(build_simulation,'create_simulation_host',None)))
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('flumen.sim_output'))
        from flumen.sim_output import build_output_group
        from helpers import particles_object,particle_at
        tree=build_output_group()
        extents=[]
        for volume in [1e-9,2e-9]:
            obj=particles_object([particle_at((0,0,0),state=1,volume=volume)])
            mod=obj.modifiers.new('Output','NODES')
            wrapper=bpy.data.node_groups.new('Render fixture','GeometryNodeTree')
            wrapper.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
            wrapper.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
            gi=wrapper.nodes.new('NodeGroupInput'); go=wrapper.nodes.new('NodeGroupOutput')
            points=wrapper.nodes.new('GeometryNodeMeshToPoints')
            node=wrapper.nodes.new('GeometryNodeGroup'); node.node_tree=tree
            wrapper.links.new(gi.outputs[0],points.inputs['Mesh'])
            wrapper.links.new(points.outputs['Points'],node.inputs['Particles'])
            wrapper.links.new(node.outputs['Geometry'],go.inputs[0])
            mod.node_group=wrapper
            evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=evaluated.to_mesh()
            extents.append(max(v.co.length for v in mesh.vertices))
            evaluated.to_mesh_clear()
        self.assertAlmostEqual(extents[1]/extents[0],2**(1/3),places=5)
