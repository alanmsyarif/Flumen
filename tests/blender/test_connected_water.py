import unittest
from unittest.mock import patch
import numpy as np
import bpy
import flumen
from flumen import gpu_runtime as runtime


class ConnectedWaterTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        flumen.register(); self.addCleanup(flumen.unregister)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.015)
        self.source=bpy.context.object; self.scene=bpy.context.scene
        material=bpy.data.materials.new('Source material'); material.use_nodes=True
        self.source.data.materials.append(material)

    def make(self):
        return runtime.create_gpu_host(self.source,self.scene,display_mode='CONNECTED')

    def wet(self,host):
        return host.get('sf_gpu_wet_proxy')

    def test_legacy_and_connected_hosts_preserve_source(self):
        coords=np.array([v.co[:] for v in self.source.data.vertices])
        mats=list(self.source.data.materials)
        legacy=runtime.create_gpu_host(self.source,self.scene)
        self.assertEqual(legacy.flumen_gpu.display_mode,'DROPS')
        self.assertFalse(legacy.flumen_gpu.interactions_enabled)
        host=self.make(); solver=runtime.get_runtime(host,self.scene)
        self.assertTrue(host.flumen_gpu.interactions_enabled)
        self.assertGreater(len(host.data.polygons),0)
        self.assertEqual(len(host.modifiers),0)
        wet=self.wet(host)
        self.assertIsNotNone(wet)
        self.assertIsNot(wet.data,self.source.data)
        self.assertIn('sf_wetness',wet.data.attributes)
        self.assertEqual(len(wet.data.vertices),len(solver.topology.vertices))
        self.assertEqual(len([o for o in self.scene.objects if o.get('sf_gpu_wet_owned')]),1)
        self.assertEqual(list(self.source.data.materials),mats)
        np.testing.assert_array_equal(np.array([v.co[:] for v in self.source.data.vertices]),coords)
        shader=host.flumen_gpu.material.node_tree.nodes.get('Principled BSDF')
        self.assertAlmostEqual(shader.inputs['IOR'].default_value,1.333,places=5)
        self.assertAlmostEqual(shader.inputs['Roughness'].default_value,.05,places=5)

    def test_duplicate_delete_undo_reset_ownership(self):
        host=self.make(); first=runtime.get_runtime(host,self.scene)
        duplicate=host.copy(); self.scene.collection.objects.link(duplicate)
        runtime.evaluate_host(duplicate,self.scene)
        second=runtime.get_runtime(duplicate,self.scene)
        self.assertIsNot(first,second)
        self.assertIsNot(host.data,duplicate.data)
        self.assertIsNot(self.wet(host),self.wet(duplicate))
        self.assertIsNot(first.surface.wetness,second.surface.wetness)
        old_wet=self.wet(host)
        runtime.reset_host(host)
        self.assertIsNone(first.pool)
        # Blender's allocator can reuse a deleted object's address immediately.
        with self.assertRaises(ReferenceError): _=old_wet.name
        bpy.data.objects.remove(duplicate,do_unlink=True); runtime.purge_deleted()
        self.assertIsNone(second.pool)
        self.assertEqual(len([o for o in bpy.data.objects if o.get('sf_gpu_wet_owned')]),1)
        runtime.on_load(None)  # Shared load/undo/redo pre-handler.
        self.assertFalse(runtime.RUNTIMES)
        self.assertEqual(len([o for o in bpy.data.objects if o.get('sf_gpu_wet_owned')]),0)
        self.assertIn(self.source.name,bpy.data.objects)
        runtime.evaluate_host(host,self.scene)
        self.assertIsNotNone(self.wet(host))

    def test_failure_does_not_leave_owned_output(self):
        before={o.as_pointer() for o in bpy.data.objects}
        meshes={m.as_pointer() for m in bpy.data.meshes}
        with patch('flumen.gpu_water_display.update_water_display',side_effect=RuntimeError('Injected display allocation failure')):
            with self.assertRaisesRegex(RuntimeError,'Injected'):
                self.make()
        self.assertEqual({o.as_pointer() for o in bpy.data.objects},before)
        self.assertEqual({m.as_pointer() for m in bpy.data.meshes},meshes)
        self.assertFalse(runtime.RUNTIMES)

    def test_shader_changes_and_same_frame_preserve_state(self):
        host=self.make(); self.scene.frame_set(2)
        solver=runtime.get_runtime(host,self.scene)
        accepted=solver.stats.accepted; wet=solver.surface.wetness.numpy().copy()
        geometry=solver.water_snapshot()
        material=host.flumen_gpu.material
        material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.1,.2,.3,1.)
        host.flumen_gpu.material=bpy.data.materials.new('User preview material')
        for _ in range(3): runtime.evaluate_host(host,self.scene)
        self.assertIs(solver,runtime.get_runtime(host,self.scene))
        self.assertEqual(solver.stats.accepted,accepted)
        self.assertIs(geometry,solver.water_snapshot())
        np.testing.assert_array_equal(solver.surface.wetness.numpy(),wet)
        host.flumen_gpu.reconstruction_scale=2.
        self.assertIn('Reset',host['sf_gpu_error'])
        with self.assertRaisesRegex(RuntimeError,'Reset'): runtime.get_runtime(host,self.scene)
        runtime.reset_host(host)
        host.flumen_gpu.wetness_drying_rate=.2
        with self.assertRaisesRegex(RuntimeError,'Reset'): runtime.get_runtime(host,self.scene)

    def test_geometry_diagnostics_are_available_on_host(self):
        host=self.make(); solver=runtime.get_runtime(host,self.scene)
        solver.geometry_buffers.vertex_budget=3
        solver._water_cache=None
        runtime.evaluate_host(host,self.scene)
        self.assertIn('budget',host['sf_gpu_geometry_error'].lower())
        self.assertGreaterEqual(host['sf_gpu_coarsening'],1.)

    def test_save_reload_reconstructs_only_owned_output(self):
        import tempfile
        from pathlib import Path
        host=self.make(); name=host.name; source_name=self.source.name
        old=runtime.get_runtime(host,self.scene)
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'connected-roundtrip.blend')
            bpy.ops.wm.save_as_mainfile(filepath=path)
            bpy.ops.wm.open_mainfile(filepath=path)
            self.assertIsNone(old.pool)
            host=bpy.data.objects[name]
            runtime.evaluate_host(host,bpy.context.scene)
            self.assertGreater(len(host.data.polygons),0)
            self.assertEqual(len([o for o in bpy.data.objects if o.get('sf_gpu_wet_owned')]),1)
            self.assertIs(host.flumen_gpu.source,bpy.data.objects[source_name])
