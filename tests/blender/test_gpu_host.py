import unittest
import bpy
import flumen
from flumen import gpu_runtime as runtime


class GPUHostTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(runtime,'create_gpu_host'), 'GPU host implementation missing')
        bpy.ops.wm.read_factory_settings(use_empty=True)
        flumen.register()
        self.addCleanup(flumen.unregister)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8)
        self.source=bpy.context.object
        self.scene=bpy.context.scene
        self.scene.render.fps=30
        self.host=runtime.create_gpu_host(self.source,self.scene)

    def test_defaults_playback_idempotence_and_display(self):
        cfg=self.host.flumen_gpu
        self.assertEqual((cfg.particles_per_frame,cfg.capacity,cfg.lifetime),(64,8192,4))
        cfg.source_start=0
        cfg.source_softness=0
        runtime.reset_host(self.host)
        self.scene.frame_set(2)
        solver=runtime.get_runtime(self.host,self.scene)
        self.assertEqual(solver.stats.accepted,128)
        self.assertEqual(len(self.host.data.vertices),128)
        self.scene.frame_set(2)
        self.assertEqual(solver.stats.accepted,128)
        self.assertIn('sf_radius',self.host.data.attributes)
        nodes=self.host.modifiers[0].node_group.nodes
        self.assertTrue(any(n.bl_idname=='GeometryNodeInstanceOnPoints' for n in nodes))
        self.assertFalse(any(n.bl_idname=='GeometryNodeRealizeInstances' for n in nodes))

    def test_setting_change_requires_reset_material_does_not(self):
        old=runtime.get_runtime(self.host,self.scene)
        self.host.flumen_gpu.material=bpy.data.materials.new('water')
        self.assertIs(runtime.get_runtime(self.host,self.scene),old)
        self.host.flumen_gpu.particles_per_frame=2
        with self.assertRaisesRegex(RuntimeError,'Reset'):
            runtime.get_runtime(self.host,self.scene)
        runtime.reset_host(self.host)
        new=runtime.get_runtime(self.host,self.scene)
        self.assertIsNot(new,old)
        self.assertEqual(new.stats.accepted,2)

    def test_source_transform_invalidates_and_duplicate_is_independent(self):
        original=runtime.get_runtime(self.host,self.scene)
        duplicate=self.host.copy(); duplicate.data=self.host.data.copy()
        self.scene.collection.objects.link(duplicate)
        second=runtime.get_runtime(duplicate,self.scene)
        self.assertIsNot(second,original)
        self.source.location.x += .1
        bpy.context.view_layer.update()
        with self.assertRaisesRegex(RuntimeError,'Reset'):
            runtime.get_runtime(self.host,self.scene)

    def test_cleanup_empty_display_and_subframe(self):
        self.host.flumen_gpu.particles_per_frame=0
        runtime.reset_host(self.host)
        self.assertEqual(len(self.host.data.vertices),0)
        with self.assertRaisesRegex(ValueError,'integer'):
            runtime.evaluate_host(self.host,self.scene,1.5)
        runtime.release_all()
        self.assertEqual(len(runtime.RUNTIMES),0)
        self.assertEqual(runtime.get_runtime(self.host,self.scene).stats.live_count,0)

    def test_deletion_and_load_release_runtime(self):
        self.assertEqual(len(runtime.RUNTIMES),1)
        bpy.data.objects.remove(self.host,do_unlink=True)
        runtime.purge_deleted()
        self.assertEqual(len(runtime.RUNTIMES),0)
        runtime.on_load(None)
        self.assertEqual(len(runtime.RUNTIMES),0)

    def test_geometry_edits_invalidate_and_render_is_explicitly_unsupported(self):
        self.source.data.vertices[0].co.z += .1
        self.source.data.update()
        bpy.context.view_layer.update()
        with self.assertRaisesRegex(RuntimeError,'Reset'):
            runtime.get_runtime(self.host,self.scene)
        runtime.reset_host(self.host)
        runtime.on_render(self.scene)
        self.assertTrue(self.host.hide_render)
        with self.assertRaisesRegex(RuntimeError,'preview'):
            runtime.evaluate_host(self.host,self.scene)
        runtime.on_render_end(self.scene)

    def test_save_load_releases_cuda_and_reconstructs(self):
        import tempfile
        from pathlib import Path
        old=runtime.get_runtime(self.host,self.scene)
        name=self.host.name
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'gpu-roundtrip.blend')
            bpy.ops.wm.save_as_mainfile(filepath=path)
            bpy.ops.wm.open_mainfile(filepath=path)
            self.assertIsNone(old.pool)
            host=bpy.data.objects[name]
            solver=runtime.get_runtime(host,bpy.context.scene)
            self.assertIsNot(solver,old)
            self.assertGreater(solver.stats.live_count,0)
