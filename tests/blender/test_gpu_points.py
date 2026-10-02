import unittest
import bpy
import flumen
from flumen import gpu_runtime as runtime


class GPUPointTests(unittest.TestCase):
    def setUp(self):
        from flumen import gpu_point_display
        self.display=gpu_point_display
        bpy.ops.wm.read_factory_settings(use_empty=True)
        flumen.register()
        self.addCleanup(flumen.unregister)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8)
        self.source=bpy.context.object
        self.scene=bpy.context.scene
        self.scene.render.fps=30

    def host(self):
        host=runtime.create_gpu_host(self.source,self.scene,display_mode='POINTS')
        host.flumen_gpu.source_start=0; host.flumen_gpu.source_softness=0
        runtime.reset_host(host)
        return host

    def test_points_publish_without_mesh_vertices_or_instances(self):
        host=self.host(); self.scene.frame_set(2)
        solver=runtime.get_runtime(host,self.scene)
        batch=self.display.published_batch(host)
        self.assertEqual(batch.displayed_count,solver.stats.live_count)
        self.assertEqual(batch.displayed_count,128)
        self.assertEqual(len(host.data.vertices),0)
        self.assertFalse(any(m.type=='NODES' and m.show_viewport and m.node_group and m.node_group.get('sf_gpu_display')
                             for m in host.modifiers))
        host.flumen_gpu.display_limit=10
        self.assertEqual(self.display.published_batch(host).displayed_count,10)
        self.assertEqual(solver.stats.displayed_count,10)

    def test_point_draw_is_read_only(self):
        host=self.host(); self.scene.frame_set(2)
        solver=runtime.get_runtime(host,self.scene)
        before=(solver.stats.accepted,solver.stats.live_count,solver.current_frame)
        host.flumen_gpu.point_color=(1,0,0,1)
        host.flumen_gpu.point_size=6
        self.scene.frame_set(2)
        self.assertIs(runtime.get_runtime(host,self.scene),solver)
        self.assertEqual((solver.stats.accepted,solver.stats.live_count,solver.current_frame),before)
        self.assertEqual(host.get('sf_gpu_error',''),'')

    def test_point_ownership_and_allocation_failure(self):
        first=self.host()
        bpy.ops.mesh.primitive_cube_add(location=(3,0,0)); self.source=bpy.context.object
        second=self.host(); self.scene.frame_set(2)
        a=self.display.published_batch(first); b=self.display.published_batch(second)
        self.assertIsNot(a,b); self.assertIsNot(a.xyzr,b.xyzr)
        self.assertTrue(self.display.handler_installed())
        key=first.as_pointer()
        bpy.data.objects.remove(first,do_unlink=True)
        runtime.purge_deleted()
        self.assertNotIn(key,self.display.owned_hosts())
        self.display.release_point_display(second); self.display.release_point_display(second)
        self.assertFalse(self.display.handler_installed())
        original=bpy.types.SpaceView3D.draw_handler_add
        def fail(*args): raise RuntimeError('handler allocation failed')
        bpy.types.SpaceView3D.draw_handler_add=fail
        try:
            with self.assertRaises(RuntimeError): self.display.create_point_display(second)
        finally:
            bpy.types.SpaceView3D.draw_handler_add=original
        self.assertEqual(self.display.owned_hosts(),set())
        self.display.create_point_display(second)
        self.assertTrue(self.display.handler_installed())
        runtime.unregister_handlers(); runtime.unregister_handlers()
        self.assertFalse(self.display.handler_installed())
        self.assertEqual(self.display.owned_hosts(),set())
        runtime.register_handlers()
