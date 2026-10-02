import unittest
import bpy
import numpy as np
import flumen
from flumen import gpu_runtime as runtime


class GPUFieldHostTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        flumen.register()
        self.addCleanup(flumen.unregister)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.05)
        self.source=bpy.context.object
        self.scene=bpy.context.scene
        self.scene.render.fps=30

    def field_host(self):
        host=runtime.create_gpu_host(self.source,self.scene,display_mode='POINTS',solver_backend='FIELD')
        settings=host.flumen_gpu
        self.assertEqual((settings.solver_backend,settings.display_mode),('FIELD','POINTS'))
        settings.source_start=0; settings.source_softness=0; settings.particles_per_frame=16
        settings.field_spacing=.005; settings.contact_spacing=.005
        runtime.reset_host(host)
        return host

    def ledger(self, solver):
        s=solver.stats
        return np.array([s.accepted,s.live_count,s.emitted_volume,s.live_volume,s.removed_volume,s.frame])

    def test_physical_reset_reuses_preparation(self):
        host=self.field_host(); self.scene.frame_set(3)
        solver=runtime.get_runtime(host,self.scene); prepared=solver.prepared
        host.flumen_gpu.gravity=(0,0,-4)
        with self.assertRaisesRegex(RuntimeError,'Reset'): runtime.get_runtime(host,self.scene)
        runtime.reset_host(host)
        reset=runtime.get_runtime(host,self.scene)
        self.assertIsNot(reset,solver)
        self.assertIs(reset.prepared,prepared)
        self.assertEqual(reset.current_frame,3)
        host.flumen_gpu.particles_per_frame=4; host.flumen_gpu.time_scale=.5
        runtime.reset_host(host)
        self.assertIs(runtime.get_runtime(host,self.scene).prepared,prepared)
        host.flumen_gpu.field_spacing=.01
        runtime.reset_host(host)
        respaced=runtime.get_runtime(host,self.scene).prepared
        self.assertIsNot(respaced,prepared)
        self.assertEqual(prepared.references,0)

    def test_cosmetic_edits_are_inert(self):
        host=self.field_host(); self.scene.frame_set(3)
        solver=runtime.get_runtime(host,self.scene); prepared=solver.prepared
        before=self.ledger(solver)
        settings=host.flumen_gpu
        settings.material=bpy.data.materials.new('water')
        settings.point_color=(1,0,0,1); settings.point_size=5; settings.display_limit=7
        settings.reconstruction_scale=2.
        host.hide_viewport=True; host.hide_viewport=False
        self.scene.frame_set(3)
        self.assertEqual(host.get('sf_gpu_error',''),'')
        self.assertIs(runtime.get_runtime(host,self.scene),solver)
        self.assertIs(solver.prepared,prepared)
        np.testing.assert_array_equal(self.ledger(solver),before)

    def test_source_edit_invalidates_contacts(self):
        host=self.field_host(); self.scene.frame_set(2)
        prepared=runtime.get_runtime(host,self.scene).prepared
        self.source.location.x+=.01
        bpy.context.view_layer.update()
        with self.assertRaisesRegex(RuntimeError,'Reset'): runtime.get_runtime(host,self.scene)
        runtime.reset_host(host)
        moved=runtime.get_runtime(host,self.scene).prepared
        self.assertIsNot(moved,prepared)
        self.source.data.vertices[0].co.z+=.01; self.source.data.update()
        bpy.context.view_layer.update()
        runtime.reset_host(host)
        self.assertIsNot(runtime.get_runtime(host,self.scene).prepared,moved)

    def test_field_host_lifecycle(self):
        first=self.field_host(); material=bpy.data.materials.new('user water'); material.use_fake_user=True
        second=self.field_host(); second.flumen_gpu.material=material
        self.scene.frame_set(2)
        a=runtime.get_runtime(first,self.scene); b=runtime.get_runtime(second,self.scene)
        self.assertIsNot(a.prepared,b.prepared)
        prepared=a.prepared
        bpy.data.objects.remove(first,do_unlink=True); runtime.purge_deleted()
        self.assertEqual(prepared.references,0)
        self.scene.frame_set(3)
        self.assertEqual(b.current_frame,3)
        self.assertIn(material.name,bpy.data.materials)
        runtime.release_all(); runtime.release_all()
        self.assertEqual(b.prepared,None)
        legacy=runtime.create_gpu_host(self.source,self.scene)
        self.assertEqual((legacy.flumen_gpu.solver_backend,legacy.flumen_gpu.display_mode),('LEGACY','DROPS'))
