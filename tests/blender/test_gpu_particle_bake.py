import tempfile
import unittest
from pathlib import Path
import bpy
import numpy as np
import flumen
from flumen import gpu_runtime as runtime


class GPUParticleBakeTests(unittest.TestCase):
    def setUp(self):
        from flumen import gpu_bake, particle_cache
        self.bake, self.cache = gpu_bake, particle_cache
        bpy.ops.wm.read_factory_settings(use_empty=True)
        flumen.register()
        self.addCleanup(flumen.unregister)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.05)
        self.source=bpy.context.object
        self.source.data.uv_layers.new(name='UVMap')
        self.scene=bpy.context.scene; self.scene.render.fps=30
        self.scene.frame_start=1; self.scene.frame_end=4
        self.host=runtime.create_gpu_host(self.source,self.scene,display_mode='POINTS',solver_backend='FIELD')
        s=self.host.flumen_gpu
        s.source_start=0; s.source_softness=0; s.particles_per_frame=16; s.field_spacing=.005; s.contact_spacing=.005
        runtime.reset_host(self.host)
        self.tmp=Path(tempfile.mkdtemp())

    def run_job(self, job):
        while not job.step(): pass

    def test_modal_bake_cancel_and_retry(self):
        self.scene.frame_set(3)
        live=runtime.get_runtime(self.host,self.scene); prepared=live.prepared
        references=prepared.references
        before=live.cache_snapshot()
        job=self.bake.BakeJob(self.host,self.scene,self.tmp/'first')
        self.assertGreater(job.estimated_bytes,0)
        self.assertEqual(prepared.references,references+1)
        job.step(); job.cancel()
        self.assertEqual(prepared.references,references)
        with self.assertRaises(ValueError): self.cache.CacheReader(self.tmp/'first')
        job=self.bake.BakeJob(self.host,self.scene,self.tmp/'second')
        self.run_job(job)
        self.assertEqual(prepared.references,references)
        self.assertIs(runtime.get_runtime(self.host,self.scene),live)
        self.assertEqual(live.current_frame,3)
        after=live.cache_snapshot()
        for name in before.arrays: np.testing.assert_array_equal(after.arrays[name],before.arrays[name])
        reader=self.cache.CacheReader(self.tmp/'second')
        reader.validate(job.header.source_fingerprint,self.bake.physical_settings(live.config))
        self.assertEqual((reader.header.start_frame,reader.header.end_frame),(1,4))
        np.testing.assert_array_equal(reader.read(3).arrays['volume'],before.arrays['volume'])
        np.testing.assert_array_equal(reader.read(3).arrays['ids'],before.arrays['ids'])
        static=reader.read_static()
        self.assertEqual(static['corner_uv'].shape,(len(static['source_triangles']),3,2))
        self.assertEqual(static['material_index'].shape,(len(static['source_triangles']),))
        actual=sum(p.stat().st_size for p in (self.tmp/'second').iterdir())
        self.assertLessEqual(actual,job.estimated_bytes)
        self.assertEqual(self.host.get('sf_particle_cache'),str(self.tmp/'second'))

    def test_failed_bake_leaves_live_state(self):
        self.scene.frame_set(2)
        live=runtime.get_runtime(self.host,self.scene); references=live.prepared.references
        (self.tmp/'taken').mkdir()
        with self.assertRaises(FileExistsError): self.bake.BakeJob(self.host,self.scene,self.tmp/'taken')
        self.assertEqual(live.prepared.references,references)
        self.assertEqual(self.scene.frame_current,2)
        legacy=runtime.create_gpu_host(self.source,self.scene)
        with self.assertRaisesRegex(ValueError,'Surface Field'): self.bake.BakeJob(legacy,self.scene,self.tmp/'legacy')
        self.assertFalse((self.tmp/'legacy').exists())

    def test_editing_writes_no_cache(self):
        original=self.cache.CacheWriter.__init__
        def forbidden(*args,**kwargs): raise AssertionError('Frame evaluation wrote a cache')
        self.cache.CacheWriter.__init__=forbidden
        try:
            for frame in (2,3,4,1,3): self.scene.frame_set(frame)
        finally:
            self.cache.CacheWriter.__init__=original
        self.assertEqual(self.host.get('sf_gpu_error',''),'')
        self.assertIsNone(self.host.get('sf_particle_cache'))
        self.assertTrue(hasattr(bpy.ops.flumen,'bake_particle_cache'))
