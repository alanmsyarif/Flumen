import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
import bpy
import numpy as np
import flumen
from flumen import gpu_runtime as runtime


def digest(folder):
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).iterdir())}


class GPUBakedWaterTests(unittest.TestCase):
    def setUp(self):
        from flumen import gpu_bake, gpu_baked_display, offline_mesher
        self.bake, self.display, self.mesher = gpu_bake, gpu_baked_display, offline_mesher
        bpy.ops.wm.read_factory_settings(use_empty=True)
        flumen.register()
        self.addCleanup(flumen.unregister)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.05)
        self.source=bpy.context.object
        self.source.data.uv_layers.new(name='UVMap')
        self.source_material=bpy.data.materials.new('Source Paint'); self.source.data.materials.append(self.source_material)
        self.source.data.color_attributes.new('Dirt','FLOAT_COLOR','POINT')
        self.scene=bpy.context.scene; self.scene.render.fps=30; self.scene.frame_start=1; self.scene.frame_end=3
        host=runtime.create_gpu_host(self.source,self.scene,display_mode='POINTS',solver_backend='FIELD')
        s=host.flumen_gpu
        s.source_start=0; s.source_softness=0; s.particles_per_frame=64; s.field_spacing=.005; s.contact_spacing=.005
        runtime.reset_host(host)
        self.tmp=Path(tempfile.mkdtemp())
        job=self.bake.BakeJob(host,self.scene,self.tmp/'cache')
        while not job.step(): pass
        from flumen.particle_cache import CacheReader
        self.mesher.mesh_cache_sequence(CacheReader(self.tmp/'cache'),self.mesher.MeshOptions(spacing=.002),
                                        self.tmp/'mesh',lambda: False)
        bpy.data.objects.remove(host,do_unlink=True); runtime.release_all()
        self.source_vertices=[tuple(v.co) for v in self.source.data.vertices]

    def test_baked_playback_without_cuda_and_material_edit(self):
        from flumen.gpu import device
        original=device.require_cuda
        def forbidden(*args,**kwargs): raise RuntimeError('CUDA must not initialize for baked playback')
        device.require_cuda=forbidden
        try:
            before=digest(self.tmp/'cache')
            water=self.display.create_baked_water(self.tmp/'cache',self.tmp/'mesh',self.scene,source=self.source)
            for frame in (1,2,3,2): self.scene.frame_set(frame)
            self.assertGreater(len(water.data.polygons),0)
            self.assertEqual(water.get('sf_baked_frame'),2)
            self.assertTrue(self.scene.render.use_lock_interface)
            velocity=water.data.attributes.get('velocity')   # Cycles motion blur on changing topology
            self.assertIsNotNone(velocity); self.assertEqual(len(velocity.data),len(water.data.vertices))
            water.data.materials[0]=bpy.data.materials.new('Other Water')
            self.scene.frame_set(3)
            self.mesher.mesh_cache_sequence(__import__('flumen.particle_cache',fromlist=['CacheReader']).CacheReader(self.tmp/'cache'),
                self.mesher.MeshOptions(spacing=.001),self.tmp/'fine',lambda: False)
            self.assertEqual(digest(self.tmp/'cache'),before)
            self.scene.render.engine='BLENDER_WORKBENCH'
            self.scene.render.resolution_x=self.scene.render.resolution_y=32
            self.scene.render.filepath=str(self.tmp/'render.png')
            camera=bpy.data.objects.new('Camera',bpy.data.cameras.new('Camera'))
            self.scene.collection.objects.link(camera); self.scene.camera=camera; camera.location=(0,-.4,0)
            camera.rotation_euler=(1.5708,0,0)
            bpy.ops.render.render(write_still=True)
            self.assertTrue((self.tmp/'render.png').exists())
        finally:
            device.require_cuda=original

    def test_baked_source_attributes_and_cleanup(self):
        water=self.display.create_baked_water(self.tmp/'cache',self.tmp/'mesh',self.scene,source=self.source)
        self.scene.frame_set(2)
        proxy=self.display.wet_proxy(water)
        self.assertIsNotNone(proxy)
        self.assertIn('UVMap',proxy.data.uv_layers)
        self.assertEqual(proxy.data.materials[0]['sf_gpu_wet_material'],True)
        self.assertIn('sf_wetness',proxy.data.attributes)
        self.assertEqual(proxy.data.attributes['sf_wetness'].domain,'CORNER')
        self.assertIn('Dirt',proxy.get('sf_baked_unsupported',''))
        wetness=np.empty(len(proxy.data.loops),np.float32)
        proxy.data.attributes['sf_wetness'].data.foreach_get('value',wetness)
        self.assertTrue(np.isfinite(wetness).all())
        names={water.name,proxy.name}; wet_material=proxy.data.materials[0].name
        bpy.data.objects.remove(water,do_unlink=True)
        self.display.purge_baked()
        self.assertFalse(any(name in bpy.data.objects for name in names))
        self.assertNotIn(wet_material,bpy.data.materials)
        self.assertIn(self.source.name,bpy.data.objects)
        self.assertIn(self.source_material.name,bpy.data.materials)
        self.assertEqual([tuple(v.co) for v in self.source.data.vertices],self.source_vertices)
        with self.assertRaises(ValueError):
            self.display.create_baked_water(self.tmp/'cache',self.tmp/'missing',self.scene)
        self.assertTrue(hasattr(bpy.ops.flumen,'mesh_particle_cache'))
