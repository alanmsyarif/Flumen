import importlib
import unittest
import bpy
import numpy as np


class GPUExtractionTests(unittest.TestCase):
    def test_evaluated_world_coordinates_and_islands(self):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu_runtime'))
        from flumen.gpu_runtime import extract_source
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new('source')
        mesh.from_pydata([(0,0,0),(1,0,0),(0,0,1),(3,0,0),(4,0,0),(3,0,1)],[],[(0,1,2),(3,4,5)])
        obj = bpy.data.objects.new('source',mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = (4,2,1)
        obj.scale = (-2,1,3)
        bpy.context.view_layer.update()
        vertices, faces, islands = extract_source(obj,bpy.context.evaluated_depsgraph_get())
        np.testing.assert_allclose(vertices[:3],[[4,2,1],[2,2,1],[4,2,4]])
        self.assertEqual(len(faces),2)
        self.assertNotEqual(islands[0],islands[1])

    def test_mirrored_transform_preserves_outward_winding(self):
        from flumen.gpu_runtime import extract_source
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh=bpy.data.meshes.new('triangle')
        mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)],[],[(0,1,2)])
        obj=bpy.data.objects.new('triangle',mesh); bpy.context.scene.collection.objects.link(obj)
        obj.scale=(-1,1,1); bpy.context.view_layer.update()
        vertices,faces,_=extract_source(obj,bpy.context.evaluated_depsgraph_get())
        a,b,c=vertices[faces[0]]
        self.assertGreater(np.cross(b-a,c-a)[2],0,'Mirrored source normal points inward')
