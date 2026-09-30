import importlib
import unittest
from dataclasses import replace
import numpy as np
import warp as wp
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.source'), 'Source implementation missing')
        self.api = importlib.import_module('flumen.gpu.source')
        self.cfg = FlowConfig(source_start=0, source_softness=0, capacity=8192)
        # Two vertical triangles, areas 0.5 and 2.0, disjoint x intervals.
        self.vertices = np.array([[0,0,0],[1,0,0],[0,0,1],[3,0,0],[5,0,0],[3,0,2]], dtype=np.float32)
        self.source = self.api.build_source(self.vertices, [[0,1,2],[3,4,5]], [0,1], self.cfg, require_cuda())
        self.addCleanup(self.source.close)

    def test_area_weighting_containment_and_determinism(self):
        ids = wp.array(np.arange(8000,dtype=np.int64),dtype=wp.int64,device='cuda:0')
        a = self.api.sample_source(self.source, 7, 1, ids)
        b = self.api.sample_source(self.source, 7, 1, ids)
        self.assertTrue(a.valid.numpy().all())
        np.testing.assert_array_equal(a.positions.numpy(), b.positions.numpy())
        p = a.positions.numpy()
        self.assertTrue((p[:,1] == 0).all())
        fraction = (p[:,0] >= 3).mean()
        self.assertLess(abs(fraction-.8), .025)
        first = p[p[:,0]<3]
        self.assertTrue((first[:,0]+first[:,2] <= 1.00001).all())
        second = p[p[:,0]>=3]
        self.assertTrue((second[:,0]-3+second[:,2] <= 2.00001).all())

    def test_empty_mask_is_bounded_and_bvh_reused(self):
        self.source.config = replace(self.cfg, source_start=1)
        before = self.source.mesh.id
        ids = wp.array(np.arange(16,dtype=np.int64),dtype=wp.int64,device='cuda:0')
        samples = self.api.sample_source(self.source, 0, 1, ids)
        self.assertEqual(int(samples.valid.numpy().sum()), 0)
        self.assertEqual(self.source.mesh.id, before)

    def test_rejects_empty_and_all_degenerate_geometry(self):
        for vertices, faces in [([], []), ([[0,0,0],[1,0,0],[2,0,0]], [[0,1,2]])]:
            with self.assertRaises(ValueError):
                self.api.build_source(vertices, faces, [0]*len(faces), self.cfg, require_cuda())
