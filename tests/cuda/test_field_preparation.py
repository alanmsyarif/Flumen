import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.state import ParticlePool


class FieldPreparationTests(unittest.TestCase):
    def modules(self):
        for name in ('prepared', 'field_solver', 'surface_chart'):
            self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.'+name),
                                 'Source-local field preparation is missing')
        from flumen.gpu.prepared import prepare_source
        from flumen.gpu.field_solver import FieldBuffers, deposit_attached
        return prepare_source, FieldBuffers, deposit_attached

    def source(self, vertices=None, triangles=None, islands=None):
        cfg = FlowConfig(capacity=8, radius=.0001, source_start=0, source_softness=0)
        device = require_cuda()
        source = build_source(vertices or [[0,0,0],[.1,0,0],[0,.1,0]],
                              triangles or [[0,1,2]], islands or [0], cfg, device)
        return source, cfg, device

    def test_subspacing_anchor_deposition(self):
        prepare, Buffers, deposit = self.modules()
        source, cfg, device = self.source()
        prepared = prepare(source, 'triangle', .04, .02)
        self.addCleanup(prepared.release)
        pool = ParticlePool(cfg, device); self.addCleanup(pool.close)
        buffers = Buffers(prepared.chart, device.alias); self.addCleanup(buffers.close)
        anchors = [[1,0],[0,1],[0,0],[.5,.5],[.5,0],[0,.5],[.2,.3],[.3,.2]]
        pool.data.active.assign([1]*8)
        pool.data.bary.assign(anchors)
        pool.data.volume.assign([1e-12]*8)
        pool.data.velocity.assign([[1,2,0]]*8)
        deposit(pool, prepared, buffers)
        self.assertAlmostEqual(float(buffers.volume.numpy().sum()), 8e-12, delta=8e-17)
        np.testing.assert_allclose(buffers.momentum.numpy().sum(axis=0), [8e-12,16e-12,0], rtol=1e-5, atol=1e-20)
        self.assertAlmostEqual(float(prepared.chart.areas.sum()), .005, delta=.00005)
        self.assertLessEqual(len(prepared.chart.vertices), 100000)
        self.assertGreater(prepared.chart.effective_spacing, cfg.radius)

    def test_chart_fold_and_edge_locality(self):
        prepare, Buffers, deposit = self.modules()
        # Almost coincident opposite sheets, with deliberately identical island labels.
        source, cfg, device = self.source(
            [[0,0,0],[.1,0,0],[0,.1,0],[0,0,.0001],[.1,0,.0001],[0,.1,.0001]],
            [[0,1,2],[3,5,4]], [0,0])
        prepared = prepare(source, 'sheets', .04, .02); self.addCleanup(prepared.release)
        pool = ParticlePool(cfg, device); self.addCleanup(pool.close)
        buffers = Buffers(prepared.chart, device.alias); self.addCleanup(buffers.close)
        pool.data.active.assign([1]+[0]*7); pool.data.bary.assign([[.2,.3]]*8)
        pool.data.volume.assign([1e-12]*8)
        deposit(pool, prepared, buffers)
        other_nodes = np.unique(prepared.chart.triangles[prepared.chart.original_faces==1])
        self.assertEqual(float(buffers.volume.numpy()[other_nodes].sum()), 0.)
        self.assertGreater(prepared.contact.ambiguous_count, 0)
        self.assertTrue((prepared.chart.source_adjacency[0] == -1).all())

    def test_preparation_budgets_and_release(self):
        prepare, _, _ = self.modules()
        source, _, _ = self.source()
        prepared = prepare(source, 'budget', .00005, .00005)
        self.assertLessEqual(len(prepared.chart.vertices), 100000)
        self.assertLessEqual(prepared.contact.sample_count, 2097152)
        self.assertGreaterEqual(prepared.chart.effective_spacing, .00005)
        self.assertGreaterEqual(prepared.contact.effective_spacing, .00005)
        self.assertGreater(prepared.contact.coarsening_factor, 1.)
        prepared.retain(); prepared.retain(); prepared.release(); prepared.release()
        self.assertIsNotNone(prepared.source.mesh)
        prepared.release(); prepared.release()  # Idempotent final cleanup.
        self.assertIsNone(source.mesh)
        self.assertIsNone(prepared.chart.points_gpu)
        self.assertIsNone(prepared.contact.distance)

    def test_nonmanifold_barrier_and_failed_preparation(self):
        prepare, _, _ = self.modules()
        source, _, _ = self.source([[0,0,0],[.1,0,0],[0,.1,0],[0,-.1,0],[0,0,.1]],
                                  [[0,1,2],[1,0,3],[0,1,4]], [0,0,0])
        from flumen.gpu.surface_chart import build_chart
        chart = build_chart(source, .04)
        self.assertGreater(chart.barrier_count, 0)
        self.assertTrue((chart.source_adjacency[:,2] == -1).all())
        chart.close()
        with patch('flumen.gpu.prepared.build_contact_field', side_effect=MemoryError('injected')):
            with self.assertRaises(MemoryError): prepare(source, 'fail', .04, .02)
        self.assertIsNone(source.mesh)

    def test_degenerate_face_provenance(self):
        self.modules()
        source, _, _ = self.source([[0,0,0],[.1,0,0],[0,.1,0]], [[0,0,0],[0,1,2]], [0,1])
        self.addCleanup(source.close)
        np.testing.assert_array_equal(source.evaluated_triangle_ids, [1])

    def test_solver_borrows_preparation_and_sampling_refresh(self):
        from dataclasses import replace
        prepare, _, _ = self.modules()
        source, cfg, device = self.source()
        prepared = prepare(source,'borrow',.04,.02); self.addCleanup(prepared.release)
        from flumen.gpu.solver import FlowSolver
        solver = FlowSolver(cfg,source,device,prepared=prepared)
        self.addCleanup(solver.close)
        handle = source.mesh.id
        source.refresh_sampling(replace(cfg,gravity=(-1,0,0),source_start=.5))
        self.assertEqual(source.mesh.id,handle)
        np.testing.assert_allclose(source.up,[1,0,0])
        self.assertAlmostEqual(source.span,.1,places=6)
        solver.close()
        self.assertIsNotNone(source.mesh)
        self.assertEqual(prepared.references,1)
