import importlib
import unittest
from dataclasses import replace
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source


class SolverTests(unittest.TestCase):
    def make(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.solver'))
        cls=importlib.import_module('flumen.gpu.solver').FlowSolver
        cfg=FlowConfig(capacity=32,particles_per_frame=2,source_start=0,source_softness=0,gravity=(0,0,-1))
        device=require_cuda()
        source=build_source([[0,0,0],[1,0,0],[0,0,1]],[[0,1,2]],[0],cfg,device)
        solver=cls(cfg,source,device,start_frame=1,fps=30,fps_base=kwargs.get('fps_base',1))
        self.addCleanup(solver.close)
        return solver

    def test_birth_order_and_idempotence(self):
        solver=self.make()
        for frame,count in [(1,2),(2,4),(3,6)]:
            self.assertEqual(solver.seek(frame).live_count,count)
        ages=solver.pool.data.age.numpy()[:6]
        np.testing.assert_allclose(ages,[2/30,2/30,1/30,1/30,0,0],atol=1e-6)
        before=solver.snapshot()
        self.assertEqual(solver.seek(3).accepted,6)
        np.testing.assert_array_equal(solver.snapshot().positions,before.positions)

    def test_skips_backwards_and_before_start(self):
        a=self.make(); b=self.make()
        for frame in range(1,6): a.seek(frame)
        b.seek(5)
        np.testing.assert_allclose(a.snapshot().positions,b.snapshot().positions,atol=1e-6)
        a.seek(2); a.seek(5)
        np.testing.assert_allclose(a.snapshot().positions,b.snapshot().positions,atol=1e-6)
        self.assertEqual(a.seek(0).live_count,0)
        self.assertEqual(a.seek(1).live_count,2)

    def test_fps_base_subframes_and_independence(self):
        a=self.make(fps_base=1.001); b=self.make()
        a.seek(3)
        self.assertEqual(b.seek(1).live_count,2)
        self.assertAlmostEqual(float(a.pool.data.age.numpy()[0]),2*1.001/30,places=6)
        with self.assertRaises(ValueError): a.seek(3.5)
