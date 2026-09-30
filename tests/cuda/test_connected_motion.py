import unittest
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.solver import FlowSolver


class ConnectedMotionTests(unittest.TestCase):
    def make(self,lifetime=4.):
        cfg=FlowConfig(capacity=32,particles_per_frame=2,initial_coating_count=4,
                       interactions_enabled=True,source_start=0,source_softness=0,
                       gravity=(0,0,-1),radius=.001,lifetime=lifetime)
        device=require_cuda()
        source=build_source([[-.01,0,-.01],[.01,0,-.01],[0,0,.01]],[[0,1,2]],[0],cfg,device)
        solver=FlowSolver(cfg,source,device)
        self.addCleanup(solver.close)
        return solver

    def test_synchronous_steps_and_zero_dt(self):
        solver=self.make(); solver.seek(1)
        self.assertIsNotNone(getattr(solver,'interaction',None),'Coupled runtime missing')
        from flumen.gpu.motion import advance_coupled
        before=solver.pool.data.position.numpy()
        advance_coupled(solver.pool,solver.source,solver.topology,solver.config,solver.interaction,0)
        np.testing.assert_array_equal(solver.pool.data.position.numpy(),before)
        solver.seek(2)
        ages=solver.pool.data.age.numpy()[solver.pool.data.active.numpy()==1]
        self.assertTrue(np.isfinite(ages).all())
        self.assertTrue((ages<=1/30+1e-6).all())
        self.assertGreaterEqual(solver.stats.substeps,8)
        self.assertLessEqual(solver.stats.substeps,64)

    def test_coupled_replay_and_volume_ledger(self):
        a=self.make(); b=self.make()
        self.assertIsNotNone(getattr(a,'interaction',None),'Coupled runtime missing')
        for frame in range(1,7): a.seek(frame)
        b.seek(6)
        np.testing.assert_allclose(a.snapshot().positions,b.snapshot().positions,atol=1e-6)
        a.seek(2); a.seek(6)
        np.testing.assert_allclose(a.snapshot().positions,b.snapshot().positions,atol=1e-6)
        self.assertAlmostEqual(a.stats.emitted_volume,a.stats.live_volume+a.stats.removed_volume,delta=a.stats.emitted_volume*1e-5)

    def test_shared_overflow_and_slot_reuse(self):
        solver=self.make(); solver.seek(1)
        solver.pool.data.velocity.assign([[100,0,0]]*32)
        solver.seek(2)
        self.assertEqual(solver.stats.substeps,64)
        self.assertGreater(solver.stats.limited_count,0)
        self.assertTrue(np.isfinite(solver.snapshot().positions).all())
        a=self.make(lifetime=.1); b=self.make(lifetime=.1)
        for frame in range(1,21): a.seek(frame)
        b.seek(20)
        self.assertGreater(a.stats.removed_volume,0)
        self.assertGreater(int(a.snapshot().ids.min()),0)
        a.seek(2); a.seek(20)
        np.testing.assert_allclose(a.snapshot().positions,b.snapshot().positions,atol=1e-6)
