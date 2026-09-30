import unittest
from dataclasses import replace
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.solver import FlowSolver


class CoatingTests(unittest.TestCase):
    def make(self,**values):
        cfg=FlowConfig(capacity=32,particles_per_frame=3,source_start=0,source_softness=0,
                       initial_coating_count=5,gravity=(0,0,-1),**values)
        device=require_cuda()
        source=build_source([[0,0,0],[1,0,0],[0,0,1]],[[0,1,2]],[0],cfg,device)
        solver=FlowSolver(cfg,source,device,fps=30,fps_base=1.001)
        self.addCleanup(solver.close)
        return solver

    def test_coating_and_frame_births(self):
        solver=self.make()
        for frame,want in [(1,8),(2,11),(3,14)]:
            stats=solver.seek(frame)
            self.assertEqual((stats.accepted,stats.live_count),(want,want))
            self.assertAlmostEqual(stats.emitted_volume,stats.live_volume+stats.removed_volume,delta=1e-12)
        before=solver.snapshot()
        solver.seek(3)
        np.testing.assert_array_equal(solver.snapshot().positions,before.positions)
        solver.seek(0); solver.seek(3)
        np.testing.assert_array_equal(solver.snapshot().ids,before.ids)
        np.testing.assert_allclose(solver.snapshot().positions,before.positions,atol=1e-6)

    def test_time_scale_uses_physical_seconds(self):
        slow=self.make(time_scale=.5); normal=self.make()
        slow.seek(3); normal.seek(3)
        self.assertEqual(slow.stats.accepted,normal.stats.accepted)
        np.testing.assert_allclose(slow.pool.data.age.numpy()[:8],2*1.001/30*.5,atol=1e-6)
        slow.pool.data.state.assign([1]*32)
        normal.pool.data.state.assign([1]*32)
        slow.seek(5)
        np.testing.assert_allclose(slow.pool.data.age.numpy()[:8],normal.pool.data.age.numpy()[:8],atol=1e-6)

    def test_huge_coating_is_bounded_without_backlog(self):
        cfg=FlowConfig(capacity=5,initial_coating_count=10**9,particles_per_frame=0,
                       source_start=0,source_softness=0)
        device=require_cuda()
        source=build_source([[0,0,0],[1,0,0],[0,0,1]],[[0,1,2]],[0],cfg,device)
        solver=FlowSolver(cfg,source,device); self.addCleanup(solver.close)
        stats=solver.seek(1)
        self.assertEqual((stats.live_count,stats.capacity_rejected),(5,10**9-5))
        solver.seek(3)
        self.assertEqual(solver.stats.requested,10**9)
        self.assertEqual(solver.pool.data.position.shape[0],5)

    def test_scaled_freefall_at_equal_physical_time(self):
        from flumen.gpu.motion import advance
        for scale,frames in [(.5,60),(1.,30)]:
            solver=self.make(time_scale=scale)
            solver.seek(1)
            solver.pool.data.position.assign([[0,-.1,1]]*32)
            solver.pool.data.velocity.zero_()
            solver.pool.data.state.assign([1]*32)
            for _ in range(frames):
                advance(solver.pool,solver.source,solver.config,solver.dt)
            np.testing.assert_allclose(solver.pool.data.position.numpy()[:8,2],.4989995,atol=1e-4)
