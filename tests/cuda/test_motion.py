import importlib
import unittest
from dataclasses import replace
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.state import ParticlePool, read_stats


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.motion'))
        self.advance = importlib.import_module('flumen.gpu.motion').advance
        self.cfg = FlowConfig(capacity=1,gravity=(0,0,-1),resistance=0,max_travel=.1,kill_height=-1000)
        self.device = require_cuda()
        self.source = build_source([[-10,0,-10],[10,0,-10],[0,0,10]],[[0,1,2]],[0],self.cfg,self.device)
        self.pool = ParticlePool(self.cfg,self.device)
        self.addCleanup(self.source.close); self.addCleanup(self.pool.close)
        self.pool.data.active.assign([1]); self.pool.data.volume.assign([1e-9])
        self.pool.data.position.assign([[0,-.1,1]]); self.pool.data.state.assign([1])
        self.pool.data.normal.assign([[0,-1,0]])

    def test_free_fall_age_and_frame_rates(self):
        for fps in (24,48):
            self.pool.data.position.assign([[0,-.1,1]])
            self.pool.data.velocity.zero_(); self.pool.data.age.zero_()
            for _ in range(fps): self.advance(self.pool,self.source,self.cfg,1/fps)
            np.testing.assert_allclose(self.pool.data.position.numpy()[0],[0,-.1,.5],atol=1e-4)
            np.testing.assert_allclose(self.pool.data.velocity.numpy()[0],[0,0,-1],atol=1e-4)
            self.assertAlmostEqual(float(self.pool.data.age.numpy()[0]),1,places=5)

    def test_attached_drag_and_zero_dt(self):
        self.pool.data.position.assign([[0,0,1]]); self.pool.data.state.zero_()
        cfg=replace(self.cfg,resistance=5)
        for _ in range(24): self.advance(self.pool,self.source,cfg,1/24)
        self.assertAlmostEqual(float(self.pool.data.position.numpy()[0,2]),1-(1/5-(1-np.exp(-5))/25),delta=1e-4)
        before=self.pool.data.position.numpy()
        self.advance(self.pool,self.source,cfg,0)
        np.testing.assert_array_equal(self.pool.data.position.numpy(),before)

    def test_anchor_projection_matches_bvh_across_edges_and_detachment(self):
        from flumen.gpu.solver import FlowSolver
        cfg=replace(self.cfg,capacity=128,initial_coating_count=128,particles_per_frame=0,
                    source_start=0,source_softness=0,minimum_substeps=1,max_travel=.002)
        vertices=[[0,0,0],[.03,0,0],[0,0,.03],[.03,0,.03],
                  [.03,.01,.05],[0,.01,.05]]
        triangles=[[0,1,2],[1,3,2],[2,3,4],[2,4,5]]
        solvers=[]
        for _ in range(2):
            source=build_source(vertices,triangles,[0]*4,cfg,self.device)
            solver=FlowSolver(cfg,source,self.device)
            self.addCleanup(solver.close); solver.seek(1); solvers.append(solver)
        fast,reference=solvers
        for _ in range(30):
            self.advance(fast.pool,fast.source,cfg,1/30,use_anchor_projection=True)
            self.advance(reference.pool,reference.source,cfg,1/30,use_anchor_projection=False)
            np.testing.assert_allclose(fast.pool.data.position.numpy(),reference.pool.data.position.numpy(),atol=1e-6)
            np.testing.assert_allclose(fast.pool.data.velocity.numpy(),reference.pool.data.velocity.numpy(),atol=1e-6)
            np.testing.assert_array_equal(fast.pool.data.state.numpy(),reference.pool.data.state.numpy())
        self.assertTrue((fast.pool.data.state.numpy()==1).any(),'Fixture must exercise detachment')
        np.testing.assert_array_equal(fast.pool.data.age.numpy(),reference.pool.data.age.numpy())

    def test_thin_wall_capture_and_fast_contact(self):
        for speed,captured in [(.2,True),(2.,False)]:
            self.pool.data.position.assign([[0,-.001,1]])
            self.pool.data.velocity.assign([[0,speed,0]])
            self.pool.data.state.assign([1])
            self.advance(self.pool,self.source,self.cfg,.02)
            self.assertLessEqual(float(self.pool.data.position.numpy()[0,1]),1e-6)
            self.assertEqual(int(self.pool.data.state.numpy()[0]),0 if captured else 1)

    def test_open_edge_detaches_without_doubling_age(self):
        self.pool.data.position.assign([[0,0,-9.999]])
        self.pool.data.velocity.assign([[0,0,-1]])
        self.pool.data.state.zero_()
        self.advance(self.pool,self.source,self.cfg,.05)
        self.assertEqual(int(self.pool.data.state.numpy()[0]),1)
        self.assertAlmostEqual(float(self.pool.data.age.numpy()[0]),.05,places=6)

    def test_retirement_and_travel_limit_are_accounted(self):
        self.pool.data.velocity.assign([[100,0,0]])
        cfg=replace(self.cfg,max_travel=.00001,lifetime=.01)
        self.advance(self.pool,self.source,cfg,.02)
        self.assertEqual(int(self.pool.data.active.numpy()[0]),0)
        self.assertAlmostEqual(read_stats(self.pool,1).removed_volume,1e-9,delta=1e-15)

    def test_nearby_island_is_not_used_as_continuation(self):
        source=build_source([[0,0,0],[1,0,0],[0,0,1],[-1,.0001,-1],[2,.0001,-1],[0,.0001,2]],
                            [[0,1,2],[3,4,5]],[0,1],self.cfg,self.device)
        self.addCleanup(source.close)
        self.pool.data.position.assign([[.5,0,.0001]])
        self.pool.data.velocity.assign([[0,0,-1]])
        self.pool.data.state.zero_()
        self.advance(self.pool,source,self.cfg,.01)
        self.assertEqual(int(self.pool.data.state.numpy()[0]),1)

    def test_underside_detaches_and_grazing_clearance_is_finite(self):
        source=build_source([[-2,-2,0],[-2,2,0],[2,2,0]],[[0,1,2]],[0],self.cfg,self.device)
        self.addCleanup(source.close)
        self.pool.data.position.assign([[0,0,0]])
        self.pool.data.normal.assign([[0,0,-1]])
        self.pool.data.state.zero_()
        self.advance(self.pool,source,replace(self.cfg,adhesion=0),.01)
        self.assertEqual(int(self.pool.data.state.numpy()[0]),1)
        self.pool.data.position.assign([[0,-.0001,1]])
        self.pool.data.velocity.assign([[1,0,0]])
        self.advance(self.pool,self.source,self.cfg,.01)
        p=self.pool.data.position.numpy()[0]
        self.assertTrue(np.isfinite(p).all())
        self.assertLess(p[1],-.0006)
