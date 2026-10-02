import unittest
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.solver import FlowSolver


class PointStreamTests(unittest.TestCase):
    def make(self, capacity=16):
        cfg=FlowConfig(capacity=capacity,particles_per_frame=0,initial_coating_count=0,
                       source_start=0,source_softness=0,display_mode='POINTS')
        device=require_cuda()
        source=build_source([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]],[0],cfg,device)
        solver=FlowSolver(cfg,source,device); self.addCleanup(solver.close)
        solver.seek(1)
        return solver

    def activate(self, solver, slots, states=None):
        d=solver.pool.data; n=solver.pool.capacity
        active=np.zeros(n,np.int32); active[slots]=1
        positions=np.zeros((n,3),np.float32); positions[:,0]=np.arange(n); positions[:,2]=1
        volumes=np.zeros(n,np.float32); volumes[slots]=(np.arange(len(slots))+1)*1e-9
        state=np.ones(n,np.int32)
        if states is not None: state[slots]=states
        normals=np.zeros((n,3),np.float32); normals[:,2]=1
        d.active.assign(active); d.position.assign(positions); d.volume.assign(volumes)
        d.state.assign(state); d.normal.assign(normals)
        return positions,volumes

    def test_point_stream_compacts_only_xyzr(self):
        solver=self.make()
        empty=solver.point_snapshot()
        self.assertEqual(empty.xyzr.shape,(0,4))
        self.assertEqual((empty.live_count,empty.displayed_count),(0,0))
        slots=[2,5,6,11]
        positions,volumes=self.activate(solver,slots,states=[1,0,1,1])
        frame=solver.current_frame; stats=solver.stats
        batch=solver.point_snapshot()
        self.assertEqual(batch.xyzr.dtype,np.float32)
        self.assertTrue(batch.xyzr.flags.c_contiguous)
        self.assertEqual(batch.xyzr.shape,(batch.displayed_count,4))
        self.assertEqual(batch.displayed_count,batch.live_count)
        self.assertEqual(batch.live_count,4)
        radii=np.cbrt(volumes[slots]*.238732414637843)
        expected=positions[slots].copy(); expected[1,2]+=radii[1]  # attached points sit on the surface
        np.testing.assert_allclose(batch.xyzr[:,:3],expected,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(batch.xyzr[:,3],radii,rtol=1e-5)
        self.assertEqual(solver.current_frame,frame)
        self.assertIs(solver.stats,stats)
        self.assertEqual(stats.displayed_count,4)
        self.assertGreaterEqual(stats.readback_ms,0.)
        # Point preview never reads identifiers or auxiliary snapshot fields.
        self.assertEqual(solver.pool.display_aux,{})
        self.assertEqual(len(solver.snapshot().ids),4)

    def test_optional_subset_is_deterministic(self):
        solver=self.make(64)
        slots=list(range(1,64,2)); self.activate(solver,slots)
        full=solver.point_snapshot().xyzr.copy()
        first=solver.point_snapshot(limit=5).xyzr.copy()
        second=solver.point_snapshot(limit=5).xyzr.copy()
        self.assertEqual(first.shape,(5,4))
        np.testing.assert_array_equal(first,second)
        for row in first: self.assertTrue((full==row).all(axis=1).any())
        self.assertEqual(solver.point_snapshot(limit=10**9).displayed_count,len(slots))
        with self.assertRaises(ValueError): solver.point_snapshot(limit=0)
