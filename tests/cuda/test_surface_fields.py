import importlib
import unittest
import numpy as np
from dataclasses import replace
from flumen.gpu.config import FlowConfig
from flumen.gpu.topology import build_topology
from flumen.gpu.solver import FlowSolver
from connected_fixtures import make_fixture


class SurfaceFieldTests(unittest.TestCase):
    def make(self,positions=None):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.surface'),'Surface field implementation missing')
        cfg=FlowConfig(capacity=2,gravity=(0,0,-1),display_mode='CONNECTED',particles_per_frame=0)
        source,pool,device=make_fixture([[0,0,0],[.01,0,0],[.01,0,.01],[0,0,.01]],
            [[0,1,2],[0,2,3]],positions or [[.005,0,.005]],config=cfg)
        topology=build_topology(source,.004,.001)
        api=importlib.import_module('flumen.gpu.surface')
        surface=api.build_surface(topology,cfg,device)
        for resource in (source,pool,topology,surface): self.addCleanup(resource.close)
        return source,pool,cfg,topology,surface,api

    def update(self,f,dt=0):
        source,pool,cfg,t,s,api=f
        api.update_surface(s,pool,source,cfg,dt)
        return api.snapshot_surface(s)

    def test_normalized_thickness_volume(self):
        f=self.make([[.005,0,.005],[.007,0,.007]])
        f[1].data.state.assign([0,1])
        batch=self.update(f)
        volume=float(f[1].data.volume.numpy()[0])
        self.assertAlmostEqual(float(np.dot(batch.thickness,f[3].areas)),volume,delta=volume*.05)
        self.assertAlmostEqual(batch.represented_volume,volume,delta=volume*1e-5)
        self.assertEqual(batch.unrepresented_volume,0)
        f[1].data.position.assign([[.5,0,.5],[.007,0,.007]])
        batch=self.update(f)
        self.assertEqual(float(batch.thickness.sum()),0)
        self.assertAlmostEqual(batch.unrepresented_volume,volume,delta=volume*1e-5)

    def test_sheet_support_and_dry_regions(self):
        f=self.make([[.004,0,.005],[.006,0,.005]])
        batch=self.update(f)
        t=f[3]
        near=(np.linalg.norm(t.vertices-[.005,0,.005],axis=1)<.001)
        self.assertTrue((batch.thickness[near]>0).all())
        far=(np.linalg.norm(t.vertices-[.005,0,.005],axis=1)>.006)
        self.assertTrue((batch.thickness[far]==0).all())
        self.assertTrue(np.isfinite(batch.thickness).all())

    def test_back_face_and_fold_do_not_receive_liquid(self):
        f=self.make()
        f[1].data.normal.assign([[0,1,0]]*2)
        self.assertEqual(float(self.update(f).thickness.sum()),0)
        vertices=[[0,0,0],[.01,0,0],[0,0,.01],[.01,0,.01],
                  [0,.0001,0],[.01,.0001,0],[0,.0001,.01],[.01,.0001,.01],
                  [.1,0,0],[.1,.0001,0]]
        triangles=[[0,1,2],[1,3,2],[4,5,6],[5,7,6],[1,8,3],[5,7,9],[8,9,3],[9,7,3]]
        source,pool,device=make_fixture(vertices,triangles,[[.003,0,.003]],config=f[2])
        topology=build_topology(source,.004,.002)
        surface=f[5].build_surface(topology,f[2],device)
        for resource in (source,pool,topology,surface): self.addCleanup(resource.close)
        batch=self.update((source,pool,f[2],topology,surface,f[5]))
        other_side=(topology.vertex_faces==2)|(topology.vertex_faces==3)
        self.assertTrue((batch.thickness[other_side]==0).all())

    def test_wetness_seconds_and_replay(self):
        for scale,fps in [(.05,24),(.5,24),(1.,24),(1.,48)]:
            f=self.make()
            for _ in range(round(fps/scale)): batch=self.update(f,scale/fps)
            covered=batch.thickness>1e-5
            np.testing.assert_allclose(batch.wetness[covered],.99326205,atol=1e-5)
            before=batch.wetness.copy(); self.update(f,0)
            np.testing.assert_array_equal(self.update(f,0).wetness,before)
            f[1].data.active.zero_()
            batch=self.update(f,1)
            np.testing.assert_allclose(batch.wetness[covered],.8987409,atol=1e-5)

    def test_solver_replay_and_empty_updates_are_bounded(self):
        f=self.make()
        # The solver owns a separate source so test cleanup does not share ownership.
        from flumen.gpu.source import build_source
        from flumen.gpu.device import require_cuda
        cfg=replace(f[2],initial_coating_count=4,source_start=0,source_softness=0,capacity=8)
        source=build_source(f[0].vertices_cpu,f[0].triangles_cpu,[0,0],cfg,require_cuda())
        solver=FlowSolver(cfg,source,require_cuda()); self.addCleanup(solver.close)
        solver.seek(4); before=solver.surface_snapshot()
        solver.seek(4)
        np.testing.assert_array_equal(solver.surface_snapshot().wetness,before.wetness)
        solver.seek(1); solver.seek(4)
        np.testing.assert_allclose(solver.surface_snapshot().wetness,before.wetness,atol=1e-5)
        f[1].data.active.zero_()
        pointers=(f[4].thickness.ptr,f[4].wetness.ptr)
        for _ in range(10000): f[5].update_surface(f[4],f[1],f[0],f[2],0)
        self.assertEqual((f[4].thickness.ptr,f[4].wetness.ptr),pointers)
