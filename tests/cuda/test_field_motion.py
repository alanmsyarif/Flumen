import importlib.util
import unittest
from dataclasses import replace
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.prepared import prepare_source
from flumen.gpu.solver import FlowSolver


class FieldMotionTests(unittest.TestCase):
    def make(self, config=None, vertices=None, triangles=None):
        cfg=config or FlowConfig(solver_backend='FIELD',display_mode='POINTS',capacity=32,
            source_start=0,source_softness=0,particles_per_frame=0,initial_coating_count=0,
            field_spacing=.02,contact_spacing=.01,radius=.0001,resistance=0,surface_damping=0,
            field_viscosity=0,surface_tension=0,gravity=(0,0,-1),merge_distance_scale=0)
        device=require_cuda()
        vertices=vertices or [[0,0,0],[.1,0,0],[0,0,.1],[.1,0,.1]]
        triangles=triangles or [[0,1,2],[1,3,2]]
        source=build_source(vertices,triangles,[0]*len(triangles),cfg,device)
        prepared=prepare_source(source,'motion',cfg.field_spacing,cfg.contact_spacing)
        self.addCleanup(prepared.release)
        solver=FlowSolver(cfg,source,device,prepared=prepared); self.addCleanup(solver.close)
        self.assertIsNotNone(getattr(solver,'field',None),'Field particle backend is missing')
        self.assertIsNone(solver.interaction)
        self.assertIsNone(solver.geometry_buffers)
        return solver

    def set_particles(self, solver, positions, bary, faces=None, states=None, volumes=None, velocities=None):
        d=solver.pool.data; count=len(positions); n=solver.pool.capacity
        d.active.assign([1]*count+[0]*(n-count))
        d.position.assign(positions+[[0,0,0]]*(n-count))
        d.bary.assign(bary+[[0,0]]*(n-count))
        d.face.assign((faces or [0]*count)+[0]*(n-count))
        d.state.assign((states or [0]*count)+[0]*(n-count))
        d.normal.assign([[0,-1,0]]*n)
        d.volume.assign((volumes or [1e-12]*count)+[0]*(n-count))
        d.velocity.assign((velocities or [[0,0,0]]*count)+[[0,0,0]]*(n-count))
        d.ids.assign(list(range(n))); d.path.assign(list(range(n)))
        solver.pool.next_id=n
        solver.pool.ledger.assign([float(d.volume.numpy().sum(dtype=np.float64)),0])
        from flumen.gpu.field_solver import deposit_attached
        deposit_attached(solver.pool,solver.prepared,solver.field)

    def test_chart_walk_matches_exact_contact(self):
        solver=self.make(); solver.seek(1)
        self.set_particles(solver,[[.049,0,.049]],[[.02,.49]],velocities=[[.1,0,.1]])
        solver.seek(2)
        d=solver.pool.data
        self.assertEqual(int(d.face.numpy()[0]),1)
        self.assertEqual(int(d.state.numpy()[0]),0)
        self.assertAlmostEqual(float(d.position.numpy()[0,1]),0.,places=7)
        anchor=d.bary.numpy()[0]; face=int(d.face.numpy()[0])
        points=solver.source.vertices_cpu[solver.source.triangles_cpu[face]]
        expected=anchor[0]*points[0]+anchor[1]*points[1]+(1-anchor.sum())*points[2]
        np.testing.assert_allclose(d.position.numpy()[0],expected,atol=1e-6,rtol=0)

    def test_bounded_fallback_crosses_fine_triangle_strip(self):
        vertices=[]; triangles=[]
        for x in range(101): vertices.extend([[x*.001,0,0],[x*.001,0,.01]])
        for x in range(100):
            a=2*x; triangles.extend([[a,a+2,a+1],[a+2,a+3,a+1]])
        solver=self.make(vertices=vertices,triangles=triangles); solver.seek(1)
        self.set_particles(solver,[[.0002,0,.004]],[[.4,.2]],velocities=[[.6,0,0]])
        solver.seek(2)
        self.assertGreater(solver.stats.contact_fallback_count,0)
        self.assertAlmostEqual(float(solver.pool.data.position.numpy()[0,0]),.0202,delta=1e-6)
        self.assertEqual(int(solver.pool.data.state.numpy()[0]),0)

    def test_thin_sheet_sweep_and_chunked_fallback(self):
        solver=self.make(vertices=[[0,0,0],[.1,0,0],[0,.1,0]],triangles=[[0,1,2]])
        solver.seek(1)
        self.set_particles(solver,[[.02,.02,.01]],[[.6,.2]],states=[1],velocities=[[0,0,-1]])
        solver.seek(2)
        self.assertGreaterEqual(float(solver.pool.data.position.numpy()[0,2]),0.)
        cfg=replace(solver.config,capacity=65537,contact_spacing=.02)
        other=self.make(cfg,[[0,0,0],[.1,0,0],[0,.1,0],[0,0,.0001],[.1,0,.0001],[0,.1,.0001]],
                        [[0,1,2],[3,5,4]])
        other.seek(1)
        count=65537
        self.set_particles(other,[[.02,.02,.00005]]*count,[[.6,.2]]*count,states=[1]*count)
        other.seek(2)
        self.assertEqual(other.stats.contact_fallback_count,count)
        positions=other.pool.data.position.numpy()
        self.assertTrue(np.isfinite(positions).all())
        self.assertGreaterEqual(float(positions[:,2].min()),0.)
        # The final particle is in the second 65536 chunk; it must match the first exactly.
        np.testing.assert_array_equal(positions[count-1],positions[0])

    def test_dense_merge_and_resampling(self):
        cfg=FlowConfig(solver_backend='FIELD',display_mode='POINTS',capacity=10000,
            particles_per_frame=0,initial_coating_count=0,source_start=0,source_softness=0,
            merge_distance_scale=.25,field_spacing=.02,contact_spacing=.01)
        solver=self.make(cfg); solver.seek(1)
        n=cfg.capacity
        self.set_particles(solver,[[.02,0,.02]]*n,[[.6,.2]]*n,velocities=[[.1,0,.2]]*n)
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.field_aggregate'),'Bounded aggregation missing')
        from flumen.gpu.field_aggregate import aggregate_field_particles,resample_particles
        total=float(solver.pool.data.volume.numpy().sum(dtype=np.float64))
        aggregate_field_particles(solver.pool,solver.prepared,cfg)
        self.assertLess(int(solver.pool.data.active.numpy().sum()),n)
        resample_particles(solver.pool,solver.prepared,n)
        active=solver.pool.data.active.numpy()==1
        self.assertEqual(int(active.sum()),n)
        self.assertEqual(len(np.unique(solver.pool.data.ids.numpy()[active])),n)
        self.assertAlmostEqual(float(solver.pool.data.volume.numpy().sum(dtype=np.float64)),total,delta=total*1e-5)
        np.testing.assert_allclose(solver.pool.data.velocity.numpy()[active],[[.1,0,.2]]*n,atol=1e-6)
        self.assertEqual(float(solver.pool.ledger.numpy()[1]),0.)

    def test_field_replay_births_and_reuse(self):
        cfg=FlowConfig(solver_backend='FIELD',display_mode='POINTS',capacity=32,
            particles_per_frame=3,initial_coating_count=5,source_start=0,source_softness=0,
            field_spacing=.02,contact_spacing=.01,merge_distance_scale=0,lifetime=.1)
        a=self.make(cfg); b=self.make(cfg)
        counts=[]
        for f in range(1,4): counts.append(a.seek(f).accepted)
        self.assertEqual(counts,[8,11,14])
        a.seek(20); b.seek(20)
        np.testing.assert_allclose(a.snapshot().positions,b.snapshot().positions,atol=1e-6,rtol=0)
        self.assertGreater(a.stats.removed_volume,0)
        a.seek(2); a.seek(20)
        np.testing.assert_allclose(a.snapshot().positions,b.snapshot().positions,atol=1e-6,rtol=0)
        self.assertLessEqual(abs(a.stats.emitted_volume-a.stats.live_volume-a.stats.removed_volume),a.stats.emitted_volume*1e-5)
        old=a.stats.accepted; wet=a.field.wetness.numpy().copy(); a.seek(20)
        self.assertEqual(a.stats.accepted,old)
        np.testing.assert_array_equal(a.field.wetness.numpy(),wet)

    def test_reports_actual_attached_and_free_counts(self):
        solver=self.make(); solver.seek(1)
        self.set_particles(solver,[[.02,0,.02],[.2,.2,.2]],[[.6,.2],[.6,.2]],states=[0,1])
        stats=solver.seek(2)
        self.assertEqual(getattr(stats,'attached_count',None),1)
        self.assertEqual(getattr(stats,'free_count',None),1)

    def test_reports_event_stage_timings_and_owned_bytes(self):
        base=FlowConfig(solver_backend='FIELD',display_mode='POINTS',capacity=32,
            source_start=0,source_softness=0,particles_per_frame=0,initial_coating_count=4,
            field_spacing=.02,contact_spacing=.01,radius=.0001,resistance=0,surface_damping=0,
            field_viscosity=0,surface_tension=0,gravity=(0,0,-1),merge_distance_scale=.25,resample_target=8)
        solver=self.make(base); solver.seek(1); stats=solver.seek(3)
        stages=[getattr(stats,name,-1.) for name in ('field_ms','contact_ms','aggregation_ms','resample_ms','deposit_ms')]
        for value in stages: self.assertGreater(value,0.)
        self.assertLessEqual(sum(stages),stats.solver_ms*1.01+.05)
        owned=stats.owned_array_bytes
        self.assertGreaterEqual(owned,solver.pool.capacity*12)
        self.assertEqual(solver.seek(4).owned_array_bytes,owned)

    def test_free_merge_with_opposing_or_zero_normals_stays_finite(self):
        from flumen.gpu.field_aggregate import aggregate_field_particles
        cfg=replace(self.make().config,merge_distance_scale=.25)
        solver=self.make(cfg); solver.seek(1)
        point=[.05,.05,.5]
        self.set_particles(solver,[point]*4,[[.3,.3]]*4,states=[1]*4)
        solver.pool.data.normal.assign([[0,0,1],[0,0,-1],[0,0,0],[0,0,0]]+[[0,0,0]]*(solver.pool.capacity-4))
        solver.pool.data.island.assign([0]*solver.pool.capacity)
        aggregate_field_particles(solver.pool,solver.prepared,solver.config)
        d=solver.pool.data; active=d.active.numpy()==1
        self.assertEqual(int(active.sum()),2)
        for name in ('position','velocity','normal','bary','volume'):
            self.assertTrue(np.isfinite(getattr(d,name).numpy()[active]).all(),name)
        np.testing.assert_allclose(d.volume.numpy()[active].sum(),4e-12,rtol=1e-6)
