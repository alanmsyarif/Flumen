import unittest
from dataclasses import replace
import numpy as np
import warp as wp
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.prepared import prepare_source
from flumen.gpu.field_solver import FieldBuffers


class FieldDynamicsTests(unittest.TestCase):
    def make(self, vertices=None, config=None):
        cfg = config or FlowConfig(capacity=8, radius=.0001, resistance=0,
                                    surface_damping=0, field_viscosity=0)
        device = require_cuda()
        source = build_source(vertices or [[0,0,0],[.1,0,0],[0,.1,0]], [[0,1,2]], [0], cfg,device)
        prepared = prepare_source(source,'dynamics',.04,.02); self.addCleanup(prepared.release)
        buffers = FieldBuffers(prepared.chart,device.alias); self.addCleanup(buffers.close)
        buffers.volume.assign(prepared.chart.areas*.001)
        buffers.thickness.assign([.001]*len(prepared.chart.vertices))
        return prepared,buffers,cfg

    def evolve(self, prepared, buffers, cfg, dt):
        from flumen.gpu import field_solver
        self.assertTrue(hasattr(field_solver,'evolve_field'),'Field evolution is missing')
        return field_solver.evolve_field(prepared,buffers,cfg,dt)

    def test_transfer_momentum_and_flat_film(self):
        p,b,cfg = self.make()
        b.velocity.assign([[1,2,0]]*len(p.chart.vertices))
        volume = b.volume.numpy().copy()
        step = self.evolve(p,b,cfg,.01)
        np.testing.assert_allclose(b.velocity.numpy(),[[1,2,0]]*len(volume),atol=1e-6,rtol=0)
        np.testing.assert_array_equal(b.volume.numpy(),volume)
        self.assertLessEqual(abs(step.represented_volume-volume.sum()),volume.sum()*1e-5)
        self.assertEqual(step.unrepresented_volume,0.)
        from flumen.gpu.field_solver import sample_field
        height,velocity = sample_field(p,b,0,(.2,.3))
        self.assertAlmostEqual(height,.001,places=8)
        np.testing.assert_allclose(velocity,[1,2,0],atol=1e-6)

    def test_incline_drainage_and_viscous_decay(self):
        cfg=FlowConfig(capacity=8,resistance=5,gravity=(0,0,-1),surface_damping=0,
                       field_viscosity=0,surface_tension=0)
        p,b,cfg=self.make([[0,0,0],[.1,0,0],[0,0,.1]],cfg)
        for _ in range(10): self.evolve(p,b,cfg,.1)
        np.testing.assert_allclose(b.velocity.numpy()[:,2],-(1-np.exp(-5))/5,atol=1e-6)
        p,b,cfg=self.make(config=replace(cfg,gravity=(0,0,-9.81),resistance=0,field_viscosity=.001))
        velocities=np.zeros_like(p.chart.vertices); velocities[:,0]=(p.chart.vertices[:,0]>.04)
        b.velocity.assign(velocities)
        before=float(np.sum(p.chart.areas*np.sum(velocities**2,axis=1)))
        self.evolve(p,b,cfg,.01)
        after=float(np.sum(p.chart.areas*np.sum(b.velocity.numpy()**2,axis=1)))
        self.assertLess(after,before)

    def test_thin_film_wall_drag(self):
        # Lubrication wall shear: drag = resistance + 3 nu / h^2, so film fronts cannot race.
        cfg=FlowConfig(capacity=8,resistance=5,gravity=(0,0,-1),surface_damping=0,
                       field_viscosity=1e-6,surface_tension=0)
        for height in (.001,.0001):
            p,b,cfg=self.make([[0,0,0],[.1,0,0],[0,0,.1]],cfg)
            b.volume.assign(p.chart.areas*height); b.thickness.assign([height]*len(p.chart.vertices))
            for _ in range(10): self.evolve(p,b,cfg,.1)
            drag=5+3e-6/height**2
            np.testing.assert_allclose(b.velocity.numpy()[:,2],-(1-np.exp(-drag))/drag,rtol=1e-3)

    def test_capillary_response_and_substep_guard(self):
        p,b,cfg=self.make()
        height=np.full(len(p.chart.vertices),.001,np.float32)
        peak=int(np.argmin(np.sum((p.chart.vertices-[.025,.025,0])**2,axis=1)))
        height[peak]=.002; b.thickness.assign(height)
        cfg=replace(cfg,repulsion_acceleration=0)
        step=self.evolve(p,b,cfg,.001)
        radial=p.chart.vertices-p.chart.vertices[peak]
        self.assertGreater(float(np.sum(radial*b.velocity.numpy())),0.)
        self.assertTrue(np.isfinite(b.velocity.numpy()).all())
        # Thickness is frozen within an interval, so substeps only resolve the velocity ODE.
        self.assertEqual(step.substeps,cfg.minimum_substeps)
        speed=float(np.linalg.norm(b.velocity.numpy(),axis=1).max())
        large=self.evolve(p,b,cfg,1.)
        self.assertEqual(large.substeps,cfg.minimum_substeps)
        np.testing.assert_allclose(large.courant,speed/p.chart.operator_spacing,rtol=1e-5)
        self.assertTrue(np.isfinite(b.velocity.numpy()).all())
        before=b.velocity.numpy().copy(); wet=b.wetness.numpy().copy()
        height[0]=np.nan; b.thickness.assign(height)
        with self.assertRaisesRegex(RuntimeError,'nonfinite'): self.evolve(p,b,cfg,.001)
        np.testing.assert_array_equal(b.velocity.numpy(),before)
        np.testing.assert_array_equal(b.wetness.numpy(),wet)

    def test_wetness_physical_time(self):
        p,a,cfg=self.make()
        b=FieldBuffers(p.chart,p.source.device); self.addCleanup(b.close)
        b.volume.assign(a.volume.numpy()); b.thickness.assign(a.thickness.numpy())
        self.evolve(p,a,cfg,.2)
        self.evolve(p,b,cfg,.1); self.evolve(p,b,cfg,.1)
        np.testing.assert_allclose(a.wetness.numpy(),b.wetness.numpy(),atol=1e-6,rtol=0)
        np.testing.assert_allclose(a.wetness.numpy(),1-np.exp(-1),atol=1e-6)
        a.thickness.zero_(); b.thickness.zero_()
        self.evolve(p,a,cfg,.2); self.evolve(p,b,cfg,.1); self.evolve(p,b,cfg,.1)
        np.testing.assert_allclose(a.wetness.numpy(),b.wetness.numpy(),atol=1e-6,rtol=0)
        before=a.wetness.numpy().copy()
        self.evolve(p,a,cfg,0)
        np.testing.assert_array_equal(a.wetness.numpy(),before)

    def test_empty_nodes_do_not_invent_flow(self):
        p,b,cfg=self.make([[0,0,0],[.1,0,0],[0,0,.1]])
        b.thickness.zero_(); b.volume.zero_()
        self.evolve(p,b,cfg,.01)
        np.testing.assert_array_equal(b.velocity.numpy(),np.zeros_like(p.chart.vertices))

    def test_sliver_source_respects_requested_field_resolution(self):
        cfg=FlowConfig(capacity=8,resistance=0,surface_damping=0,field_viscosity=0)
        source=build_source([[0,0,0],[.1,0,0],[0,.00001,0]],[[0,1,2]],[0],cfg,require_cuda())
        p=prepare_source(source,'sliver',.001,.02); self.addCleanup(p.release)
        b=FieldBuffers(p.chart,source.device); self.addCleanup(b.close)
        b.thickness.assign([.00001]*len(p.chart.vertices))
        b.volume.assign(p.chart.areas*.00001)
        step=self.evolve(p,b,cfg,1/60)
        self.assertLessEqual(step.substeps,64)
        np.testing.assert_allclose(b.velocity.numpy(),0,atol=1e-6,rtol=0)
        self.assertLess(p.chart.min_edge,.00001)
        self.assertGreaterEqual(p.chart.operator_spacing,.0005)

    def test_graph_and_eager_evolution_match(self):
        p,a,cfg=self.make(config=FlowConfig(capacity=8,field_viscosity=.001,resistance=5,surface_damping=0))
        b=FieldBuffers(p.chart,p.source.device); self.addCleanup(b.close)
        b.volume.assign(a.volume.numpy()); b.thickness.assign(a.thickness.numpy())
        velocity=np.zeros_like(p.chart.vertices); velocity[:,0]=p.chart.vertices[:,0]*10
        a.velocity.assign(velocity); b.velocity.assign(velocity)
        from flumen.gpu.field_solver import evolve_field
        for _ in range(5):
            evolve_field(p,a,cfg,.01,use_graph=True)
            evolve_field(p,b,cfg,.01,use_graph=False)
        np.testing.assert_allclose(a.velocity.numpy(),b.velocity.numpy(),atol=1e-6,rtol=0)
        np.testing.assert_array_equal(a.volume.numpy(),b.volume.numpy())
        np.testing.assert_allclose(a.wetness.numpy(),b.wetness.numpy(),atol=1e-6,rtol=0)
