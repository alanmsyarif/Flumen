import importlib
import unittest
from math import pi
import numpy as np
from connected_fixtures import make_fixture
from flumen.gpu.config import FlowConfig
from test_surface_mesh import signed_volume


def components(mesh):
    _,ids=np.unique(np.round(mesh.vertices,8),axis=0,return_inverse=True)
    parent=np.arange(len(np.unique(ids)))
    def find(x):
        while parent[x]!=x: x=parent[x]
        return x
    for a,b,c in ids[mesh.triangles]:
        parent[find(b)]=find(a); parent[find(c)]=find(a)
    return len({find(x) for x in ids})


class FreeMeshTests(unittest.TestCase):
    def make(self,positions,velocities=None,barrier=False):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.free_mesh'),'Free liquid reconstruction missing')
        cfg=FlowConfig(capacity=len(positions),radius=.001,display_mode='CONNECTED')
        z=0. if barrier else -.1
        source,pool,device=make_fixture([[-1,-1,z],[1,-1,z],[1,1,z],[-1,1,z]],[[0,1,2],[0,2,3]],
            positions,states=[1]*len(positions),velocities=velocities,config=cfg)
        api=importlib.import_module('flumen.gpu.free_mesh')
        buffers=api.FreeMeshBuffers(pool.capacity,device.alias)
        for resource in (source,pool,buffers): self.addCleanup(resource.close)
        return cfg,source,pool,buffers,api

    def build(self,f,vertices=250000,triangles=500000):
        cfg,source,pool,buffers,api=f
        return api.build_free_mesh(pool,source,cfg,vertices,triangles,buffers)

    def test_isolated_closed_drop_and_current_neck(self):
        f=self.make([[.00017,.00013,.00019]])
        mesh=self.build(f)
        self.assertGreater(len(mesh.vertices),0)
        self.assertEqual(components(mesh),1)
        _,inverse=np.unique(np.round(mesh.vertices,8),axis=0,return_inverse=True)
        tri=inverse[mesh.triangles]
        edges=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        self.assertTrue((counts==2).all())
        self.assertGreater(signed_volume(mesh),0)
        self.assertLess(mesh.diagnostics['rendered_volume_error'],.25)
        np.testing.assert_allclose(np.linalg.norm(mesh.normals,axis=1),1,atol=1e-5)
        f=self.make([[0,0,0],[.0015,0,0]])
        self.assertEqual(components(self.build(f)),1)
        f[2].data.position.assign([[0,0,0],[.008,0,0]])
        self.assertEqual(components(self.build(f)),2)
        f[2].data.active.assign([1,0])
        mesh=self.build(f)
        self.assertEqual(components(mesh),1)
        self.assertLess(mesh.vertices[:,0].max(),.002)
        f[2].data.active.zero_()
        self.assertEqual(len(self.build(f).vertices),0)

    def test_bounded_anisotropy_and_sparse_far_cloud(self):
        f=self.make([[0,0,0],[10,10,10]],velocities=[[0,0,0],[0,0,100]])
        mesh=self.build(f)
        self.assertTrue(np.isfinite(mesh.vertices).all())
        self.assertLess(mesh.diagnostics['volumetric_samples'],20000)
        scales=f[4].anisotropy_scales(np.array([0.,1.,100.]),.001)
        self.assertTrue(np.isfinite(scales).all())
        np.testing.assert_allclose(scales[:,0]*scales[:,1]*scales[:,2],1,atol=1e-6)
        self.assertLessEqual(float((scales.max(axis=1)/scales.min(axis=1)).max()),4.00001)

    def test_sampling_output_caps_and_solid_barrier(self):
        f=self.make([[0,0,0],[0,0,0]])
        mesh=self.build(f,vertices=3,triangles=1)
        self.assertEqual(len(mesh.vertices),0)
        self.assertIn('budget',mesh.diagnostics['error'].lower())
        self.assertLessEqual(mesh.diagnostics['volumetric_samples'],2097152)
        mesh=self.build(f,vertices=1500,triangles=500)
        self.assertLessEqual(len(mesh.vertices),1500)
        self.assertLessEqual(len(mesh.triangles),500)
        f=self.make([[0,0,-.0008],[0,0,.0008]],barrier=True)
        mesh=self.build(f)
        self.assertGreater(len(mesh.vertices),0)
        self.assertEqual(components(mesh),2)
        centers=mesh.vertices[mesh.triangles].mean(axis=1)
        self.assertFalse((np.abs(centers[:,2])<.0001).any())

    def test_key_domain_is_explicit(self):
        f=self.make([[1e10,0,0]])
        mesh=self.build(f)
        self.assertEqual(len(mesh.vertices),0)
        self.assertIn('domain',mesh.diagnostics['error'].lower())

    def test_solver_assembles_shared_budget_and_caches_current_frame(self):
        from flumen.gpu.solver import FlowSolver
        from flumen.gpu.device import require_cuda
        from flumen.gpu.source import build_source
        cfg=FlowConfig(capacity=2,radius=.001,display_mode='CONNECTED',particles_per_frame=0)
        device=require_cuda()
        source=build_source([[0,0,0],[.01,0,0],[.01,.01,0],[0,.01,0]],[[0,1,2],[0,2,3]],[0,0],cfg,device)
        solver=FlowSolver(cfg,source,device); self.addCleanup(solver.close)
        solver.seek(1)
        d=solver.pool.data
        d.position.assign([[.005,.005,0],[.015,.005,.005]])
        d.active.assign([1,1]); d.state.assign([0,1]); d.normal.assign([[0,0,1]]*2)
        d.volume.assign([4*pi/3*.001**3]*2)
        from flumen.gpu.surface import update_surface
        update_surface(solver.surface,solver.pool,source,cfg,0)
        geometry=solver.water_snapshot()
        self.assertGreater(len(geometry.attached.vertices),0)
        self.assertGreater(len(geometry.free.vertices),0)
        self.assertLessEqual(len(geometry.attached.vertices)+len(geometry.free.vertices),250000)
        self.assertLessEqual(len(geometry.attached.triangles)+len(geometry.free.triangles),500000)
        self.assertIs(geometry,solver.water_snapshot())
        self.assertEqual(solver.stats.water_vertices,len(geometry.attached.vertices)+len(geometry.free.vertices))
        d.active.assign([1,0]); solver.dt=0; solver.seek(2)
        self.assertEqual(len(solver.water_snapshot().free.vertices),0)
