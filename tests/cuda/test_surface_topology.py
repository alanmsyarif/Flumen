import importlib
import unittest
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.state import ParticlePool,snapshot
from flumen.gpu.motion import advance


class TopologyTests(unittest.TestCase):
    def source(self,vertices,triangles,islands):
        cfg=FlowConfig(capacity=1,gravity=(0,0,-1),resistance=0,max_travel=.1)
        device=require_cuda()
        source=build_source(vertices,triangles,islands,cfg,device)
        self.addCleanup(source.close)
        return source,cfg,device

    def build(self,source,**kwargs):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.topology'),'Surface topology missing')
        topology=importlib.import_module('flumen.gpu.topology').build_topology(source,**kwargs)
        self.addCleanup(topology.close)
        return topology

    def test_global_face_ids_survive_island_projection_and_reattachment(self):
        source,cfg,device=self.source([[-2,0,-2],[-1,0,-2],[-2,0,-1],
                                     [0,0,0],[1,0,0],[0,0,1]],[[0,1,2],[3,4,5]],[0,1])
        pool=ParticlePool(cfg,device); self.addCleanup(pool.close)
        pool.data.active.assign([1]); pool.data.state.assign([0]); pool.data.island.assign([1])
        pool.data.position.assign([[.2,0,.3]]); pool.data.normal.assign([[0,-1,0]])
        pool.data.face.assign([1]); pool.data.volume.assign([1e-9])
        advance(pool,source,cfg,.01)
        self.assertEqual(int(pool.data.face.numpy()[0]),1)
        pool.data.position.assign([[.2,-.001,.3]]); pool.data.state.assign([1])
        pool.data.velocity.assign([[0,.2,0]])
        advance(pool,source,cfg,.01)
        self.assertEqual((int(pool.data.state.numpy()[0]),int(pool.data.face.numpy()[0])),(0,1))
        batch=snapshot(pool)
        self.assertEqual((int(batch.faces[0]),int(batch.islands[0])),(1,1))
        np.testing.assert_allclose(batch.normals,[[0,-1,0]],atol=1e-6)
        self.assertEqual(batch.velocities.shape,(1,3))
        self.assertAlmostEqual(float(batch.volumes[0]),1e-9,delta=1e-15)
        pool.data.active.zero_()
        empty=snapshot(pool)
        self.assertEqual((empty.normals.shape,empty.states.shape),((0,3),(0,)))

    def test_proxy_budget_and_area(self):
        source,_,_=self.source([[0,0,0],[1,0,0],[1,0,1],[0,0,1]],[[0,1,2],[0,2,3]],[0,0])
        a=self.build(source,support_radius=.2,target_edge=.1,vertex_budget=100)
        b=self.build(source,support_radius=.2,target_edge=.1,vertex_budget=100)
        self.assertLessEqual(len(a.vertices),100)
        self.assertGreater(a.coarsening_factor,1)
        np.testing.assert_array_equal(a.vertices,b.vertices)
        self.assertAlmostEqual(float(a.areas.sum()),1.,delta=.01)
        self.assertTrue((a.areas>0).all())
        np.testing.assert_allclose(np.linalg.norm(a.normals,axis=1),1,atol=1e-6)

    def test_topology_support_rejects_fold_shortcut(self):
        # Parallel patches are connected only through a distant strip.
        vertices=[[0,0,0],[1,0,0],[0,0,1],[1,0,1],
                  [0,.0001,0],[1,.0001,0],[0,.0001,1],[1,.0001,1],
                  [10,0,0],[10,.0001,0]]
        triangles=[[0,1,2],[1,3,2],[4,5,6],[5,7,6],[1,8,3],[5,7,9],[8,9,3],[9,7,3]]
        source,_,_=self.source(vertices,triangles,[0]*8)
        t=self.build(source,support_radius=.004,target_edge=20,vertex_budget=100)
        self.assertTrue(t.allows_faces(0,1))
        self.assertFalse(t.allows_faces(0,2))
        self.assertFalse(t.allows_faces(2,0))

    def test_mirrored_and_disconnected_support(self):
        source,_,_=self.source([[0,0,0],[-1,0,0],[0,0,1],
                               [0,.001,0],[-1,.001,0],[0,.001,1]],[[0,1,2],[3,4,5]],[0,1])
        t=self.build(source,support_radius=.1,target_edge=2)
        self.assertFalse(t.allows_faces(0,1))
        np.testing.assert_allclose(t.normals[:3],[[0,1,0]]*3,atol=1e-6)
