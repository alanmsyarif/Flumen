import importlib
import unittest
from dataclasses import replace
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.topology import build_topology
from connected_fixtures import make_fixture


class InteractionTests(unittest.TestCase):
    def make(self,positions,**values):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.interaction'),'Interaction implementation missing')
        cfg=FlowConfig(capacity=len(positions),gravity=(0,0,-1),**values)
        source,pool,device=make_fixture([[-1,0,-1],[1,0,-1],[0,0,1]],[[0,1,2]],positions,config=cfg)
        topology=build_topology(source,.004,2)
        neighbors=importlib.import_module('flumen.gpu.neighbors').NeighborBuffers(pool.capacity,device.alias)
        scratch=importlib.import_module('flumen.gpu.interaction').InteractionBuffers(pool.capacity,device.alias)
        for resource in (source,pool,topology,neighbors,scratch): self.addCleanup(resource.close)
        return source,pool,cfg,topology,neighbors,scratch

    def compute(self,fixture):
        source,pool,cfg,t,n,s=fixture
        importlib.import_module('flumen.gpu.neighbors').build_neighbors(pool,source,t,cfg,n)
        importlib.import_module('flumen.gpu.interaction').compute_forces(pool,cfg,n,s)
        return s.forces.numpy()

    def test_prior_state_symmetry(self):
        for distance,sign in [(.0005,-1),(.0015,1),(.003,1)]:
            f=self.make([[0,0,0],[distance,0,0]])
            before=f[1].data.velocity.numpy()
            forces=self.compute(f)
            self.assertGreater(float(forces[0,0])*sign,0)
            np.testing.assert_allclose(forces[0],-forces[1],atol=1e-6)
            self.assertLessEqual(float(np.linalg.norm(forces[0])),20.00001)
            self.assertEqual(float(forces[0,1]),0)
            np.testing.assert_array_equal(f[1].data.velocity.numpy(),before)

    def test_coincident_and_overfull_neighbors(self):
        positions=[[0,0,0]]+[[.00001*i,0,0] for i in range(70)]
        f=self.make(positions)
        forces=self.compute(f)
        self.assertTrue(np.isfinite(forces).all())
        self.assertEqual(int(f[4].counts.numpy()[0]),64)
        self.assertEqual(int(f[4].overflow.numpy()[0]),6)
        ids=f[1].data.ids.numpy()[f[4].indices.numpy()[0,:64]]
        np.testing.assert_array_equal(ids,np.arange(1,65))
        permutation=np.arange(71)[::-1]
        f[1].data.position.assign(np.asarray(positions)[permutation])
        f[1].data.ids.assign(permutation)
        self.compute(f)
        np.testing.assert_array_equal(f[1].data.ids.numpy()[f[4].indices.numpy()[70,:64]],ids)

    def test_grid_scaling_preserves_neighbors_and_overflow(self):
        from flumen.gpu.neighbors import NeighborBuffers,build_neighbors
        positions=[[.00001*i,0,0] for i in range(71)]
        positions += [[offset+.00001*i,0,0] for offset in (-.256,.256) for i in range(20)]
        fixture=self.make(positions)
        source,pool,cfg,topology,_,_=fixture
        outputs=[]
        for resolution in (64,256):
            neighbors=NeighborBuffers(pool.capacity,pool.device,grid_resolution=resolution)
            self.addCleanup(neighbors.close)
            build_neighbors(pool,source,topology,cfg,neighbors)
            outputs.append((neighbors.indices.numpy(),neighbors.counts.numpy(),neighbors.overflow.numpy()))
        for small,large in zip(*outputs): np.testing.assert_array_equal(small,large)

    def test_damping_and_zero_strengths(self):
        f=self.make([[0,0,0],[.002,0,0]],cohesion_acceleration=0,repulsion_acceleration=0)
        f[1].data.velocity.assign([[1,0,0],[-1,0,0]])
        forces=self.compute(f)
        self.assertLess(forces[0,0],0)
        self.assertGreater(forces[1,0],0)
        f=list(f); f[2]=replace(f[2],surface_damping=0)
        np.testing.assert_array_equal(self.compute(f),np.zeros((2,3)))

    def test_back_side_and_foreign_state_are_not_neighbors(self):
        f=self.make([[0,0,0],[.001,0,0],[.002,0,0]])
        f[1].data.normal.assign([[0,-1,0],[0,1,0],[0,-1,0]])
        f[1].data.state.assign([0,0,1])
        self.compute(f)
        self.assertEqual(int(f[4].counts.numpy()[0]),0)

    def test_same_island_fold_is_not_a_neighbor(self):
        f=list(self.make([[.2,0,.2],[.2,.0001,.2]]))
        vertices=[[0,0,0],[1,0,0],[0,0,1],[1,0,1],
                  [0,.0001,0],[1,.0001,0],[0,.0001,1],[1,.0001,1],
                  [10,0,0],[10,.0001,0]]
        triangles=[[0,1,2],[1,3,2],[4,5,6],[5,7,6],[1,8,3],[5,7,9],[8,9,3],[9,7,3]]
        from flumen.gpu.source import build_source
        source=build_source(vertices,triangles,[0]*8,f[2],require_cuda())
        topology=build_topology(source,.004,20)
        self.addCleanup(source.close); self.addCleanup(topology.close)
        f[0]=source; f[3]=topology
        f[1].data.face.assign([0,2])
        self.compute(f)
        np.testing.assert_array_equal(f[4].counts.numpy(),[0,0])
