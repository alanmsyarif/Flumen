import importlib
import unittest
import numpy as np
from dataclasses import replace
from flumen.gpu.config import FlowConfig
from flumen.gpu.topology import build_topology
from flumen.gpu.neighbors import NeighborBuffers,build_neighbors
from flumen.gpu.state import read_stats
from connected_fixtures import make_fixture


class MergeTests(unittest.TestCase):
    def make(self,positions,volumes=None,velocities=None):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.merge'),'Merge implementation missing')
        cfg=FlowConfig(capacity=len(positions),gravity=(0,0,-1))
        source,pool,device=make_fixture([[-1,0,-1],[1,0,-1],[0,0,1]],[[0,1,2]],positions,
                                       volumes=volumes,velocities=velocities,config=cfg)
        topology=build_topology(source,.004,2)
        neighbors=NeighborBuffers(pool.capacity,device.alias)
        for x in (source,pool,topology,neighbors): self.addCleanup(x.close)
        return source,pool,cfg,topology,neighbors

    def merge(self,f):
        source,pool,cfg,topology,neighbors=f
        build_neighbors(pool,source,topology,cfg,neighbors)
        importlib.import_module('flumen.gpu.merge').merge_pairs(pool,source,topology,cfg,neighbors)

    def test_reciprocal_merge_volume_and_momentum(self):
        f=self.make([[0,0,0],[.0001,0,0]],[1e-9,3e-9],[[2,0,0],[-1,0,0]])
        f[1].data.age.assign([.4,.7])
        self.merge(f)
        pool=f[1]
        np.testing.assert_array_equal(pool.data.active.numpy(),[1,0])
        self.assertAlmostEqual(float(pool.data.volume.numpy()[0]),4e-9,delta=1e-15)
        np.testing.assert_allclose(pool.data.velocity.numpy()[0],[-.25,0,0],atol=1e-6)
        self.assertAlmostEqual(float(pool.data.age.numpy()[0]),.7,places=6)
        stats=read_stats(pool,1)
        self.assertEqual(stats.removed_volume,0)
        self.assertAlmostEqual(stats.live_volume,stats.emitted_volume,delta=1e-15)

    def test_nonreciprocal_chain_and_id_ownership(self):
        f=self.make([[-.00015,0,0],[0,0,0],[.00015,0,0]])
        self.merge(f)
        np.testing.assert_array_equal(f[1].data.active.numpy(),[1,0,1])
        f=self.make([[0,0,0],[.0001,0,0]])
        f[1].data.ids.assign([7,2])
        self.merge(f)
        np.testing.assert_array_equal(f[1].data.active.numpy(),[0,1])
        self.assertEqual(int(f[1].data.ids.numpy()[1]),2)

    def test_state_mismatch_and_oversized_pair_stay_separate(self):
        f=self.make([[0,0,0],[.0001,0,0]])
        f[1].data.state.assign([0,1])
        self.merge(f)
        self.assertEqual(int(f[1].data.active.numpy().sum()),2)
        f=list(self.make([[0,0,0],[.0001,0,0]]))
        f[2]=replace(f[2],maximum_merged_radius_scale=1)
        self.merge(f)
        self.assertEqual(int(f[1].data.active.numpy().sum()),2)
