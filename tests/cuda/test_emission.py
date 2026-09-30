import importlib
import unittest
from dataclasses import replace
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source


class EmissionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.emission'))
        self.api = importlib.import_module('flumen.gpu.emission')
        self.state = importlib.import_module('flumen.gpu.state')
        self.cfg = FlowConfig(capacity=5,particles_per_frame=3,source_start=0,source_softness=0)
        self.device = require_cuda()
        self.source = build_source([[0,0,0],[1,0,0],[0,0,1]],[[0,1,2]],[0],self.cfg,self.device)
        self.pool = self.state.ParticlePool(self.cfg,self.device)
        self.addCleanup(self.pool.close)
        self.addCleanup(self.source.close)

    def test_inclusive_emission_schedule_and_zero(self):
        cfg = replace(self.cfg,emission_start=2,emission_end=4)
        self.assertEqual([self.api.requested_births(cfg,f) for f in range(1,6)],[0,3,3,3,0])
        cfg = replace(cfg,mode='BURST',burst_count=7)
        self.assertEqual([self.api.requested_births(cfg,f) for f in range(1,6)],[0,7,0,0,0])
        self.assertEqual(self.api.requested_births(replace(self.cfg,particles_per_frame=0),1),0)

    def test_capacity_recycling_unique_ids_and_volume(self):
        self.api.emit(self.pool,self.source,self.cfg,1)
        self.assertEqual(len(self.state.snapshot(self.pool).ids),3)
        self.api.emit(self.pool,self.source,self.cfg,2)
        s = self.state.read_stats(self.pool,2)
        self.assertEqual((s.live_count,s.requested,s.accepted,s.capacity_rejected),(5,6,5,1))
        ages = self.pool.data.age.numpy(); ages[:2] = 10
        self.pool.data.age.assign(ages)
        self.api.retire(self.pool,self.cfg)
        self.api.emit(self.pool,self.source,self.cfg,3)
        ids = self.state.snapshot(self.pool).ids
        self.assertEqual(len(set(ids)),5)
        self.assertTrue({2,3,4}.issubset(ids))
        self.assertTrue((ids[ids>4] > 4).all())
        s = self.state.read_stats(self.pool,3)
        self.assertAlmostEqual(s.emitted_volume, s.live_volume+s.removed_volume,delta=1e-15)

    def test_saturation_is_bounded(self):
        cfg = replace(self.cfg,particles_per_frame=2**31-1,emission_end=10001)
        before = self.pool.data.position.ptr
        for frame in range(1,10001):
            self.api.emit(self.pool,self.source,cfg,frame)
        self.assertEqual(self.pool.data.position.ptr,before)
        s = self.state.read_stats(self.pool,10000)
        self.assertEqual(s.live_count,5)
        self.assertEqual(s.accepted,5)
        self.assertEqual(s.requested,10000*(2**31-1))
