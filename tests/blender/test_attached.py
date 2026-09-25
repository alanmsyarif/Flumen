import importlib.util
import math
import unittest
from helpers import reset_scene, mesh_object, vertical_plane, step_fixture, particle_at


class AttachedTests(unittest.TestCase):
    def setUp(self):
        reset_scene()

    def group(self):
        self.assertIsNotNone(importlib.util.find_spec('flumen.sim_attached'), 'Attached step missing')
        from flumen.sim_attached import build_attached_group
        return build_attached_group()

    def test_plane_step_matches_analytic_drag(self):
        from flumen.sim_math import attached_step
        plane = vertical_plane()
        for drag in [0, 1e-6, 1e-3, 5]:
            actual = step_fixture(self.group(), [particle_at((0,0,0))],plane,
                                  dt=.1, Resistance=drag, **{'Max Travel':.1})[0]
            p,v = attached_step((0,0,0),(0,0,0),(0,1,0),(0,0,-9.81),drag,.1)
            self.assertLess(math.dist(actual['position'],p),1e-4)
            self.assertLess(math.dist(actual['sf_velocity'],v),1e-4)
            self.assertAlmostEqual(actual['sf_age'], .1, places=6)

    def test_underside_adhesion_threshold(self):
        plane = mesh_object([(-2,-2,0),(-2,2,0),(2,2,0),(2,-2,0)],[(0,1,2,3)])
        for adhesion, expected in [(5,1),(15,0)]:
            point=particle_at((0,0,0),normal=(0,0,-1))
            actual=step_fixture(self.group(),[point],plane,dt=.01,**{'Adhesion Acceleration':adhesion})[0]
            self.assertEqual(actual['sf_state'],expected)
            if expected:
                self.assertLess(actual['position'][2],-0.0001)

    def test_open_rim_does_not_trap_slow_particle(self):
        plane = mesh_object([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)],[(0,1,2,3)])
        point=particle_at((.9999,0,0),normal=(0,0,1),velocity=(.02,0,0))
        actual=step_fixture(self.group(),[point],plane,dt=.02,Resistance=0.0)[0]
        self.assertEqual(actual['sf_state'],1)
        self.assertGreater(actual['position'][0],1)

    def test_zero_dt_and_empty_state(self):
        plane=vertical_plane()
        point=particle_at((0,0,.5),velocity=(0,0,-.3))
        point['sf_blocked']=True
        actual=step_fixture(self.group(),[point],plane,dt=0.0)[0]
        self.assertLess(math.dist(actual['position'],point['position']),1e-7)
        self.assertEqual(actual['sf_age'],0)
        self.assertTrue(actual['sf_blocked'])
        self.assertEqual(step_fixture(self.group(),[],plane,dt=.1),[])

    def test_travel_limit_is_reported(self):
        plane=vertical_plane()
        actual=step_fixture(self.group(),[particle_at((0,0,0),velocity=(0,0,-100))],plane,dt=.1)[0]
        self.assertLessEqual(math.dist(actual['position'],(0,0,0)),.00201)
        self.assertTrue(actual['sf_step_limited'])

    def test_nearby_island_is_not_used_as_continuation(self):
        plane=mesh_object([(-1,-1,0),(0,-1,0),(0,1,0),(-1,1,0),
                           (0,-1,-.0001),(1,-1,-.0001),(1,1,-.0001),(0,1,-.0001)],
                          [(0,1,2,3),(4,5,6,7)])
        point=particle_at((-.0001,0,0),velocity=(.1,0,0),normal=(0,0,1),island=0)
        actual=step_fixture(self.group(),[point],plane,dt=.01,Resistance=0.0)[0]
        self.assertEqual(actual['sf_state'],1)
        self.assertEqual(actual['sf_island'],0)
        self.assertGreater(actual['position'][2],0)

    def test_concave_fold_does_not_snap_to_opposing_sheet(self):
        fold=mesh_object([(-1,-1,0),(0,-1,0),(0,1,0),(-1,1,0),
                          (-1,-1,.001),(0,-1,.001),(0,1,.001),(-1,1,.001)],
                         [(0,1,2,3),(1,5,6,2),(5,4,7,6)])
        point=particle_at((-.0001,0,0),volume=1e-12,velocity=(.1,0,0),normal=(0,0,1))
        actual=step_fixture(self.group(),[point],fold,dt=.01,Resistance=0.0)[0]
        self.assertLess(actual['position'][2],.0005)
        self.assertTrue(all(math.isfinite(v) for v in actual['position']))
