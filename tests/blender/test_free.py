import importlib.util
import math
import unittest
from helpers import reset_scene, mesh_object, vertical_plane, step_fixture, particle_at


class FreeTests(unittest.TestCase):
    def setUp(self):
        reset_scene()

    def group(self):
        self.assertIsNotNone(importlib.util.find_spec('flumen.sim_free'), 'Free step missing')
        from flumen.sim_free import build_free_group
        return build_free_group()

    def test_free_fall_and_velocity(self):
        empty=mesh_object([],[])
        actual=step_fixture(self.group(),[particle_at((0,0,1),state=1)],empty,dt=.1,**{'Max Travel':.1})[0]
        self.assertLess(math.dist(actual['position'],(0,0,.95095)),1e-5)
        self.assertLess(math.dist(actual['sf_velocity'],(0,0,-.981)),1e-5)

    def test_thin_wall_stops_crossing(self):
        wall=vertical_plane()
        point=particle_at((0,.01,0),velocity=(0,-1,0),state=1)
        actual=step_fixture(self.group(),[point],wall,dt=.02,Gravity=(0,0,0),**{'Max Travel':.1})[0]
        self.assertGreater(actual['position'][1],0)
        self.assertAlmostEqual(actual['sf_velocity'][1],0,places=5)

    def test_slow_inward_hit_reattaches_but_outward_does_not(self):
        wall=vertical_plane()
        point=particle_at((0,.001,0),velocity=(0,-.1,0),state=1)
        actual=step_fixture(self.group(),[point],wall,dt=.01,Gravity=(0,0,0))[0]
        self.assertEqual(actual['sf_state'],0)
        self.assertAlmostEqual(actual['position'][1],0,places=5)
        point=particle_at((0,.001,0),velocity=(0,.1,0),state=1)
        actual=step_fixture(self.group(),[point],wall,dt=.01,Gravity=(0,0,0))[0]
        self.assertEqual(actual['sf_state'],1)
        self.assertGreater(actual['position'][1],.001)

    def test_lifetime_and_kill_height_remove_particles(self):
        empty=mesh_object([],[])
        self.assertEqual(step_fixture(self.group(),[particle_at((0,0,0),state=1,age=1)],empty,dt=.1,Lifetime=.5),[])
        self.assertEqual(step_fixture(self.group(),[particle_at((0,0,-11),state=1)],empty,dt=.01),[])

    def test_zero_dt_does_not_change_particle(self):
        empty=mesh_object([],[])
        point=particle_at((0,0,1),state=1,velocity=(1,2,3))
        actual=step_fixture(self.group(),[point],empty,dt=0)[0]
        self.assertEqual(actual['position'],point['position'])
        self.assertEqual(actual['sf_velocity'],point['sf_velocity'])

    def test_oblique_graze_keeps_radius_clearance(self):
        plane=vertical_plane()
        point=particle_at((0,.001,0),velocity=(.5,-.05,0),state=1)
        actual=step_fixture(self.group(),[point],plane,dt=.02,Gravity=(0,0,0),**{'Max Travel':.1,'Capture Speed':0.0})[0]
        radius=(3e-9/(4*math.pi))**(1/3)
        self.assertGreaterEqual(actual['position'][1],radius-1e-6)
        self.assertGreater(actual['position'][0],0)
