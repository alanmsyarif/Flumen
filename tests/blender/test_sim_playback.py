import math
import unittest
import bpy
from helpers import (reset_scene, mesh_object, vertical_plane, particle_at, simulation_fixture,
                     step_frames, reset_simulation, read_diagnostics)


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        reset_scene()
        bpy.context.scene.render.fps=24

    def test_one_second_attached_motion_and_reset(self):
        host=simulation_fixture([particle_at((0,0,1))],vertical_plane(),Gravity=(0,0,-1),Resistance=0.0)
        frames=step_frames(host,1,25)
        p=frames[-1][0]
        self.assertLess(math.dist(p['position'],(0,0,.5)),1e-4)
        self.assertAlmostEqual(p['sf_age'],1.0,places=5)
        reset_simulation(host)
        second=step_frames(host,1,25)
        self.assertLess(math.dist(second[-1][0]['position'],p['position']),1e-6)

    def test_one_second_free_fall(self):
        host=simulation_fixture([particle_at((0,0,1),state=1)],mesh_object([],[]),Gravity=(0,0,-1))
        p=step_frames(host,1,25)[-1][0]
        self.assertLess(math.dist(p['position'],(0,0,.5)),1e-4)
        self.assertAlmostEqual(p['sf_velocity'][2],-1,places=4)

    def test_detachment_does_not_double_integrate_age(self):
        plane=mesh_object([(-2,-2,0),(-2,2,0),(2,2,0),(2,-2,0)],[(0,1,2,3)])
        host=simulation_fixture([particle_at((0,0,0),normal=(0,0,-1))],plane,**{'Adhesion Acceleration':5.0})
        p=step_frames(host,1,2)[-1][0]
        self.assertEqual(p['sf_state'],1)
        self.assertAlmostEqual(p['sf_age'],1/24,places=6)
        self.assertLess(p['position'][2],-0.001)

    def test_removed_volume_is_accounted_for_after_state_becomes_empty(self):
        host=simulation_fixture([particle_at((0,0,1),state=1)],mesh_object([],[]),Lifetime=.02)
        self.assertEqual(step_frames(host,1,3)[-1],[])
        d=read_diagnostics(host)
        self.assertAlmostEqual(d['sf_removed_volume']/1e-9,1,places=5)
        self.assertAlmostEqual(d['sf_initial_volume']/1e-9,1,places=5)
        self.assertEqual(d['sf_live_volume'],0)

    def test_frame_rate_and_two_hosts_are_independent(self):
        plane=vertical_plane()
        a=simulation_fixture([particle_at((0,0,1))],plane,Gravity=(0,0,-1),Resistance=0.0)
        b=simulation_fixture([particle_at((0,0,1))],plane,Gravity=(0,0,-2),Resistance=0.0)
        from helpers import read_points
        for frame in range(1,25):
            bpy.context.scene.frame_set(frame)
            pa,pb=read_points(a)[0],read_points(b)[0]
        self.assertLess(pb['position'][2],pa['position'][2]-.1)
        reset_simulation(a)
        bpy.context.scene.render.fps=48
        p=step_frames(a,1,49)[-1][0]
        self.assertLess(math.dist(p['position'],(0,0,.5)),1e-4)

    def test_state_control_change_after_reset_uses_new_gravity(self):
        host=simulation_fixture([particle_at((0,0,1),state=1)],mesh_object([],[]),Gravity=(0,0,-1))
        old=step_frames(host,1,25)[-1][0]
        group=next(n for n in host.modifiers[0].node_group.nodes if n.type=='GROUP')
        group.inputs['Gravity'].default_value=(0,0,-2)
        reset_simulation(host)
        new=step_frames(host,1,25)[-1][0]
        self.assertLess(math.dist(new['position'],(0,0,0)),1e-4)
        self.assertGreater(old['position'][2]-new['position'][2],.49)
