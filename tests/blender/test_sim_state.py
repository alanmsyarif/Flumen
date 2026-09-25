import importlib.util
import unittest
import bpy
from helpers import reset_scene, debug_modifier, read_points, mesh_object


def simulation_api(test):
    test.assertIsNotNone(importlib.util.find_spec('flumen.build_simulation'), 'Simulation builder missing')
    from flumen import build_simulation
    return build_simulation


class SimStateTests(unittest.TestCase):
    def setUp(self):
        reset_scene()

    def host(self, source, **settings):
        api = simulation_api(self)
        host = mesh_object([], [], 'Flow host')
        tree = api.build_simulation_group()
        mod, group = debug_modifier(host, tree, 'Particles', {'Collision Object': source, **settings})
        return host

    def test_one_time_seeds_ids_and_budget(self):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
        source = bpy.context.object
        host = self.host(source, **{'Particle Budget': 32, 'Seed Density': 1000.0, 'Gravity': (0,0,-1)})
        bpy.context.scene.frame_set(1)
        first = read_points(host)
        self.assertGreater(len(first), 0)
        self.assertLessEqual(len(first), 32)
        self.assertEqual(len({p['sf_id'] for p in first}), len(first))
        self.assertTrue(all(p['sf_volume'] > 0 for p in first))
        self.assertTrue(all(p['sf_age'] == 0 for p in first))
        for frame in range(2, 5):
            bpy.context.scene.frame_set(frame)
            points = read_points(host)
        self.assertEqual({p['sf_id'] for p in first}, {p['sf_id'] for p in points})
        from helpers import reset_simulation
        reset_simulation(host)
        replay=read_points(host)
        self.assertEqual([p['sf_id'] for p in replay],[p['sf_id'] for p in first])
        for a,b in zip(first,replay):
            self.assertLess(sum((x-y)**2 for x,y in zip(a['position'],b['position'])),1e-12)

    def test_empty_and_zero_density(self):
        source = mesh_object([], [])
        self.assertEqual(read_points(self.host(source)), [])
        bpy.ops.mesh.primitive_uv_sphere_add()
        self.assertEqual(read_points(self.host(bpy.context.object, **{'Seed Density': 0.0})), [])

    def test_source_allocation_is_bounded_before_truncation(self):
        from flumen.sim_source import build_source_group
        source=mesh_object([(-5,-5,0),(5,-5,0),(5,5,10),(-5,5,10)],[(0,1,2),(0,2,3)])
        tree=build_source_group()
        distribution=next(n for n in tree.nodes if n.bl_idname=='GeometryNodeDistributePointsOnFaces')
        output=next(n for n in tree.nodes if n.type=='GROUP_OUTPUT')
        tree.links.new(distribution.outputs['Points'],output.inputs['Particles'])
        mod,group=debug_modifier(source,tree,'Particles',{'Particle Budget':32,'Seed Density':1000,
            'Source Start':.9,'Source Softness':0})
        gi=next(n for n in mod.node_group.nodes if n.type=='GROUP_INPUT')
        mod.node_group.links.new(gi.outputs['Geometry'],group.inputs['Collision'])
        self.assertLess(len(read_points(source)),256,'Intermediate seeds greatly exceed the budget')

    def test_transformed_source_positions_are_world_coordinates(self):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12)
        source = bpy.context.object
        source.location = (4,0,2)
        source.scale = (2,1,1)
        host = self.host(source)
        points = read_points(host)
        self.assertGreater(len(points), 0)
        self.assertTrue(all(2 <= p['position'][0] <= 6 for p in points))
        self.assertTrue(all(2.2 < p['position'][2] <= 3.01 for p in points))
