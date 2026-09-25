from pathlib import Path
import math
import unittest
import tempfile
import bpy
from helpers import (reset_scene, mesh_object, particle_at, simulation_fixture,
                     step_frames, reset_simulation, read_points)


class BakeTests(unittest.TestCase):
    def setUp(self):
        reset_scene()
        self.directory=tempfile.TemporaryDirectory(prefix='flumen-bake-')
        self.addCleanup(self.directory.cleanup)
        bpy.context.scene.render.fps=24
        bpy.context.scene.frame_start=1
        bpy.context.scene.frame_end=12

    def test_bake_reload_and_random_frame_access(self):
        host=simulation_fixture([particle_at((0,0,1),state=1)],mesh_object([],[]),Gravity=(0,0,-1))
        second=simulation_fixture([particle_at((1,0,1),state=1,volume=2e-9)],mesh_object([],[]),Gravity=(0,0,-2))
        second.name='Second bake host'
        reference=[]; second_reference=[]
        for frame in range(1,13):
            bpy.context.scene.frame_set(frame)
            reference.append(read_points(host)); second_reference.append(read_points(second))
        self.assertLess(reference[-1][0]['position'][2],.99)
        reset_simulation(host)
        reset_simulation(second)
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        host.select_set(True)
        bpy.context.view_layer.objects.active=host
        host.name='Bake host'
        mod=host.modifiers[0]
        mod.bake_target='PACKED'
        bpy.context.preferences.filepaths.save_version=0
        target=Path(self.directory.name)/'bake-roundtrip.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(target))
        with bpy.context.temp_override(object=host,active_object=host,selected_objects=[host],selected_editable_objects=[host]):
            result=bpy.ops.object.simulation_nodes_cache_bake(selected=False)
        self.assertIn('FINISHED',result)
        second.modifiers[0].bake_target='PACKED'
        with bpy.context.temp_override(object=second,active_object=second,selected_objects=[second],selected_editable_objects=[second]):
            self.assertIn('FINISHED',bpy.ops.object.simulation_nodes_cache_bake(selected=False))
        # Random-frame reads after reopening prove that the state was persisted;
        # Blender 5.2 has no public NodesModifierBake.is_baked property.
        bpy.ops.wm.save_as_mainfile(filepath=str(target))
        bpy.ops.wm.open_mainfile(filepath=str(target))
        host=bpy.data.objects['Bake host']
        second=bpy.data.objects['Second bake host']
        for frame in (12,3,9,1,6):
            bpy.context.scene.frame_set(frame)
            actual=read_points(host)
            self.assertEqual(len(actual),1)
            self.assertLess(math.dist(actual[0]['position'],reference[frame-1][0]['position']),1e-5)
            self.assertEqual(actual[0]['sf_id'],reference[frame-1][0]['sf_id'])
            self.assertEqual(actual[0]['sf_state'],reference[frame-1][0]['sf_state'])
            self.assertAlmostEqual(actual[0]['sf_volume']/reference[frame-1][0]['sf_volume'],1,places=6)
            actual_second=read_points(second)
            self.assertLess(math.dist(actual_second[0]['position'],second_reference[frame-1][0]['position']),1e-5)
            self.assertAlmostEqual(actual_second[0]['sf_volume']/2e-9,1,places=6)
        # Presentation is downstream of stored state: material edits retain baked positions.
        group=next(n for n in host.modifiers[0].node_group.nodes if n.type=='GROUP')
        group.inputs['Flow Material'].default_value=bpy.data.materials.new('New presentation')
        bpy.context.scene.frame_set(9)
        self.assertLess(math.dist(read_points(host)[0]['position'],reference[8][0]['position']),1e-5)
