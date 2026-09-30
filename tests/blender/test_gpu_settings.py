import unittest
from dataclasses import fields
from types import SimpleNamespace
import bpy
import flumen
from flumen.gpu.config import FlowConfig
from flumen.gpu_properties import config_for, _bounds


class GPUSettingsTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        flumen.register()
        self.addCleanup(flumen.unregister)
        self.host=bpy.data.objects.new('Settings',None)
        bpy.context.scene.collection.objects.link(self.host)

    def test_float_ui_endpoints_round_trip(self):
        for field in fields(FlowConfig):
            if not isinstance(field.default,float):
                continue
            for endpoint in _bounds[field.name]:
                with self.subTest(setting=field.name,endpoint=endpoint):
                    setattr(self.host.flumen_gpu,field.name,endpoint)
                    self.assertEqual(getattr(config_for(self.host),field.name),endpoint)
            setattr(self.host.flumen_gpu,field.name,field.default)

    def test_invalid_values_are_not_clamped(self):
        for value in (.10001,0.,float('nan'),float('inf')):
            values={f.name:f.default for f in fields(FlowConfig)}
            values['radius']=value
            host=SimpleNamespace(flumen_gpu=SimpleNamespace(**values))
            with self.subTest(value=value), self.assertRaises(ValueError):
                config_for(host)
