import unittest
import warp as wp
from flumen.gpu.device import require_cuda


@wp.kernel
def increment(a: wp.array(dtype=wp.int32)):
    a[wp.tid()] += 1


class DeviceTests(unittest.TestCase):
    def test_executes_on_cuda(self):
        info = require_cuda()
        a = wp.zeros(32, dtype=wp.int32, device=info.alias)
        wp.launch(increment, 32, inputs=[a], device=info.alias)
        wp.synchronize_device(info.alias)
        self.assertTrue(a.device.is_cuda)
        self.assertTrue((a.numpy() == 1).all())
