"""Run with Blender --background --python to prove actual CUDA execution."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.gpu-deps'))
from flumen.gpu.device import require_cuda
info = require_cuda()
import warp as wp


@wp.kernel
def increment(values: wp.array(dtype=wp.int32)):
    i = wp.tid()
    values[i] += 1


start = time.perf_counter()
values = wp.zeros(32, dtype=wp.int32, device=info.alias)
wp.launch(increment, 32, inputs=[values], device=info.alias)
wp.synchronize_device(info.alias)
assert (values.numpy() == 1).all()
assert values.device.is_cuda
print('CUDA VERIFIED', info, 'kernel seconds', time.perf_counter()-start, 'Python', sys.version)
