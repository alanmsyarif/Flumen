"""Actual-device unittest runner, using Blender's Python and NumPy."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '.gpu-deps'), str(ROOT / 'tests' / 'cuda')]
from flumen.gpu.device import require_cuda
require_cuda()  # Never silently skip this suite when CUDA is unavailable.
args = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'/'cuda'), pattern=args[0] if args else 'test_*.py')
if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
    raise RuntimeError('CUDA tests failed')
