"""Run actual Blender graph tests without installing pytest in Blender."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'blender'))
parser = argparse.ArgumentParser()
parser.add_argument('--pattern', default='test_*.py')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests' / 'blender'), pattern=args.pattern)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise RuntimeError('Flumen Blender tests failed')
