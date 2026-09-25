"""Convenience script to add the repository directory to Blender's Python path.

Recommended for development: open Blender's Scripting workspace and run this file.
It registers the addon directly from the repo without copying files.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import flumen
try:
    flumen.unregister()
except Exception:
    pass
flumen.register()
print("Flumen dev addon registered from", ROOT)
