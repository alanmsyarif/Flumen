"""Save a screenshot of the live GPU demo; run in a fresh GUI process."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'examples')]
import bpy
from create_gpu_demo import create_demo
scene,host=create_demo()
scene.frame_set(90)


def capture():
    bpy.ops.screen.screenshot(filepath=str(ROOT/'artifacts'/'GPU_Preview.png'))
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(capture,first_interval=3)
