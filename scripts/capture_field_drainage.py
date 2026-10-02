"""180-frame stationary drainage capture with quantitative motion diagnostics.

Run with Blender GUI (no --background):
blender --factory-startup --python scripts/capture_field_drainage.py -- --output artifacts/field-drainage.json
"""
from math import radians
from pathlib import Path
import argparse
import json
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.gpu-deps'),str(ROOT/'scripts'),str(ROOT/'examples')]
import bpy
import numpy as np
from mathutils import Euler
import flumen
from flumen import gpu_runtime as runtime
from connected_viewport import resize_region
from create_field_water_demo import create_field_demo

STILLS = (1, 30, 60, 90, 120, 180)


def node_metrics(solver, previous_channels):
    thickness = solver.field.thickness.numpy()
    wet = thickness > 1e-6
    values = thickness[wet]
    contrast = float(values.std()/values.mean()) if values.size else 0.
    channels = set(np.flatnonzero(wet & (thickness >= np.percentile(values, 90)))) if values.size else set()
    persistence = (len(channels & previous_channels)/len(channels | previous_channels)
                   if previous_channels is not None and channels | previous_channels else None)
    return dict(wet_nodes=int(wet.sum()), thickness_contrast=contrast,
                max_thickness_mm=float(thickness.max())*1e3, channel_persistence=persistence), channels


class Capture:
    def __init__(self, args):
        self.args = args; self.window = bpy.context.window_manager.windows[0]
        flumen.register()
        self.scene, self.host = create_field_demo(source_start=args.source_start)
        self.attempts = 0; self.records = []; self.previous_channels = None

    @property
    def area(self):
        return next(a for a in self.window.screen.areas if a.type == 'VIEW_3D')

    def region(self):
        return next(r for r in self.area.regions if r.type == 'WINDOW')

    def prepare(self):
        try:
            if self.attempts == 0:
                with bpy.context.temp_override(window=self.window, area=self.area):
                    bpy.ops.wm.window_fullscreen_toggle(); bpy.ops.screen.screen_full_area(use_hide_panels=True)
                space = self.area.spaces.active
                space.show_region_ui = space.show_region_toolbar = space.show_region_header = False
                view = space.region_3d; view.view_perspective = 'PERSP'
                view.view_location = (0, 0, .14); view.view_distance = .62
                view.view_rotation = Euler((radians(82), 0, radians(25))).to_quaternion()
            region = self.region()
            if (region.width, region.height) != (1920, 1080) and self.attempts < 4:
                resize_region(region.width, region.height); self.attempts += 1; return 1.
            self.run()
        except Exception:
            self.finish(traceback.format_exc())
        return None

    def run(self):
        solver = runtime.get_runtime(self.host, self.scene)
        pool = solver.pool
        start_z = None
        for frame in range(1, self.args.frames+1):
            before = pool.data.state.numpy().copy() if frame > 1 else None
            self.scene.frame_set(frame)
            if self.host.get('sf_gpu_error'): raise RuntimeError(self.host['sf_gpu_error'])
            state = pool.data.state.numpy(); active = pool.data.active.numpy() == 1
            z = pool.data.position.numpy()[:, 2]; volume = pool.data.volume.numpy()
            attached = active & (state == 0)
            if start_z is None: start_z = (float(np.mean(z[attached])), float(np.percentile(z[attached], 5)))
            detached = 0 if before is None else int(((before == 0) & (state == 1) & active).sum())
            metrics = None
            if frame == 1 or frame % 30 == 0:
                metrics, self.previous_channels = node_metrics(solver, self.previous_channels)
            record = dict(frame=frame, attached=int(attached.sum()), free=int((active & (state == 1)).sum()),
                          detach_events=detached,
                          mean_drop_m=start_z[0]-float(np.mean(z[attached])) if attached.any() else None,
                          front_drop_m=start_z[1]-float(np.percentile(z[attached], 5)) if attached.any() else None,
                          max_particle_volume_ul=float(volume[active].max())*1e9,
                          particles_above_10x_initial=int((volume[active] > 10*4.18879e-12).sum()),
                          ledger=solver.stats.emitted_volume-solver.stats.live_volume-solver.stats.removed_volume)
            if metrics: record.update(metrics)
            self.records.append(record)
            if frame in STILLS:
                self.redraw(); image = self.args.output.resolve().with_name(f'{self.args.output.stem}-f{frame:03d}.png')
                with bpy.context.temp_override(window=self.window, area=self.area, region=self.region()):
                    bpy.ops.screen.screenshot(filepath=str(image))
                if not image.exists(): raise RuntimeError(f'Still was not saved: {image}')
        self.finish()

    def redraw(self):
        with bpy.context.temp_override(window=self.window, area=self.area, region=self.region()):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

    def finish(self, error=None):
        output = self.args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(measurement_kind='field_drainage_capture', status='failed' if error else 'passed', error=error,
                      frames=self.args.frames, source_start=self.args.source_start, stills=list(STILLS),
                      notes='Diagnostics are numbers; points use one opaque color. Detach events count attached->free '
                            'state changes per interval. Channel persistence is Jaccard overlap of the top-10% '
                            'thickness nodes 30 frames apart.', records=self.records)
        output.write_text(json.dumps(report, indent=2), encoding='utf8')
        print('FIELD_DRAINAGE_RESULT', output, report['status'], flush=True)
        if error: print(error, flush=True)
        bpy.ops.wm.quit_blender()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--source-start', type=float, default=.55)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    bpy.context.preferences.view.show_splash = False
    def start():
        try:
            capture = Capture(args); bpy.app.timers.register(capture.prepare, first_interval=.5)
        except Exception:
            print('FIELD_DRAINAGE_RESULT failed', traceback.format_exc(), flush=True); bpy.ops.wm.quit_blender()
        return None
    bpy.app.timers.register(start, first_interval=1.)


if __name__ == '__main__': main()
