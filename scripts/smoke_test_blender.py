"""Blender background smoke check; add -- --save PATH to keep the scene."""
import argparse
import importlib.util
from pathlib import Path
import sys
import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_output(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        assert len(mesh.vertices) > len(obj.data.vertices), 'No generated flow in render output'
        age = mesh.attributes.get('sf_age')
        assert age is not None and max(v.value for v in age.data) >= 8, 'Trail history missing'
        return len(mesh.vertices), len(mesh.edges), len(mesh.polygons)
    finally:
        evaluated.to_mesh_clear()


def validate_field_output(solver):
    """Live FIELD state must be nonempty, finite and volume-conservative."""
    s = solver.stats
    assert s.live_count > 0, 'FIELD solver has no live particles'
    assert s.emitted_volume > 0, 'FIELD solver emitted nothing'
    ledger = abs(s.emitted_volume-s.live_volume-s.removed_volume)/s.emitted_volume
    assert ledger <= 1e-4, f'FIELD volume ledger is off by {ledger:.2e}'
    pool = getattr(solver, 'pool', None)
    if pool is not None:
        import numpy as np
        assert np.isfinite(pool.data.position.numpy()).all(), 'FIELD positions are nonfinite'


def validate_baked_output(host):
    """A baked-water host must show nonempty, finite cached geometry."""
    assert host.get('sf_baked_frame') is not None, 'Object is not a baked water host'
    mesh = host.data
    assert len(mesh.polygons) > 0, 'Baked water frame is empty'
    import numpy as np
    co = np.empty(3*len(mesh.vertices), np.float32); mesh.vertices.foreach_get('co', co)
    assert np.isfinite(co).all(), 'Baked water vertices are nonfinite'


def smoke_field_bake(package, source, scene, scratch):
    """FIELD points preview, cancellable explicit bake, CUDA-free offline mesh playback, cleanup."""
    name = package.__name__
    runtime = __import__(name+'.gpu_runtime', fromlist=['x'])
    bake = __import__(name+'.gpu_bake', fromlist=['x'])
    mesher = __import__(name+'.offline_mesher', fromlist=['x'])
    cache = __import__(name+'.particle_cache', fromlist=['x'])
    baked = __import__(name+'.gpu_baked_display', fromlist=['x'])
    points = __import__(name+'.gpu_point_display', fromlist=['x'])
    device = __import__(name+'.gpu.device', fromlist=['x'])
    scene.frame_start, scene.frame_end = 1, 4; scene.frame_set(1)
    host = runtime.create_gpu_host(source, scene, display_mode='POINTS', solver_backend='FIELD')
    s = host.flumen_gpu
    s.source_start = 0; s.source_softness = 0; s.particles_per_frame = 64; s.field_spacing = .05; s.contact_spacing = .05
    runtime.reset_host(host)
    for frame in range(1, 5): scene.frame_set(frame)
    solver = runtime.get_runtime(host, scene)
    assert solver.prepared is not None, 'FIELD preparation missing'
    assert solver.stats.accepted == 4*64, 'Continuous FIELD births did not advance'
    validate_field_output(solver); live = solver.stats.live_count
    assert points.published_batch(host).displayed_count == live, 'Point batch is not full count'
    job = bake.BakeJob(host, scene, scratch/'cancelled'); job.step(); job.cancel()
    try:
        cache.CacheReader(scratch/'cancelled'); raise AssertionError('Cancelled cache validated')
    except ValueError:
        pass
    job = bake.BakeJob(host, scene, scratch/'cache')
    while not job.step(): pass
    reader = cache.CacheReader(scratch/'cache')
    mesher.mesh_cache_sequence(reader, mesher.MeshOptions(spacing=.02), scratch/'mesh', lambda: False)
    runtime.release_all()
    original = device.require_cuda
    def forbidden(*args, **kwargs): raise RuntimeError('Live CUDA used during baked playback')
    device.require_cuda = forbidden
    try:
        water = baked.create_baked_water(scratch/'cache', scratch/'mesh', scene, source=source)
        for frame in (1, 4, 2): scene.frame_set(frame)
        validate_baked_output(water)
    finally:
        device.require_cuda = original
    print('FLUMEN_FIELD_BAKE_SMOKE_TEST_OK', live, len(water.data.polygons))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    parser.add_argument('--package', type=Path, help='Test a staged extension instead of development source')
    parser.add_argument('--gpu', action='store_true', help='Require GPU playback using only staged wheels')
    parser.add_argument('--wheel-env', type=Path, help='Wheel environment managed by the parent smoke runner')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if args.gpu:
        if not args.package or not args.wheel_env:
            parser.error('Use scripts/smoke_gpu_package.py to manage the isolated GPU dependency environment')
        sys.path.insert(0, str(args.wheel_env))
    if args.package:
        spec = importlib.util.spec_from_file_location('sf_staged', args.package / '__init__.py',
                                                    submodule_search_locations=[str(args.package)])
        package = importlib.util.module_from_spec(spec)
        sys.modules['sf_staged'] = package
        spec.loader.exec_module(package)
        from sf_staged.build_nodes import build_flumen_group
    else:
        import flumen as package
        from flumen.build_nodes import build_flumen_group
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    obj = bpy.context.object
    tree = build_flumen_group(force_rebuild=True)
    mod = obj.modifiers.new('Flumen', 'NODES')
    mod.node_group = tree
    print('SURFACE_FLOW_SMOKE_TEST_OK', *validate_output(obj))
    obj.modifiers.remove(mod)
    package.register()
    try:
        assert bpy.ops.flumen.create_simulation() == {'FINISHED'}
        host=bpy.context.object
        for frame in range(1,6):
            bpy.context.scene.frame_set(frame)
            evaluated=host.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh=evaluated.to_mesh()
            try:
                assert len(mesh.vertices)>0, 'Animated render is empty'
                if frame==5:
                    assert max(v.value for v in mesh.attributes['sf_age'].data)>.1, 'Animated state did not advance'
            finally:
                evaluated.to_mesh_clear()
        print('SURFACE_FLOW_ANIMATED_SMOKE_TEST_OK')
        if args.gpu:
            runtime = __import__(package.__name__+'.gpu_runtime', fromlist=['create_gpu_host'])
            scene=bpy.context.scene
            scene.frame_set(scene.frame_start)
            gpu_host=runtime.create_gpu_host(obj,scene)
            gpu_host.flumen_gpu.source_start=0
            gpu_host.flumen_gpu.source_softness=0
            runtime.reset_host(gpu_host)
            for frame in range(1,6): scene.frame_set(frame)
            solver=runtime.get_runtime(gpu_host,scene)
            assert solver.device.alias.startswith('cuda:'), 'GPU smoke fell back to CPU'
            assert solver.stats.accepted==320, 'Continuous emission did not advance'
            assert len(gpu_host.data.vertices)>0, 'GPU display is empty'
            import warp
            assert Path(warp.__file__).is_relative_to(args.wheel_env), 'Smoke used an external Warp install'
            print('SURFACE_FLOW_GPU_SMOKE_TEST_OK',solver.device,solver.stats)
            import tempfile
            with tempfile.TemporaryDirectory(prefix='flumen-smoke-') as scratch:
                smoke_field_bake(package,obj,scene,Path(scratch))
                runtime.release_all()
    finally:
        package.unregister()
    if args.gpu:
        handlers = [h for h in bpy.app.handlers.frame_change_post if package.__name__ in getattr(h, '__module__', '')]
        assert not handlers, 'Flumen handlers survived unregister'
        print('FLUMEN_CLEANUP_SMOKE_TEST_OK')
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save).resolve()))


if __name__ == '__main__':
    main()
