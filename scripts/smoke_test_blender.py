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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    parser.add_argument('--package', type=Path, help='Test a staged extension instead of development source')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
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
    finally:
        package.unregister()
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.save).resolve()))


if __name__ == '__main__':
    main()
