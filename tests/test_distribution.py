from pathlib import Path
import importlib.util
import tempfile


def test_staged_extension_contains_exact_canonical_modules():
    script = Path('scripts/build_extension.py')
    assert script.is_file(), 'Canonical extension builder missing'
    spec = importlib.util.spec_from_file_location('build_extension', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        stage = module.build_extension(Path(directory) / 'stage')
        for path in Path('flumen').glob('*.py'):
            assert (stage / path.name).read_bytes() == path.read_bytes()
        assert (stage / 'blender_manifest.toml').is_file()
        assert (stage / 'LICENSE').is_file()
        assert not list(stage.rglob('*.pyc'))
