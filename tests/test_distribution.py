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
        for path in Path('flumen').rglob('*.py'):
            assert (stage / path.relative_to('flumen')).read_bytes() == path.read_bytes()
        assert (stage / 'blender_manifest.toml').is_file()
        assert (stage / 'LICENSE').is_file()
        assert not list(stage.rglob('*.pyc'))


def test_gpu_build_rejects_missing_or_corrupted_wheel(tmp_path):
    import pytest
    spec=importlib.util.spec_from_file_location('build_extension',Path('scripts/build_extension.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    wheels=tmp_path/'wheels'; wheels.mkdir()
    with pytest.raises(FileNotFoundError):
        module.build_extension(tmp_path/'missing',wheel_dir=wheels)
    (wheels/'warp_lang-1.17.0-py3-none-win_amd64.whl').write_bytes(b'not the pinned wheel')
    with pytest.raises(ValueError,match='hash'):
        module.build_extension(tmp_path/'corrupt',wheel_dir=wheels)
