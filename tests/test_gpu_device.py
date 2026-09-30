import importlib
import subprocess
import sys
import pytest


def test_pure_import_does_not_require_warp():
    result = subprocess.run([sys.executable, '-c',
        "import flumen; import flumen.gpu.config; import sys; assert 'warp' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_native_library_has_actionable_error(monkeypatch):
    assert importlib.util.find_spec('flumen.gpu'), 'GPU package missing'
    device = importlib.import_module('flumen.gpu.device')
    def unavailable(name):
        raise OSError('native library unavailable')
    monkeypatch.setattr(device.importlib, 'import_module', unavailable)
    with pytest.raises(device.GPUUnavailableError, match='Warp'):
        device.require_cuda()
