"""Lazy CUDA initialization; legacy workflows do not depend on Warp."""
from dataclasses import dataclass
import importlib


class GPUUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    alias: str
    name: str
    warp_version: str
    driver_version: object


def require_cuda() -> DeviceInfo:
    try:
        wp = importlib.import_module('warp')
        wp.init()
        devices = wp.get_cuda_devices()
        if not devices:
            raise RuntimeError('No CUDA device detected')
        device = devices[0]
        if not device.is_cuda:
            raise RuntimeError('CPU fallback is not supported')
        return DeviceInfo(device.alias, device.name, wp.__version__,
                          wp.get_cuda_driver_version())
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise GPUUnavailableError(
            f'GPU Flow needs NVIDIA CUDA and the packaged Warp dependency. '
            f'See docs/GPU_SETUP.md. Details: {exc}') from exc
