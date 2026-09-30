# GPU Flow development setup

GPU Flow requires Blender 5.2 on Windows x64 and an NVIDIA CUDA device.
Warp 1.17.0 is pinned in `requirements-gpu.txt`; NumPy is supplied by Blender.
Existing Geometry Nodes workflows do not require Warp.

From the repository directory:

```powershell
python -m pip download --no-deps --require-hashes -r requirements-gpu.txt -d .gpu-wheels
python -m pip install --no-index --no-deps --target .gpu-deps .gpu-wheels/warp_lang-1.17.0-py3-none-win_amd64.whl
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/check_gpu_runtime.py
```

Development runners explicitly add `.gpu-deps` to Python's search path.
The add-on never downloads packages on registration. Distributed extensions
use bundled wheels. GPU initialization errors do not activate a CPU fallback.
