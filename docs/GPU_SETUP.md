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

## Build and install

```powershell
python scripts/build_extension.py artifacts/extension-stage-gpu --wheel-dir .gpu-wheels
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --command extension validate artifacts/extension-stage-gpu
python scripts/smoke_gpu_package.py artifacts/extension-stage-gpu
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --command extension build --source-dir artifacts/extension-stage-gpu --output-dir artifacts
```

Use a fresh staging directory each time. The builder verifies the wheel hash,
copies nested GPU modules, and includes dependency/license metadata. The parent
smoke runner removes its temporary dependencies after Blender exits, so Windows
has released the DLLs. No extra NumPy wheel is needed in Blender.

Install `artifacts/flumen-0.0.4.zip` with Blender Preferences > Get Extensions >
Install from Disk. This GPU package is Windows x64 only. Enable Flumen, open
`artifacts/Flumen_GPU_Demo.blend`, select the GPU Flow host, and play from frame 1.
The demo requires the extension; its GPU runtime is reconstructed rather than baked.

Keep collision geometry stationary and the host unparented at identity transforms.
Changing physics, emission, scene FPS/start, or source geometry requires Reset
GPU Flow. Material edits do not. Subframes and offline GPU rendering are not
supported in this increment; CUDA hosts are created with render visibility disabled.

The measured acceptance environment was Blender 5.2.0 / Python 3.13.13 with
Warp 1.17.0 on an RTX 5050. Setup/JIT can take longer than steady-state playback.
