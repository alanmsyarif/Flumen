# GPU Flow delivery — 2026-09-30

Approved increment 1 is implemented on `feat/gpu-flow`, forked from `main`
at `087fc2c`. Install `artifacts/flumen-0.0.3.zip`, enable the extension, open
`artifacts/Flumen_GPU_Demo.blend`, and play from frame 1. For your own mesh,
select it and use the Flumen GPU panel to create GPU Flow. Choose Continuous,
set Particles Per Frame, emission start/end, capacity and lifetime, then Reset
GPU Flow after changing simulation settings. Capacity limits concurrent live
particles; expired slots are recycled. Rejected births are not queued.

This delivers CUDA surface particles and detached drops with continuous emission.
Connected films, merging/cohesion, persistent rivulets, deforming geometry and
offline GPU bake/render remain later work; it does not yet reproduce the full
reference video's liquid appearance.

## Evidence

- 38 Python, 17 actual CUDA and 52 Blender tests passed on the final source.
- Final 1920×1080 solid viewport benchmark: 600 draws, 56.59 FPS average,
  16.70 ms median and 25.02 ms p95, passing the 30 FPS gate. No skipped frames.
  RTX 5050, Blender 5.2.0, Warp 1.17.0; this fixture has 8192 slots and
  64 births/frame. Performance varies with the scene and other GPU workloads.
- Verified pinned wheel hash, Blender manifest validation, and packaged
  CUDA/legacy smoke tests. Packaged CUDA test accepted all 320 requested births.
- Independent final review found one Important UI endpoint rounding defect,
  no Critical issues or deferred minors. Regression failed at seven endpoints
  before the fix and passed afterward. Reviewer did not independently rerun
  GPU performance or CUDA correctness; those are covered by execution evidence.
- Preview: `artifacts/GPU_Preview.png`; raw timing report:
  `artifacts/benchmark-gpu-viewport.json`.

## Execution rulings

1. CUDA tests run with unittest in Blender instead of a pytest CUDA plugin,
   avoiding another Blender dependency. Cost if unsuitable: adapt the runner.
2. Analytic drag uses absolute 0.0001 m position tolerance; float32 projection
   produced 0.0000573 m error. Cost: smaller numerical errors are accepted.
3. Independent particles use adaptive substeps within each GPU thread.
   Cost: revisit scheduling before adding particle coupling/cohesion.
4. Exact birth-count integration tests use a full source mask; narrow masks
   intentionally allow bounded sampling rejection. Cost: emission near a
   narrow source may be lower than the requested per-frame count.
5. GPU hosts are hidden from offline rendering and rendering suspends live
   updates with an explicit error. Cost: this GPU increment is viewport-only.

All eight implementation tasks and the single final review fix pass are complete.
No findings were deferred. Local source integration remains a user choice;
the installable ZIP and demo can be used independently of merging the branch.
