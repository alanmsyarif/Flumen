# Connected water validation — 2026-10-01

This is an experimental connected-water increment, not a completed match to
the supplied SIGGRAPH paper/video. The visual and 30 FPS acceptance gates have
not passed. The older Drops benchmark does not establish Connected performance.

**Updated requirement, 2026-10-01:** the user permits final meshing to run
offline during a later bake. The active real-time target is particle simulation
and interactive preview at one million particles. The full-mesh measurements
below are historical diagnostics, not the revised acceptance gate. The saved
clips predate the large-particle collision/neighbor optimizations. A revised
[particle-first design](superpowers/specs/2026-10-01-million-particle-preview-design.md)
awaits review; no million-particle real-time claim or bake implementation exists.

## Fixture and reproduction

Blender 5.2.0 LTS / Python 3.13.13, Warp 1.17.0, RTX 5050 8 GiB, Windows x64.
The stationary Suzanne is 0.3055 × 0.1841 × 0.2198 m with 15,744 evaluated
triangles, 8,192 particle slots, 4,096 initial coating births and 64 births/frame.
Time scale is 0.5 at 30 FPS; radius is 1 mm and lifetime is four physical seconds.
Emission ends at frame 120 for the clips and frame 720 for the benchmarks.
The refined proxy has 31,658 vertices; its worst-edge coarsening factor is 6.29.
Exact controls and counts are saved in each JSON report.

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --factory-startup --python-exit-code 1 --python scripts/capture_connected_water.py -- --view WATER --frames 180 --output artifacts/connected-water.mp4
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --factory-startup --python-exit-code 1 --python scripts/capture_connected_water.py -- --view GEOMETRY --frames 180 --output artifacts/connected-geometry.mp4
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --factory-startup --python-exit-code 1 --python scripts/benchmark_connected_water.py -- --view WATER --output artifacts/connected-eevee.json
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --factory-startup --python-exit-code 1 --python scripts/benchmark_connected_water.py -- --view GEOMETRY --output artifacts/connected-solid.json
```

Run GPU jobs sequentially and leave the preview window unobscured and unmodified.
Each completed draw is checked at 1920×1080. The capture reads the final composed
Blender window; intermediate POST_PIXEL pixels were incorrect on this build.
Timing includes the next event-loop turn and DwmFlush. CUDA is synchronized.
Warmup/JIT/setup are excluded; benchmarks require 120 warmup plus 600 measured
sequential draws and never skip simulation intervals. Resizing invalidates a run.

## Visual evidence

Both six-second clips contain 180 captured frames at 1920×1080, encoded at 30 FPS.
These are playback clips, not proof that capture ran at 30 FPS. Selected stills
at frames 1/30/60/90/120/180 are in [connected-stills](../artifacts/connected-stills).
The supplied reference bust was compared qualitatively; different source
geometry is not a pixel match. The moving-hand reference remains unsupported.

| Approved criterion | Observed evidence | Result |
| --- | --- | --- |
| Connected coating drains | Raised patches on the crown at frame 1; liquid moves to the nose, cheeks and ear undersides by frames 90–180 | Partial: coarse perforated patches, rather than a smooth initial film |
| Persistent uneven channels | Nose and side-of-cheek paths remain at frames 120–180 after emission stops | Fail: paths are too broken and angular to match the reference |
| Merging and narrowing/breaking necks | 4,854 merges by frame 120 and 8,202 by frame 180; free droplets below the source | Partial: isolated CUDA fixtures prove live neck separation; the full clip does not clearly establish reference-quality breakup |
| Wetness persists and dries | Mean wetness rises to 0.8571 at frame 120, falls to 0.8201 at 180 while most crown liquid disappears; water view retains a glossy source | Pass for state persistence; appearance remains approximate |
| Geometry is connected in both views | Shared raised patches and underside accumulations are visible in opaque and transmitting water views | Partial: actual connected mesh, but still many disconnected patches |

Overall visual gate: **FAIL**. It is not a bead-only renderer or a retained
trajectory tube, but that does not by itself satisfy the reference appearance.
The water shader uses IOR 1.333 and roughness 0.05. Surface support and proxy
resolution still limit thin channel fidelity; the water view also exaggerates
patch boundaries. Film smoothing and volume-dependent drip detachment remain
unimplemented improvements, not claims about this package.

At frame 180 there are 3,415 live particles and 230,840 generated vertices,
without geometry budget errors. Reconstructed signed-volume error rises from
0.083% at frame 1 to 50.58% at frame 180 as liquid accumulates in curved/free
regions. The conserved particle ledger and normalized thickness field do not
make this extracted visual surface volume-accurate; that gap is explicit.

## Performance evidence

The first complete Eevee run is preserved in
[connected-eevee-baseline.json](../artifacts/connected-eevee-baseline.json):
3.42 FPS average, 292.25 ms mean, 341.38 ms p95 and 379.72 ms maximum.
It completed all 720 draws, with zero geometry errors and a final ledger relative
error of 3.81e-8. Mean solver cost was 96.07 ms, reconstruction 34.29 ms and full
frame update 240.67 ms. Peak process working set was 1,284 MiB; whole-device
VRAM was 4,134 MiB, including other applications.

Profiling identified Blender RNA mesh import as a major update cost. Shared
attached vertices and native smooth normals avoided triangle-soup output and
slow custom-normal import. The next measured correction uses uint32 buffers
for RNA UNSIGNED index fields, allowing direct bulk transfer. CUDA event timing
measures interaction work without changing positions or the ledger. Geometry
and field copies and Blender display updates are measured separately.

The interrupted optimization run is preserved in
[connected-eevee-incomplete.json](../artifacts/connected-eevee-incomplete.json).
It stopped after 188 total draws when the viewport changed to 1574×973; its
68 measured samples cannot pass acceptance.

The final complete Eevee run in
[connected-eevee.json](../artifacts/connected-eevee.json) achieved **3.59 FPS**,
278.32 ms mean, 309.67 ms p95 and 370.39 ms maximum. All 600 measured draws
completed at 1920×1080 after 120 warmup draws, with no skipped intervals or
geometry errors. Final live count was 6,600; all 50,176 requested births
(including coating) were accepted. Peak process working set was 1,263 MiB and
whole-device VRAM was 6,497 MiB. Real-time Connected gate: **FAIL**.

| Mean stage cost | Eevee ms |
| --- | ---: |
| Solver, synchronized CPU wall time | 91.79 |
| Interaction CUDA event time, within solver | 87.25 |
| Reconstruction, including geometry copies | 31.61 |
| Geometry and field transfer | 1.23 |
| Blender mesh/wetness update | 105.85 |
| Entire frame update | 230.92 |
| Draw, composition and event scheduling remainder | 47.40 |

The dominant costs remain interaction work and Blender mesh updates, both
individually exceeding the 33.3 ms frame budget. The full-frame p95 improved
from 341.38 to 309.67 ms after the buffer correction; this is not close to real
time. The maximum reconstructed-volume error during the measured run was
70.16%, although the conserved particle ledger remains balanced.

The solid measurement in
[connected-solid.json](../artifacts/connected-solid.json) was interrupted by a
viewport resize and is not a valid 600-draw acceptance run. It was not repeated
after the user removed real-time final meshing from the requirement.
Reconstruction timing includes its geometry transfers; interaction timing is
a subset of solver timing. Do not add nested measurements as disjoint costs.

## One-million-particle feasibility

The probe keeps **1,000,000 live particles**, omits all meshing and viewport
drawing, uses the same 15,744-triangle Suzanne, a 0.1 mm particle radius, time
scale 0.5 and no lifetime retirement during the run. It samples ten intervals
after two warmup intervals. Merging is disabled to maintain the live count;
cohesion, repulsion and damping remain active in the interaction case. This
short stress probe is not a 600-draw real-time acceptance result.

| Configuration | Original solver FPS | Optimized solver FPS |
| --- | ---: | ---: |
| Independent particles, minimum 8 substeps | 16.48 | 22.31 |
| Independent particles, minimum 1 substep | 21.10 | 25.05 |
| Interacting particles, minimum 1 substep | 0.076 | 0.146 |

Raw states/configurations are in
[million-particle-baseline.json](../artifacts/million-particle-baseline.json) and
[million-particle-feasibility.json](../artifacts/million-particle-feasibility.json).
The interaction case slows progressively as particles concentrate: its final
interval still takes about 20 seconds. Minimum substeps is a lower bound;
the solver can request more shared steps as motion/forces increase.

The optimized path projects onto the already-known attached triangle and uses
the existing BVH fallback across edges. A CUDA differential regression covers
movement, edge crossings and detachment against the original path. Spatial
hash resolution now scales from 64 to at most 256; occupied-neighbor loops and
insertion shifts no longer traverse unused entries, and mutual face checks exit
once established. Deterministic nearest selection and overflow counts are
verified across small/large grids, including distant aliased cells.

The storage cap of 64 neighbors **does not bound the number of candidates
searched** in a concentrated region. It also does not remove repeated per-step
collision and pair-query work. That is why separating meshing is necessary but
insufficient for the requested million-particle fluid interaction performance.

Reproduce an individual solver-only probe with the checked-in script:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/benchmark_particle_solver.py -- --capacity 1000000 --frames 10 --warmup 2 --minimum-substeps 1 --merge-distance 0 --output artifacts/particle-solver-independent.json
# Add --interactions for the dense interacting stress case.
```

Verification at this checkpoint: 66 pure Python, 54 actual CUDA and 59 Blender
tests passed. These establish numerical/lifecycle correctness, not the revised
million-particle frame-time target.

## Delivered preview

Install the packaged extension, open `artifacts/Flumen_Connected_Water_Demo.blend`,
select its GPU host and Reset GPU Flow, then play from frame 1. CUDA state is
reconstructed; this is not a portable bake. The clips are
`artifacts/connected-water.mp4` and `artifacts/connected-geometry.mp4`.
Legacy Drops and Geometry Nodes workflows remain available. GPU offline render
and moving/deforming collision surfaces remain unsupported.
