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
and its implementation plan are approved; implementation is in progress.
No million-particle real-time claim or bake implementation exists.

## Surface-field implementation evidence

Source-local charts retain global face/barycentric anchors and use normalized
three-node transfer even when radius is below field spacing. Uniform refinement
is capped at 100,000 nodes; nonmanifold/steep edges are explicit barriers.
Contact samples are capped at 2,097,152, retain face/island provenance and a
conservative interpolation error bound, and flag nearby competing sheets.
Deterministic segmented float64 sums preserve deposited volume and momentum.

The water field uses density 1,000 kg/m³, tangent gravity, height-pressure and
surface-tension gradients. Its physical graph Laplacian is
`sum_j(w_ij*(h_j-h_i))/A_i` with nonnegative symmetric weights. Viscosity uses
eight implicit Jacobi iterations per substep; resistance/damping use exponential
integration. Existing pressure/cohesion acceleration caps are reported when
activated. The stability bound includes tangent speed, gravity-height waves
and capillary waves on the effective operator spacing; more than 64 steps raises an
explicit error rather than skipping elapsed time. Particle state owns volume;
field volume remains a derived deposit. Dry nodes cannot invent flow. Wetness
uses exact physical-time exponential deposition/drying and zero-dt is inert.

Actual CUDA fixtures verify flat-film equilibrium, downhill analytic drag,
viscous energy decay, restoring capillary response, bounded step failure,
source locality, subspacing volume transfer and physical-time wetness. These
unit fixtures are numerical evidence, not million-particle viewport acceptance.

A first million-particle Suzanne probe exposed a 487-step demand from narrow
source triangles. Numerical gradient/Laplacian operators are now regularized at
half the requested field spacing (or the larger minimum chart edge), retaining
exact source geometry/anchors. Triangle gradient scaling preserves constant-field
equilibrium; nonnegative graph weights use the same minimum metric length.
The raw minimum source/chart edge and operator spacing are distinct diagnostics.
This intentionally limits numerical detail below the chosen field resolution;
finer offline meshing does not restore that simulated detail.

The first [FIELD attached probe](../artifacts/field-million-attached-probe.json)
(24.73 FPS) predates CUDA graph reuse. After graph reuse, 32-bit deposit sort
keys and chunked contact fallback, the same Suzanne fixture (one million live
particles, 2 mm field/contact spacing, resistance 60, two warmup plus ten
measured intervals) gives these solver-only results on the RTX 5050:

| Distribution | Solver-only FPS | Median ms | Notes |
|---|---|---|---|
| [attached](../artifacts/field-million-attached-graph.json) | 47.3 | 19.7 | 64 field steps per interval |
| [free](../artifacts/field-million-free-graph.json) | 61.8 | 16.1 | still pays 5.7 ms deposit, 2.4 ms field |
| [mixed](../artifacts/field-million-mixed-graph.json) | 48.1 | 19.5 | |
| [dense](../artifacts/field-million-dense-graph.json) | failed | | frame 2: field needs 231 steps, over the 64 bound |

All passing runs are finite with ledger relative error at most 6.5e-10. Stage
times are CUDA events summed per seek and add up to solver time (attached:
field 4.1, contact 2.3, aggregation 4.9, resampling 1.5, deposit 5.6 ms).
Tracked CUDA arrays use 582 MiB, excluding BVH and graph storage. The dense
fixture places all particles on one anchor; its explicit bounded failure is
recorded, not hidden. The attached distribution runs at the 64-step hard limit
after power-of-two rounding (36-54 required).

A [60-interval attached run](../artifacts/field-million-attached-long.json)
first failed at interval 28: Suzanne's 196 open-boundary faces (eye sockets,
ears) drip water into crevices whose ambiguous contact cells send every trapped
particle to exact BVH fallback, which grew past the 65536-entry queue. By user
ruling, fallback now runs in repeated 65536-entry chunks. The run passes at
41.5 FPS solver-only (p95 25.4 ms) with 170k fallback particles by the end.
A free-merge regression with opposing and zero normals stays finite.

The solver runs above do not draw. A short actual
[1080p point probe](../artifacts/point-viewport-probe.json)
([screenshot](../artifacts/point-viewport-probe.png)) runs the attached fixture in
Blender's GUI (OpenGL, solid shading, 2 px points): one million simulated and
one million drawn points at 30.6 FPS (median frame 32.6 ms, p95 34.3 ms) over
30 frames after 3 warmup frames. Medians: solver 19.8, point readback 1.5,
VBO upload 2.1, draw submit 2.4 and whole redraw 10.1 ms. Copies cost about
3.6 ms, so CUDA/graphics interop is not justified; the solver dominates. This
short probe was superseded by the full gate below.

### Million-particle 1080p viewport gate (Task 6)

`scripts/benchmark_particle_viewport.py` drives Blender's GUI at 1920x1080
(OpenGL, solid shading, 2 px points). Each measured draw advances exactly one
integer interval; a POST_PIXEL one-pixel readback waits for GPU completion.
`flumen/gpu/preview_report.py` validates every report: 120 warmup plus 600
measured draws, at least 1,000,000 live and displayed throughout, mean FPS from
wall time >= 30, p95 <= 33.3 ms, no skipped intervals, no unresolved contacts,
ledger <= 1e-5 and owned arrays distinct from whole-device VRAM.

Getting there needed three solver changes, each made by user ruling with
regression tests:

1. Drip release: attached water on downward-facing surfaces releases as drops
   when local film thickness times how much the surface faces down exceeds the
   capillary length (about 2.7 mm). Before this, adhesion 15 > g meant hanging
   water never released and pooled indefinitely.
2. Thin-film wall drag: field damping adds 3 nu / h^2, so near-dry film fronts
   no longer reach about 0.45 m/s under capillary forces.
3. Fixed field substeps: thickness is frozen within each interval, so the old
   wave-speed substep demand guarded a coupled scheme that does not exist. With
   8 substeps instead of about 55, a 720-frame run matched within run-to-run
   noise. Substeps now equal `minimum_substeps`; the interval Courant number is
   reported (max about 8.3) and only nonfinite field state aborts.

Deposit also now sorts only attached contributions (bit-identical results).

Final [attached](../artifacts/particle-viewport-attached.json),
[free](../artifacts/particle-viewport-free.json),
[mixed](../artifacts/particle-viewport-mixed.json) and
[dense](../artifacts/particle-viewport-dense.json) runs, RTX 5050, Blender 5.2.0,
Warp 1.17.0, with DaVinci Resolve closed:

| Distribution | Gate | Mean FPS | Median ms | p95 ms | Max ms | Ledger |
|---|---|---|---|---|---|---|
| attached | PASS | 34.3 | 28.8 | 31.2 | 41.5 | 1.6e-9 |
| free | PASS | 45.6 | 21.7 | 23.6 | 25.8 | 5.0e-10 |
| mixed | PASS | 42.1 | 23.5 | 25.3 | 26.8 | 5.7e-10 |
| dense | PASS | 47.2 | 21.0 | 23.0 | 26.0 | 3.0e-9 |

Attached medians: solver 15.6 (field 1.2, contact 2.3, aggregation 4.9,
resampling 1.4, deposit 5.5), readback 1.5, upload 2.2, draw submit 2.2 and
whole redraw 10.2 ms. Tracked CUDA arrays use 582 MiB; whole-device VRAM read
1.75-2.0 GiB. Attached has only about 2 ms of p95 margin. With Resolve running,
one attached run failed (p95 36.8 ms) with every stage, including viewport
redraw, uniformly 20-30% slower in bands; those runs are not counted. The
single-anchor dense blob exceeds the drip threshold and falls, so it stops
being clustered attached flow early. A separate 600-frame
[dense solver stress](../artifacts/field-million-dense-stress.json) runs at
93.6 FPS solver-only (p95 11.5 ms, ledger 4.5e-9).

### Drainage against the reference (Task 6): FAIL

`scripts/capture_field_drainage.py` with `examples/create_field_water_demo.py`
coats the top of Suzanne with one million particles emitted from the collision
surface itself (attached, zero initial velocity), then runs 180 frames (3 s
simulated). [Metrics](../artifacts/field-drainage.json) and stills
([f001](../artifacts/field-drainage-f001.png), [f180](../artifacts/field-drainage-f180.png)):

- the coating front moves down only 1.7 cm (mean 1.2 cm);
- thickness contrast rises from 0.33 to 0.84 and top-10% channel overlap across
  30 frames rises from 0.08 to 0.39, so concentration forms and persists;
- 903 particles grow above 10x initial volume; 150k attached-to-free events;
  absolute ledger error stays below 1.3e-15 m^3.

Compared with the reference bust frames, the front stays a uniform sheet with
short tongues instead of distinct rivulet fingers and long hanging drips, and
drains far too slowly. Recaptured drops on the chin also show a lattice-like
dot pattern. Likely cause: resistance 60 (chosen earlier for benchmark
stability) plus the new wall drag; a 0.1 mm film's terminal speed on a vertical
wall is about g/(60+300) = 2.7 cm/s. Visual tuning is unfinished; the passing
performance gates use these same settings.

### Particle cache and offline meshing (Tasks 7-8)

Explicit bakes write validated particle caches (`flumen/particle_cache.py`):
one SHA256-checked NumPy file per frame, about 88 MB per million particles.
`flumen/offline_mesher.py` meshes caches on the CPU without CUDA: a film shell
over a refined source lattice (conservative deposit through cached face/bary
anchors) plus tiled marching-tetrahedra drops with crack-free tile seams.
Baked playback and rendering run with CUDA initialization forbidden.

[One million-particle frame](../artifacts/offline-mesh-probe.json) meshes in
7.8 s at 1 mm film spacing (3.5M triangles, 84 MB, film volume error 0.5%).
Later optimizations kept output byte-identical (SHA256 of frames 1/90/170)
while a late drop-heavy frame went from 565 s to 93 s on one core, with
frame-parallel workers on top.

[180-frame clip](../artifacts/offline-clip.json): 250,000 particles with radius
scaled to keep the million-particle fixture's water volume, 0.1 mm drop grid and
2 mm film spacing. Mesh: 31 min on 6 workers (median 47.5 s, max 145 s per
frame), 8.3 GB; cache 4.0 GB; renders 4.6 min (Workbench) and 6.1 min (EEVEE).
Median film volume error 0.6%; at frame 180, 89% of free-drop volume meshed and
11% was below grid resolution (reported, not lost). Stills:
[opaque f180](../artifacts/offline-clip-opaque-f180.png),
[water f060](../artifacts/offline-clip-water-f060.png),
[water f180](../artifacts/offline-clip-water-f180.png); MP4s stay local.

Against the reference bust the clip still fails visually: the crown film is
smooth and connected and the front reaches the nose with one tongue, but
drainage is far too slow, no distinct rivulets or hanging drips form, and thin
film near the 10 um clip threshold renders speckled (noisy deposit with about
ten particles per 2 mm node).

### Look fixes (after Task 8)

[Drainage sweeps](../artifacts/drainage-sweep.json) showed resistance barely
matters for the fixture's thin coat (0.05-0.17 mm, dominated by 3 nu/h^2 wall
drag), while the reference pours millimetre-thick water. With eight times the
water (2x particle radius) at resistance 60 the front advances 2.9/5.0/9.7/11.7 cm
at frames 30/60/120/180 with persistent channels; no solver default changed, so
the speed gates above remain valid.

Offline meshing gained volume-conserving film smoothing, PCA (Yu & Turk)
neighbour drop kernels, a film thickness cap whose excess becomes pendant drops
(the nose spikes were film nodes with tiny lattice area holding up to 96 mm of
"thickness"), screen-resolution drop grids and an out-of-shot crop. The current
[clip](../artifacts/offline-clip.json) (250k particles, 8x water, PCA drops, 2 mm
cap, smoothing 20, 0.4 mm drops, 2 mm film) meshes 180 frames in 21 min on 8
workers. Frame 30 shows a connected sheet reaching the nose with tongues and
drips falling from brow and ears ([opaque f030](../artifacts/offline-clip-opaque-f030.png),
[water f090](../artifacts/offline-clip-water-f090.png)); late frames still break the
thin residual film into patches ([opaque f180](../artifacts/offline-clip-opaque-f180.png)).

### Screen-space water preview

An optional Water preview style (`flumen/gpu_water_screen.py`) draws particles as
sphere sprites, smooths depth with a narrow-range filter, shades films with the
collision surface's geometry and drops with filtered depth. Its
[1080p gate](../artifacts/particle-viewport-attached-water.json) passes: attached,
1M simulated and drawn, 33.3 FPS mean, p95 32.0 ms (Points: 34.3 / 31.2).
Offscreen framebuffers are rebuilt each draw: cached ones silently lost depth
testing in Blender 5.2.

Suites at that point: 109 Python, 80 CUDA and 74 Blender tests.

### Rivulets, streams and wet coat (look work, 2026-10-02)

- **Wet sheen (offline, cosmetic):** the film keeps a minimum thickness scaled by
  the cached wetness, so drained areas stay a continuous thin coat instead of
  breaking into patches. The added volume is reported as `sheen_volume` (0.35 uL
  versus 15.8 uL of film at frame 180).
- **Downhill film streaking:** this filter elongated thickness structure 2.7x in a
  test but made no visible difference, so it was removed.
- **Mesher drop trails:** rejected, because drops (median radius 0.12 mm, below the
  0.4 mm grid) would need about 48x their real volume to be visible. Instead, free
  vertices carry kernel-weighted particle velocity, and baked playback writes a
  `velocity` attribute. Cycles motion blur then streaks falling drops into streams
  (EEVEE ignores it).
- **Film z-fighting:** the film's inner face lay exactly on the source, and Cycles
  drew a maze pattern. The shell now floats 10 um off the surface
  ([Cycles still](../artifacts/offline-look-cycles-f030.png)).
- **Contact hysteresis (solver, opt-in):** dry nodes resist tangential acceleration
  up to (sigma/rho) dcos (1-w)/(h L), and held nodes do not wet.
  - The first version still let pinned nodes wet, which only produced uniform
    creep.
  - Sweep at 90 frames ([grid](../artifacts/rivulet-pinning-sweep.png): none, 0.05,
    0.1, 0.2): 0.1 splits the front into tongues; 0.2 holds the water.
  - At 0.07 over 180 frames, a central rivulet runs down the muzzle to the chin
    with dry stripes beside it ([frames 60/90/120/180](../artifacts/rivulet-pinning-h07.png)).
  - 1080p gates with 0.07 pass for all distributions: attached 35.0 FPS / p95
    30.6 ms, free 44.7 / 23.7, mixed 41.9 / 24.7, dense 47.5 / 21.8
    (`artifacts/particle-viewport-*-pinning.json`).
  - A full combined clip was started, then stopped during meshing at user request.
    No final clip with these settings exists.

Suites: 112 Python, 81 CUDA and 75 Blender tests pass. The staged 0.0.4 package
passes extension validation and both smoke runners, including FIELD bake,
cancellation and CUDA-free playback.

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
