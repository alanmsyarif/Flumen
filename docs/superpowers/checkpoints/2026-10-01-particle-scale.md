# Saved checkpoint — million-particle field work

**Paused at the user's explicit request on 2026-10-01. Resume only when requested.**

The revised [design](../specs/2026-10-01-million-particle-preview-design.md) and
[plan](../plans/2026-10-01-million-particle-preview.md) are both approved. Native
sequential execution is selected, with one fresh independent final review.

## Current saved implementation

- Revised Task 1 complete: bounded chart/contact preparation, conservative
  three-node transfer, global face provenance and mirrored-winding correction
  (`5ed088b`).
- Revised Task 2 complete: bounded field gravity/pressure/capillary dynamics,
  implicit viscosity, physical-time wetness and block-reduced stats (`d747231`).
- Revised Task 3 in progress: FIELD transport with staged contact validation,
  bounded aggregation/resampling and benchmark distributions are implemented.
  FIELD allocates no legacy particle-neighbor arrays or live meshing buffers.
- Latest changes reuse captured field operations and the previous completed
  interval's deposit. Attached/free counts are reported. Captured/eager evolution
  agrees in regression tests; latest performance has not been measured.

Checkpoint verification: **68 Python, 73 CUDA and 61 Blender tests passed**.
Focused dynamics (7) and motion (6) cover graph/eager agreement, fine-triangle
crossings, thin-sheet collision, overfull-contact rollback, dense conservation,
births, replay and distribution counts. No independent final review has run.

The retained [million-particle attached probe](../../../artifacts/field-million-attached-probe.json)
predates graph reuse and duplicate-transfer removal: two warmup plus ten measured
solver-only intervals, one million live particles, **24.73 FPS**, finite state and
ledger relative error **6.49e-10**. It uses field/contact spacing 2 mm, resistance
60, nominal radius 0.1 mm and time scale 0.5 on the 15,744-triangle Suzanne.
Default 1 mm field spacing had exceeded the 64-substep bound after regularization.
This short probe misses the target and proves no viewport acceptance.

Numerical rulings: derivative operators use at least half requested field spacing
while retaining exact source geometry/anchors; motion detail below that spacing
is suppressed. Required substeps round upward to powers of two, with a hard limit
of 64. Main attached walks cross at most 8 source edges; queued walks add at most
32 before narrowly eligible exact fallback. Overfull/unresolved contacts reject
the proposed interval. Resampling retains 99% parent volume and gives 1% to a
child at the identical anchor/velocity; no emitted mass is added. Uneven marker
weights require further evaluation.

## Resume at revised Task 3

1. Read this checkpoint, approved spec/plan and retained native ledger. Do not
   redo revised Tasks 1–2 or original connected-water Tasks 1–8.
2. Measure graph/deposit optimizations with the same million-particle attached
   fixture; characterize free, mixed and dense distributions separately. Save
   explicit benchmark failures and last valid states, not only successful runs.
3. Finish accurate GPU stage timings and owned-array memory reporting. Existing
   field/contact values are wall diagnostics, aggregation records launch time,
   and owned-array bytes are not populated. Separate tracked allocations from
   BVH/graph storage and whole-device VRAM.
4. Investigate the growing contact ambiguity/fallback count before long runs.
   Profile empty-field execution and compact attached-only transfer before
   optimizing; preserve thin/folded-sheet provenance and bounded failures.
5. Check opposing/zero normals during free aggregation; add a failing regression
   before fixing any demonstrated nonfinite result.
6. Complete Task 3 validation/ledger, then Tasks 4–9: compact direct point draw,
   physical/cosmetic signatures and preparation reuse, full-count viewport and
   reference gates, immutable particle cache, CPU offline mesher/CUDA-free
   playback, final independent review, docs and package checks.

Scratch is retained at `.superpowers/sdd/2026-10-01-million-particle-preview/`:
`progress.md`, `task-3.md`, RED/GREEN logs, `task3-checkpoint-cuda-full.log` and
`task3-checkpoint-blender-full.log`. Preserve the older connected-water scratch.
The plan does not mark Task 3 complete. Point preview, portable particle bake,
version 0.0.4, corrected reference visuals and real-time acceptance are unfinished.
Local PDF/video references, demo scenes and ignored clips are preserved.
The user explicitly authorized this checkpoint commit/push; a future merge or
publication requires its own authorization.

## Historical checkpoint before revised implementation

Branch: `feat/gpu-flow`, worktree `.worktrees/gpu-flow`.
Implementation base: `09f47ac69e3b5caae1930f03327b58e61479bf1e`.
Original approved design/plan: `2026-09-30-gpu-connected-water`.

Tasks 1–8 are complete. Task 9 has both 180-frame clips, comparison stills,
complete baseline and optimized Eevee measurements, an invalid interrupted
solid measurement, and one-million-particle solver probes. Task 10 packaging
and the independent whole-increment review have not happened.

The user changed the requirement on 2026-10-01: final meshing may run offline
during a later bake; particle simulation/preview must be real time at one million
particles for editing. No new release or million-particle acceptance is claimed.

The old full-mesh Eevee gate failed at 3.59 FPS / 309.67 ms p95. That gate is no
longer the active requirement. Current million-particle probes exclude meshing
and drawing: independent particles reached 22.31 FPS (minimum eight substeps)
or 25.05 FPS (minimum one); dense interacting particles still slowed to about
20 seconds per interval. Merging was disabled to retain the exact live count.
Do not present the short solver-only probes as complete viewport acceptance.

Verified optimizations include shared attached vertices/native smooth normals,
mesh index buffers matching RNA UNSIGNED types, CUDA interaction timing,
separate transfer/update timing, known-triangle projection with BVH fallback,
larger bounded hash grids and shorter occupied-neighbor loops. Differential
CUDA tests cover BVH equivalence, edge crossing/detachment and grid-independent
nearest-neighbor/overflow behavior. Timing reset is also covered.

Final source suites at this checkpoint: **66 Python, 54 CUDA, 59 Blender pass**.
The new solver-only benchmark script was exercised with one million live
particles. The saved demo/clips are pre-particle-optimization experimental
captures and are not a baked or portable GPU scene.

Evidence and reproduction: `docs/CONNECTED_WATER_VALIDATION.md` and
`artifacts/{connected-*,million-particle-*,particle-solver-script-check.json}`.
Large clips/demo files remain local and ignored. Selected stills/raw reports
and reproducible generators are retained in source history.

The next proposed architecture is in
`docs/superpowers/specs/2026-10-01-million-particle-preview-design.md`.
The user **approved this revised design on 2026-10-01**. It specifies source-local
GPU field interactions, lightweight full-count point drawing, particle caches
and offline meshing. The concrete implementation plan is
`docs/superpowers/plans/2026-10-01-million-particle-preview.md`. The historical
state below is superseded by the current saved implementation above.
Keep the original plan scratch ledger; do not delete it or repeat completed
Tasks 1–8. Once the plan is reviewed, start its Task 1 and carry the old unfinished
visual/package gates into its Tasks 6–9.
