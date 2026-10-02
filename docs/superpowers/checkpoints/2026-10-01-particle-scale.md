# Saved checkpoint — million-particle field work

**Updated 2026-10-02: revised Tasks 3-8 complete. Resume at Task 9 when requested.**

Task 8: offline CPU mesher, CUDA-free baked playback and a 180-frame clip
(250k particles). Visual reference gate still fails (slow drainage, no rivulets,
speckled thin film). Local clip work (~12 GB) lives in `.clip-work/` (ignored).

Task 7: validated particle caches (`flumen/particle_cache.py`, `flumen/gpu_bake.py`).
A 1M-particle, 10-frame bake writes 884 MB (about 88 MB per frame) at 1.56 s
per frame; reading needs no CUDA.

Task 6 outcome: all four 1080p million-particle viewport gates pass with
background GPU apps closed; the drainage comparison against the reference
fails (too slow, no rivulets). Details and user rulings (drip release,
thin-film wall drag, fixed field substeps, chunked fallback) are in
`docs/CONNECTED_WATER_VALIDATION.md` and the native ledger. Visual tuning of
drainage is an open item to schedule with the user.

The revised [design](../specs/2026-10-01-million-particle-preview-design.md) and
[plan](../plans/2026-10-01-million-particle-preview.md) are both approved. Native
sequential execution is selected, with one fresh independent final review.

## Current saved implementation

- Task 1 (`5ed088b`) and Task 2 (`d747231`) complete, as before.
- Task 3 complete (`60cf74b`..HEAD): FIELD transport, bounded aggregation and
  resampling; 32-bit deposit sort keys (frame-1 deposit bit-identical);
  CUDA-event stage timings including deposit/resampling plus owned CUDA array
  bytes; contact fallback resolved in repeated 65536-entry chunks (user ruling
  2026-10-02); benchmark writes explicit failure reports with last valid stats.
- Solver-only, one million particles, Suzanne, 2 mm spacing, resistance 60:
  attached 47.3 FPS, free 61.8, mixed 48.1; dense single-anchor fails at frame 2
  (field needs 231 steps > 64). 60-interval attached run passes at 41.5 FPS.
  See `docs/CONNECTED_WATER_VALIDATION.md` and `artifacts/field-million-*`.

Verification: **68 Python, 75 CUDA and 61 Blender tests passed**. No independent
final review has run.

Open observations, not blockers:
- Attached runs sit at the 64 field-step hard limit after power-of-two rounding.
- Free-only runs still pay deposit (5.7 ms) and field (2.4 ms) with no attached
  particles; compacting attached-only contributions would remove most of it.
- Pipeline differs run-to-run by about 1e-8 m in position by frame 4 (within the
  1e-6 replay tolerance); frame-1 deposit is deterministic.
- Suzanne's open eye/ear boundaries trap water in ambiguous crevices; fallback
  count grows about 2.4k per interval at roughly 3-4 ms contact cost.

## Resume at Task 4

1. Read this checkpoint, approved spec/plan and the native ledger
   `.superpowers/sdd/2026-10-01-million-particle-preview/progress.md`. Do not
   redo Tasks 1-3 or original connected-water Tasks 1-8.
2. Tasks 4-9: compact direct point draw, physical/cosmetic signatures and
   preparation reuse, full-count viewport and reference gates, immutable particle
   cache, CPU offline mesher/CUDA-free playback, final independent review, docs
   and package checks.

Point preview, portable particle bake, version 0.0.4, corrected reference
visuals and real-time viewport acceptance are unfinished. Local PDF/video
references, demo scenes and ignored clips are preserved. A merge, push or
publication requires the user's authorization.

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
