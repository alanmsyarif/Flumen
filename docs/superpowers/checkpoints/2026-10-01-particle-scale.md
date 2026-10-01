# Verified checkpoint — particle scale requirement

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
It is a **draft awaiting review**, not an approved replacement implementation
plan. It proposes source-local GPU field interactions, lightweight full-count
point drawing, particle caches and offline meshing. Preserve Native execution
after approval. Keep the original plan scratch ledger; do not delete it or
repeat completed Tasks 1–8.
