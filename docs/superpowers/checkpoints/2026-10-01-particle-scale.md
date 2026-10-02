# Saved checkpoint — million-particle field work

**Updated 2026-10-02: plan complete (Tasks 1-9). 0.0.4 packaged locally, not pushed or published.**

## Final state (Task 9)

- **Package:** `artifacts/flumen-0.0.4.zip` (151,353,501 bytes, bundles Warp 1.17.0).
  - SHA256 `9b55646cc72653c43262011614bd5ac3a5b93daa049bc79c5c23c64a35d21748`
  - Built from the stage `artifacts/extension-stage-particles-v4` at `786256e` plus the
    version bump (committed with this checkpoint).
  - The stage passes `extension validate`, `scripts/smoke_test_blender.py --package` and
    `scripts/smoke_gpu_package.py`.
  - The GPU smoke covers FIELD preparation, continuous births, the full point batch,
    bake cancel/retry, offline meshing, playback with live CUDA forbidden, and handler
    cleanup.
- **Suites:** 112 Python, 82 CUDA and 75 Blender tests, all OK with no skips.
- **Gates:** the 1080p 1M-particle viewport gates pass for all four distributions, both
  at defaults and with contact hysteresis 0.07.
  - The drainage comparison against the reference fails at defaults, so FIELD ships as
    **experimental**.
  - With 8x water and hysteresis 0.07, rivulets form (`artifacts/rivulet-pinning-h07.png`).
  - No final combined clip was rendered: the user stopped it during meshing.
- **Independent final review** (one fresh reviewer, whole range from `09f47ac`):
  no Critical findings, 5 Important, all fixed in `786256e`:
  1. FIELD effective-spacing and force-cap diagnostics
  2. FIELD controls and the interactions toggle
  3. stale README and docs
  4. lock interface for baked renders
  5. unresolved-contact rollback test (mutation-checked)
- **Minor findings:**
  - Fixed: version-bump line endings, stage dir gitignore, resample churn and draw-time
    notes.
  - Deferred:
    - cache staleness is not validated on Mesh
    - deleting the host during a modal leaks its timer
    - the UI mesher uses one worker
    - playback re-hashes each frame
    - the Water style ignores clip planes
    - the original runtime error is replaced by "inputs changed"
    - prepare_source closes its caller's source on failure
    - the legacy Drops anchor-projection change, made before the plan
- **Local data (gitignored):**
  - `.clip-work/clip` (previous full clip: cache, mesh, frames)
  - `.clip-work/clip2` (stopped run: cache plus 96/180 mesh frames)
  - `.clip-work/finger/*` (hysteresis sweep caches)
  - `.clip-work/pca_check/*`


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

## Execution ledger (from scratch `progress.md`, verbatim)

# SDD ledger — plan: docs/superpowers/plans/2026-10-01-million-particle-preview.md

User approved plan 2026-10-01, Native method. Existing worktree gpu-flow, branch feat/gpu-flow; plan base 1c3d80d. Numerical implementation base for final whole-increment review remains 09f47ac69e3b5caae1930f03327b58e61479bf1e.
Pre-flight 1->2,3,5,7: prepared source/chart/contact ownership and global barycentric anchors supply bounded field dynamics, contact transport, reset reuse and cache provenance; compatible.
Pre-flight 2->3,4,6: field buffers and defaulted stats supply advection/point stream/stage reports; compatible.
Pre-flight 3->4,5,6,7: field FlowSolver retains legacy frame/emission/volume semantics; point/cache snapshots are read-only; compatible.
Pre-flight 4,5->6: point handler draws only published state; full-count reports measure actual frame updates and draws; compatible.
Pre-flight 1,5,7->8: immutable source/chart provenance and cache state supply CPU-only offline meshing/playback; compatible.
Pre-flight 1..8->9: canonical source and evidence feed package smoke and independent final review; compatible.
Ruling: use native Python/PowerShell bookkeeping instead of POSIX helper scripts — Windows shell — cost if wrong: bookkeeping reconciliation only.
Baseline carried from verified unchanged checkpoint: 66 Python / 54 CUDA / 59 Blender; no runtime/dependency changes since checkpoint. Original plan Tasks 1–8 remain completed; original unfinished gates carried forward.
Task 1 active, BASE 1c3d80d. Write config/preparation/anchor-transfer tests before implementation.
Task 1: complete (BASE 1c3d80d; config/preparation/borrow/settings/mirrored tests RED->GREEN; 68 Python, 60 CUDA, 61 Blender passed). Commands: python -m pytest -q; scripts/run_cuda_tests.py full; scripts/run_blender_tests.py full. Logs task1-{cuda,blender}-full.log. FIELD settings validated; bounded chart and contact preparation, stable segmented FP64 transfer, filtered face provenance, explicit borrowed ownership and outward mirrored normals verified. No million-particle performance claim.
Task 2 active: bounded node dynamics, physical wetness and low-contention stats reductions.
Task 2: complete (commits 5ed088b..d747231; 68 Python / 65 CUDA / 61 Blender passed; logs task2-{cuda,blender}-full.log). Dynamics tests RED->GREEN; dry-node velocity regression additionally RED->GREEN. Flat field, analytic incline drag, viscous energy decay, restoring capillary sign, 64-step failure, immutable volume and exact physical wetness verified. Statistics changed to 256-slot partial sums with FP64 final reduction; legacy ledger/replay tests stay green.
Task 3 active, BASE d747231. Native brief task-3.md. Write/observe motion+aggregation RED before implementation.
Task 3: motion/replay/65537-fallback rollback/dense conservation tests RED->GREEN. Warp tuple '_' reused across bool/vec3 and native vec3i inequality compiler errors diagnosed and fixed; no behavior bypass.
Task 3: Ruling: regularize field derivative operators at half requested field spacing while keeping exact source contacts — actual million Suzanne probe demanded 487 field steps due to source slivers; radius-independent field resolution must govern numerical work — cost if wrong: suppressed motion detail below requested resolution. New sliver equilibrium/stability regression RED->GREEN required; raw source min edge and operator spacing separately reported.
Task 3: resampling uses 99% retained parent /1% child at identical anchors/velocity — volume/momentum conserving and retains growing aggregate scale better than equal splits — cost if wrong: uneven marker weights can increase interpolation noise; report splits and reconstruction error.
Task 3: Ruling: queued attached fallback extends the exact adjacency walk by at most32 edges (main walk8), before narrowly eligible BVH fallback — million probe found three valid fine-triangle crossings rejected by one-neighbor eligibility; fine-strip regression reproduces this — cost if wrong: more queued triangle work and explicit rejection of valid motion beyond40 edges. Preserve actual local provenance; do not authorize projected contacts using approximate centroid neighborhoods.
Task 3 paused at explicit user request. Checkpoint: 68 Python / 73 CUDA / 61 Blender passed. Graph/eager and actual attached/free count regressions RED->GREEN; cached post-interval deposit reused. Ruling: round required field steps upward to powers of two (max64) and retain up to8 captured graphs - bounded launch overhead without reducing stability work - cost if wrong: extra GPU iterations/cache memory. No post-graph benchmark or complete Task3 characterization. Durable resume: docs/superpowers/checkpoints/2026-10-01-particle-scale.md. User authorized checkpoint commit and push; stop after saving.
Task 3 resumed 2026-10-02 from checkpoint 60cf74b. Post-graph solver-only (1M, Suzanne, 2 mm field/contact, resistance 60): attached 36.7 FPS mean (steady ~24.2 ms; frame 4 graph capture 53 ms), free 46.9, mixed 40.3; all finite, ledger <=6.5e-10. Substeps now 64 (pow2 rounding of 36-54) - at hard limit. Dense fails explicitly frame 2: field needs 231 steps >64; benchmark now writes status=failed report with last valid stats.
Task 3: synchronized profile (attached): deposit_attached 9.2 ms (previously untimed), aggregation 4.9 ms (stats showed 0.7, unsynced), advance 6.8 (field 3.9), resample 1.7. Ruling: deposit sorts int32 node keys; stable CUB radix sort keeps ascending contribution index, replacing int64 node*stride+index - frame-1 deposit bit-identical old/new - cost if wrong: nondeterministic reduction order. Deposit ~9 -> ~6 ms; attached steady ~19.3 ms (44.9 FPS mean). 68 Python / 73 CUDA / 61 Blender pass (task3-deposit-*-full.log). Observed: legacy pipeline already differs run-to-run by frame 4 (~1e-8 position), within 1e-6 replay tolerance.
Task 3: stage timing RED->GREEN (test_reports_event_stage_timings_and_owned_bytes; task3-timing-{red,green}.log). Ruling: field stages timed with CUDA events at solver level, summed over all intervals a seek processes (like solver_ms); field_ms stays wall time of evolve_field (bracketed by host readbacks), contact_ms = advance event - field_ms; owned_array_bytes = deduplicated CUDA wp.array capacity reachable from pool/field/prepared, excluding BVH/graphs - cost if wrong: event time includes queue idle gaps. 68 Python / 74 CUDA / 61 Blender pass. Solver-only 1M: attached 47.3 FPS (median 19.7 ms), free 61.8, mixed 48.1; stages sum to solver_ms; owned 582 MiB. Free still pays deposit 5.7 ms + field 2.4 ms with zero attached.
Task 3: fallback growth investigated (checkpoint step 4). 60-frame attached run FAILS at frame 28: "Contact fallback needs 66538 particles, exceeding 65536" (artifacts/field-million-attached-long.json, task3-long-attached.log). Cause: Suzanne fixture has 196 open-boundary faces (eye sockets, ears); boundary detachment drips particles into eyeball/socket crevices and hollow interior (winding number: ~80% of free particles inside). Temporary kernel reason probe (reverted): ~99% of queued particles are in ambiguous contact cells, <1% zero-clearance after face probe. Fallback ~= trapped free count, +~2.4k/interval. Not a nonfinite/ledger failure; bounded raise works as specified. Fix needs user decision: queue bound semantics vs capture/crevice handling vs fixture.
Task 3: user ruling 2026-10-02 - chunked fallback (Recommended option). compact_queue takes a prefix offset; exact contacts run in 65536 chunks; overfull no longer raises, unresolved still raises with rollback. test_thin_sheet_sweep_and_chunked_fallback RED (overflow RuntimeError) -> GREEN; 65537th particle matches first exactly. 68 Python / 74 CUDA / 61 Blender pass (task3-chunk-*-full.log). 60-frame 1M attached now passes: 41.5 FPS solver-only, median 24.2 ms, p95 25.4, finite, ledger 2.7e-10; fallback 170k by frame 63, contact 3-4 ms. Plan text updated.
Task 3: complete 2026-10-02. Opposing/zero-normal free merge regression passes without fix (no nonfinite demonstrated). Final suites 68 Python / 75 CUDA (task3-final-cuda-full.log) / 61 Blender. Docs: validation table, status, checkpoint updated; resume at Task 4.
Task 4 active 2026-10-02, BASE 333c7f6. Probe: Blender 5.2 GUI OpenGL; POINT_UNIFORM_COLOR attrs (pos VEC3); 1M VBO fill 1.7 ms; gpu unavailable in --background, so upload lazily in draw handler; xyz slice copy 4 ms.
Task 4: RED (missing point_snapshot / gpu_point_display; task4-{cuda,blender}-red.log) -> GREEN. Point stream: active-count readback, xyzr-only gather (even rank-stride subset), pinned copy of shown*16 bytes; full-snapshot ids/aux now lazily allocated. Draw: one global POST_VIEW SpaceView3D handler, POINT_UNIFORM_COLOR (CLIPPED when use_clip_planes), lazy upload in draw handler (gpu unavailable in background), strided xyz VBO fill (no host copy), depth test, state restored. Display settings point_size/point_color/display_limit outside FlowConfig; refresh_points republishes current frame without physics. Ruling: one read-only GPU batch per host shared by all viewport areas of the window (Blender shares the GPU context; batches are never written by draws) instead of per-area buffers - cost if wrong: multi-window contexts may need per-context batches. API probe caught 5.2 name use_clip_planes (not use_clip_3d). 68 Python / 77 CUDA / 64 Blender pass. Actual 1080p GUI probe (scripts/point_viewport_probe.py): 1M simulated+drawn, 30.6 FPS, median 32.6 ms, p95 34.3; readback 1.5, upload 2.1, draw submit 2.4, solver 19.8 ms. Interop not added (copies ~3.6 ms). Screenshot verified correct placement/depth.
Task 5 active, BASE 607c615.
Task 5: RED (create_gpu_host solver_backend missing; task5-red.log) -> GREEN (task5-green.log). physical_key(config) excludes FIELD offline-only reconstruction_scale; legacy keeps all fields. preparation_key = version + evaluated world geometry fingerprint (extract_source applies matrix/winding, islands) + field/contact spacing. Runtime keeps its own prepared reference; Reset parks it in _RETAINED and get_runtime reuses it only on identical preparation_key, else rebuilds and releases; purge/release_all/load/undo release retained references. mark_dirty flags only physical signature changes. create_gpu_host(..., solver_backend) keeps old calls; FIELD requires POINTS. New operator flumen.create_field_preview; UI shows effective field/contact spacing, fallback and attached/free counts, legacy pairwise note. Panel draw smoke-tested via stub layout for FIELD and LEGACY. FIELD not made default (awaits Task 6 evidence). 69 Python / 77 CUDA / 68 Blender pass.
Task 6 active, BASE 301b329.
Task 6: validator RED (missing module) -> GREEN (21 synthetic cases; average FPS from wall time; owned bytes must be below whole-device VRAM). Driver scripts/benchmark_particle_viewport.py subclasses point_viewport_probe.Probe; distribution helper extracted from solver benchmark. Full 120+600 1080p gates ALL FAIL on field stability bound: attached frame 85 (67 steps), free 180 (71; median frame 30.8 ms before failure), mixed 196 (65; median 34.9 ms), dense 2 (231). Trace (solver-only): demand oscillates 40-62 from max node speed spikes; capillary term constant ~25 steps (cap wave 0.268 m/s at 1 mm operator spacing); gravity wave grows with pooled max thickness 0.5 -> 4.9 mm by frame 95. Root cause: undersides never release drops (adhesion 15 > g 9.81), so water pools past capillary length (~2.7 mm). Background replay passed frame 85 where GUI failed: run-to-run drift flips borderline demand. Awaiting user decision on drip-release physics / step bound.
Task 6: user ruling 2026-10-02 - drip release. Attached particles on downward-facing surfaces release as free when interpolated field thickness * (g_hat . n) exceeds capillary length sqrt(sigma/(rho |g|)) (~2.7 mm). test_underside_film_beyond_capillary_length_drips RED (old kernel, release assertion) -> GREEN; thin underside and thick upward films stay attached. 90 Python / 78 CUDA / 68 Blender pass. Trace: ~500k released by frame 160, underside hmax ~3 mm, but demand still 60-64 (max node speed spikes ~0.5 m/s + constant capillary ~18 steps + wave ~12); solver-only now fails frame 164 (65 steps) instead of 85.
Task 6: user ruling 2026-10-02 - thin-film wall drag. field_kick damping = resistance + surface_damping + 3 nu / max(h^2,1e-24) (nu = field_viscosity). test_thin_film_wall_drag RED (h=1 mm used drag 5 not 8) -> GREEN for h=1 mm and 0.1 mm. 90 Python / 79 CUDA / 68 Blender pass. Trace: 400 solver-only frames complete; demand ~50-59; max speed ~0.48 m/s persistently, just under capture_speed 0.5 (likely recaptured-drop impact velocity deposited to nodes; not located); thin coating drains much slower; hmax creeping 0.2 -> 1.8 mm by 388.
Task 6: attached gate after drip+wall drag reaches frame 475 (353 measured), median 31.3 ms, p95 33.59; stops at 66 steps. Budget: ~0.48 m/s node speed (~32 steps, just under capture_speed; likely recaptured drops), capillary ~18, pooled-film wave ~11 near 2.7 mm drip threshold. Stage medians: solver 17.9 (field 2.7, contact 2.3, aggregation 4.9, resample 1.4, deposit 5.7), readback 1.5, upload 2.0, draw 2.4, redraw 10.1.
Task 6: finding - field thickness is frozen within each interval (_evolve_kernels only integrates velocity: exact exponential drag kick + implicit viscosity), so the wave-speed substep demand guarded a coupled scheme that does not exist. Scratch experiment: forcing 8 substeps instead of ~55 for 720 solver-only frames gives the same result within run-to-run noise (free 127,825 vs 127,387; h99 0.148 vs 0.141 mm), finite, ledger <=8.7e-10. User ruling 2026-10-02 (supersedes the earlier "implicit capillary" pick, which rested on my wrong framing): fixed substeps = minimum_substeps, demand abort removed, per-interval Courant number (max node speed*dt/operator spacing) reported as stats.field_courant, abort only on nonfinite field state (explicit counter; atomic_max ignores NaN). Task 2 step-limit test rewritten as test_capillary_response_and_substep_guard RED (1 != 8) -> GREEN. 90 Python / 79 CUDA / 68 Blender pass.
Task 6: gates after substep fix - attached PASS (34.9 FPS, p95 29.9), free PASS (36.5, 27.7), mixed PASS (35.8, 30.6), dense FAIL (34.8, p95 35.2; 65 slow frames, solver ~20 ms; dense blob drips and falls, all free). Justified bounded optimization: deposit compacts attached-only contributions (mask + scan + 3*attached sort) instead of sorting 3*capacity keys with sentinels. Frame-1 deposit bit-identical to baseline (all five arrays). Regression guard test_deposit_counts_only_attached_particles (behavior-preserving, passes on old code too). 90 Python / 80 CUDA / 68 Blender pass.
Task 6: final-round gates (after compaction): dense PASS 46.2 FPS p95 30.1; free PASS 40.2 p95 29.7; mixed PASS 37.6 p95 30.3; attached FAIL p95 36.8 (50 slow frames in bands 340-354, 386-407; every stage incl. viewport redraw 20-30% slower, deposit unchanged) then 22 FPS on a rerun polluted by my per-second nvidia-smi polling (discarded; GPU clocks at max, no throttle flags). DaVinci Resolve (~1.2 cores busy, shares GPU) and FileAssociation (~0.9 core) running during gates. Clean rerun pending user.
Task 6: FINAL clean gates (Resolve closed, FileAssociation still running): attached PASS 34.3 FPS p95 31.2 max 41.5; free PASS 45.6/23.6; mixed PASS 42.1/25.3; dense PASS 47.2/23.0. All 1M live and displayed, ledger <=3.0e-9, finite. Owned arrays 582 MiB; whole-device 1.75-2.0 GiB. Attached p95 margin ~2 ms: sensitive to background GPU/CPU load (earlier contaminated runs recorded in this ledger).
Task 6: drainage capture (examples/create_field_water_demo.py, scripts/capture_field_drainage.py; emission from the collision surface, attached, zero velocity): front 1.7 cm in 180 frames, contrast 0.33->0.84, channel persistence 0.08->0.39, 903 grown drops, 150k detach events, ledger <1.3e-15. Visual reference gate FAIL: too slow, uniform sheet, no rivulet fingers; lattice dot pattern of recaptured drops on chin. Likely resistance 60 + wall drag. Dense solver stress 600 frames 93.6 FPS p95 11.5. Task 6 complete with performance PASS and visual FAIL recorded.
Task 7 active, BASE 967b506. Note: cache module drafted before tests; moved aside to observe RED honestly.
Task 7: cache tests RED (module absent; module had been drafted first and was moved aside) -> GREEN (10 pure-Python cases incl. subprocess read without warp/bpy). Bake tests RED (gpu_bake missing) -> GREEN (cancel/retry, failure leaves live state and preparation references, editing writes no cache). Rulings: bake limited to Surface Field hosts (legacy hosts keep native bakes); CacheHeader adds field_nodes (wetness length, for estimate/read bounds) beyond plan interface; destination = existing parent folder + new cache name; finish() re-hashes every file before marking complete. Probe (scripts/bake_particle_probe.py, 1M particles, 10 frames): 884 MB (estimate 885 MB), 1.56 s/frame bake, 10.1 s to read all frames, peak process 856 MiB, live state unchanged; baked frame 5 vs live: ids/slots/volumes/states/faces identical, position <=3.0e-8 m, velocity <=1.2e-7, bary <=1.1e-5 (separate-solver GPU drift, within 1e-6 m replay tolerance). 100 Python / 80 CUDA / 71 Blender pass.
Task 8 active, BASE 0a2875c. water_types move bracketed by CUDA test_surface_mesh/test_free_mesh (5+5 pass).
Task 8: offline mesher tests RED (module absent) -> GREEN (4). CPU barrier deviation: grid samples within 0.55 pitch of source cleared; no per particle-sample ray test (live CUDA mesher does both).
Task 8: baked display tests RED (module absent) -> GREEN; playback/render with require_cuda forbidden; wet proxy slot 0 = owned wet copy, other slots share source materials (limitation: only slot 0 shows wetness); unsupported source attributes listed. Mesh operator steps iter_mesh_cache per timer. 104 Python / 80 CUDA / 73 Blender pass.
Task 8: film lattice now sized by 95th-percentile edge (conforming, reports film_max_spacing); separate film_spacing (operator default 0.5 mm). Film shell vectorized by clip case (1/2/3 wet corners, boundary and cut walls); tests stay green. Removed library tracemalloc (it made the Python film path ~25x slower; a 1M frame never finished in 10 min). 1M frame probe: film 1 mm 7.8 s, 84 MB, 3.5M tris, film volume error 0.5%; film 0.5 mm 26.5 s, 271 MB, 11.3M tris, 0.6%; free drops at 0.1 mm: 45% meshed, 92% of free volume subresolution (reported). Disk: 48 GB free; 180 frames at 1M = ~16 GB cache + 15-49 GB mesh.
Task 8 speedups (user request 2026-10-02), all byte-identical to the previous mesher (SHA256 of frames 1/90/170 of the 250k clip cache unchanged): film lattice built once per sequence; deposit/drop field accumulate with one in-order bincount; contour limited to the box of cells with a corner >= ISO; particle kernel boxes clipped to each tile and bucketed by box size, contributions re-sorted to particle/point order before bincount; iter_mesh_cache(workers=N) meshes frames in spawn processes (Blender's bundled python.exe), results in frame order; parallel == serial SHA test. Frame times: 1: 1.2 -> 0.6 s, 90: 79.6 -> 32.2 s, 170: 565 -> 92.6 s. Earlier guess that np.add.at dominated was wrong: profile showed per-tile box waste. Clip script: --reuse-cache, --workers.
Task 8: parallel meshing inside Blender failed (BrokenProcessPool): spawned workers re-ran the parent main script (imports bpy). Fix: hide __main__.__file__ while the pool spawns; verified in Blender background (2 workers).
Task 8: complete 2026-10-02. 180-frame 250k clip (radius scaled to 1M volume; drops 0.1 mm, film 2 mm): mesh 31 min on 6 workers (median 47.5 s, max 145 s/frame), 8.3 GB; cache 4.0 GB reused; renders 4.6 + 6.1 min. Film volume error median 0.6%; f180 free meshed 89%, subresolution 11%. Visual gate FAIL: crown film connected, front reaches nose, but slow drainage, no rivulets/drips, speckled thin film near 10 um clip (deposit noise ~10 particles per 2 mm node). 105 Python / 80 CUDA / 73 Blender pass.
Look fix A (user 2026-10-02): MeshOptions.film_smoothing (default 0 keeps byte-identical outputs) diffuses deposited film volume along lattice edges with symmetric fluxes (exact volume, convex steps). Noisy test film dry nodes 362 -> 19 at 6 steps. Operator and clip default 6.
Look fix B: resistance sweep 60/20/5/1/0 at the 1M-equivalent water volume: front 1.8-1.9 cm by f180 regardless (thin 0.05-0.17 mm film is wall-drag dominated: physically slow). Reference uses 0.8 mm particle separation (mm-thick coat, ~30-60x our water). Volume x resistance sweep: 8x water (2x radius) with resistance 60 gives progressive front 2.9/5.0/9.7/11.7 cm, 157k of 250k released, channel persistence 0.50; lower resistance or more water dumps water within ~30 frames. Ruling proposed: no physics default change; look demo uses 2x radius. Speed gates unaffected (spec fixture unchanged). artifacts/drainage-sweep.json.
Look fix: frame 60 of 8x-water cache took 389 s at 0.2 mm drop grid (6.6M drop tris; field 210 s, barrier 84 s, contour 69 s). Drops at 0.2 mm are ~0.5 px in the clip view. MeshOptions.free_crop skips out-of-shot drops and reports cropped_volume. 0.4 mm + crop: 40.8 s (9.5x), 0.7M drop tris, 40 MB/frame, 74% drop volume meshed, 29% subresolution.
Screen-space water (user-requested increment beyond the approved plan, 2026-10-02): flumen/gpu_water_screen.py + point_style WATER (display-only settings water_color/water_smoothing/water_radius_scale). Passes: collision-surface linear depth (own depth buffer), sphere sprites discarded behind the surface, separable narrow-range filter with see-through/hole fill and skip (not clamp) outside range, composite shading films with surface geometry and drops with filtered-depth normals, depth biased forward. Key bug: cached GPUFrameBuffer objects silently lost depth testing (LESS == ALWAYS in-draw); minimal in-draw test showed framebuffers built per draw work, so targets cache textures only and framebuffers are rebuilt each draw. Readback confirmed surface depth == ray cast (0.4864 vs 0.4863 m) and water 0.3-2 mm in front. Visual: clean glossy wet sheet, no speckle. Performance not yet measured without the clip render competing for the GPU.
Look fix step 2 (user 2026-10-02): offline drop_kernel 'pca' (Yu & Turk: bounded neighbour search, weighted covariance, volume-preserving clamped stretch, isolated drops round, smoothed centres). Frame 90: needles 80 -> 49, time 33 -> 29 s. The visible nose spikes were NOT drops: film nodes with tiny lattice area (0.017 mm2 vs 0.27 median) held 96 mm "thickness". film_max_thickness caps the film and turns excess into pendant drops (volume moved, reported as pooled_volume). Smoothing 20 + cap 2 mm + PCA: spikes gone, near-continuous sheet. Library defaults stay off (outputs byte-identical); operator/clip defaults: smoothing 20, cap 2 mm, PCA. Tests: PCA round/stream, cap pendant drop (volume conserved). 109 Python pass.
Clip with PCA + cap + smoothing 20: mesh 21 min (8 workers), renders 2.8 + 3.9 min; f030 connected sheet with tongues and drips, no spikes; f180 residual thin film patchy. Water preview gate (attached, 1080p, 120+600): PASS 33.3 FPS, p95 32.0 ms, draw submit 7.7 ms, redraw 11.1 ms.
Look 1/4 (user: all four look items): film_sheen = cosmetic minimum thickness x cached wetness (wetness ~1 wherever water flowed). Frame 180: leopard patches gone, continuous coat; sheen 0.35 uL vs film 15.8 uL, reported as sheen_volume. Operator/clip default 2e-5 m.
Look 2/4: downhill streak weights (align^8) passed a numeric anisotropy test (2.7x) but frame-90 EEVEE renders with/without were near-identical: smoothing only rearranges existing tens-of-um thickness noise; channels must come from physics (task 4). Removed (not committed) - no visible benefit.
Look 3/4: mesher-side drop trails rejected (drops median r=0.12mm < 0.4mm spacing; visible trails fabricate 48x real free volume, 2e9 box cells). Instead: free vertices carry kernel-weighted particle velocity (momentum lerp / ISO, exact for uniform motion), saved as free_velocity; baked display writes POINT 'velocity' attribute; Cycles motion blur (shutter 1) turns falling drops into streams. EEVEE ignores it. Clip: --water-engine cycles --shutter 1. Cycles also reveals maze ripple on head film (open).
Look extra: Cycles maze on head = film inner face coplanar with source (z-fight; confirmed by hiding proxy). Film shell floated FILM_GAP=10um along normal; maze gone. artifacts/offline-look-cycles-f030.png (sheen + blur shutter 1 + gap).
Look 4/4: contact_hysteresis (dcos, default 0 = off). field_kick: dry nodes hold up to (sigma/rho) dcos (1-w)/(h L), L=operator spacing; pinned nodes do not wet (field_wetness), so held fronts stay held until thick enough -> thickness-selective breakthrough -> fingers follow wet tracks. First version (pinned nodes still wetting) only gave uniform creep. Sweep 90f: 0.05 sheet+channels, 0.1 tongues, 0.2 held. 0.07 @180f: central rivulet down muzzle to chin with dry stripes (artifacts/rivulet-pinning-h07.png). Gates @0.07 all PASS: attached 35.0/30.6, free 44.7/23.7, mixed 41.9/24.7, dense 47.5/21.8. Bake 250k/180f 127s. Tests 112 py, 74 blender, 81 cuda.
