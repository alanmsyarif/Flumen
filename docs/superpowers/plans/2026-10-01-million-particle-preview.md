# Million-particle preview and offline water bake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make stationary-surface water editable with one million live GPU particles and continuous frame emission, then bake the final connected water surface separately.

**Architecture:** Replace per-particle neighborhood forces in the new backend with conservative transfer to a bounded surface field and source-local contact transport. Draw a compact particle stream directly in Blender's viewport; explicit immutable particle caches feed a separate offline mesher. Preserve the existing Drops, pairwise Connected experiments and Geometry Nodes workflows.

**Tech Stack:** Python, NumPy, Warp 1.17.0/CUDA, Blender 5.2 Python/GPU API; Windows x64 and the RTX 5050 test machine. No additional runtime dependency.

**Spec:** `docs/superpowers/specs/2026-10-01-million-particle-preview-design.md`, approved by the user on 2026-10-01.

## Global Constraints

- Stationary collision surfaces; world coordinates in metres, Scene Unit Scale 1.0. Moving surfaces and a full FLIP/APIC reproduction are outside this increment.
- Preserve continuous integer-frame births, scaled physical time, deterministic replay and particle-owned volume. Wetness is independent of reconstruction and updates once per completed physical interval.
- At most 1,000,000 particle slots, 100,000 surface nodes and 2,097,152 contact-field samples. The field backend must not allocate the legacy `(capacity,64)` neighborhood arrays.
- Field resolution is independent of particle radius. Unsupported geometry/resolution, limited motion and unresolved volume must have explicit diagnostics.
- Acceptance: 120 warmup plus 600 measured actual 1920x1080 draws; at least 1,000,000 live simulated and displayed particles in every measured frame; average >=30 FPS and p95 <=33.3 ms. No skipped simulation intervals or hidden display decimation.
- Primary fixture: approximately 0.3 m Suzanne, 15,000–25,000 evaluated triangles, 0.1 mm nominal radius, time scale 0.5, scene FPS 30. Measure attached, free, mixed and dense concentrated distributions separately.
- Ledger relative error <=1e-5; equivalent replay positions within 1e-6 m; finite state, bounded storage/work, deterministic births and safe reuse of retired slots.
- Materials, point appearance and offline mesh quality do not reset physics. Physical edits reuse unchanged prepared source contacts. A changed evaluated source fingerprint invalidates preparation and cache compatibility.
- No mandatory cache writes or final mesh updates during interactive editing. Finished mesh playback/rendering works without CUDA; incomplete/corrupt/incompatible particle caches are rejected.
- Use canonical `flumen/` sources and the existing worktree/branch. Preserve local references, old evidence and original unfinished plan scratch. Do not claim any performance or visual gate passed without its evidence.
- Preserve **Native** execution: implement sequentially, then one fresh independent whole-increment reviewer. No per-task implementation agents. No push, merge or publication is part of this plan.

## Review Focus

1. Tiny particles on coarse triangles must deposit all attached volume, including triangle-edge anchors (Task 1).
2. Folded, mirrored, thin or nonmanifold surfaces must not route water onto an unrelated sheet; unresolved contacts must fail explicitly (Tasks 1, 3).
3. Dense clusters, coincident drops and billion-particle birth requests must stay capacity/work bounded and conserve volume through aggregation/reuse (Tasks 2, 3).
4. Redraws, material changes, source edits, duplicate hosts, undo/load and failed preparation must preserve independent ownership and never advance physics twice (Tasks 4, 5).
5. Cancelled bakes, truncated frames, changed sources/settings and mesh-tile boundaries must never produce a falsely complete cache or lose/duplicate water (Tasks 7, 8).

---

## Files, contracts and execution

Extend `gpu/{config,source,state,emission,solver}.py` and the existing Blender adapters. New numerical modules have no `bpy` imports: `surface_chart.py` (refined charts/adjacency), `contact_field.py` (bounded free contact acceleration), `prepared.py` (static ownership), `field_solver.py` (transfer/field evolution), `field_motion.py` (particle transport), `field_aggregate.py` (bounded merges/resampling). New CPU-only `particle_cache.py` and `offline_mesher.py` own serialized frames and mesh reconstruction. New Blender adapters: `gpu_point_display.py`, `gpu_bake.py`, `gpu_baked_display.py`.

Types owned by tasks:

- Task 1: `SurfaceChart` contains CPU/GPU nodes, triangles, areas, normals, adjacency, original global-face provenance and per-source-face refinement coordinates. `ContactField` contains bounded samples, closest global face/island, distance/error bounds and ambiguity flags. `PreparedSource` owns the existing `SourceMesh`, chart and contact field, fingerprint and explicit reference count; `retain()`/`release()` free once at zero. `FlowSolver(..., prepared: PreparedSource | None = None)` borrows by retaining; legacy callers continue to own/close their source as before.
- Task 2: `FieldBuffers` owns node volume/momentum, thickness, tangent velocity, wetness, ping-pong scratch, reductions and timing events. `FieldStep` contains `substeps`, `represented_volume`, `unrepresented_volume`, `limited_count` and stage timings. Extend `FrameStats` with defaulted `backend`, `field_ms`, `contact_ms`, `aggregation_ms`, `readback_ms`, `upload_ms`, `draw_ms`, `displayed_count`, `contact_fallback_count`, `contact_unresolved_count`, `resampled_count`, `owned_array_bytes`, `field_nodes`, `contact_samples`.
- Task 4: `PointBatch(xyzr: np.ndarray, live_count: int, displayed_count: int, frame: int)` uses contiguous float32 `(shown,4)` positions/radii. No IDs, normals or velocities are copied for point preview. Legacy `DisplayBatch` stays unchanged.
- Task 7: frozen `CacheHeader` and `CachedFrame` schemas below; independent of Warp/Blender imports. Task 8: frozen `MeshOptions(surface_spacing: float, free_spacing: float, tile_cells: int = 32)` and existing `MeshBatch`/`WaterGeometry` output types.

Start execution from the approved plan commit; do not repeat original Tasks 1–8. Carry the original unfinished visual/package work into Tasks 6–9 here. Record RED/GREEN commands, evidence and justified deviations in `.superpowers/sdd/2026-10-01-million-particle-preview/progress.md`; retain the old ledger until the whole increment finishes.

PowerShell commands used below (execute from `.worktrees/gpu-flow`):

```powershell
$FlumenBlender = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
# CUDA(file):
& $FlumenBlender --background --factory-startup --python-exit-code 1 --python scripts/run_cuda_tests.py -- 'test_filename.py'
# Blender(file):
& $FlumenBlender --background --factory-startup --python-exit-code 1 --python scripts/run_blender_tests.py -- --pattern 'test_filename.py'
# Pure suite:
python -m pytest -q
```

Each `CUDA(file)`/`Blender(file)` below expands to that command with the named file. CUDA tests use existing unittest/actual-device patterns and never silently skip. Run GPU jobs sequentially. Use the recorded 66 Python / 54 CUDA / 59 Blender checkpoint as baseline; rerun it only if the execution base or environment changes. Each task commits only its named files plus its associated documentation.

### Task 1: Reusable source charts and conservative anchor transfer

**Files:** Create `flumen/gpu/{surface_chart,contact_field,prepared,field_solver}.py`, `tests/cuda/test_field_preparation.py`; modify `flumen/gpu/{config,source,solver}.py`, `flumen/gpu_properties.py`, `flumen/gpu_runtime.py`, `tests/blender/{test_gpu_settings,test_gpu_source}.py`; create `tests/test_gpu_field_config.py`.

**Interfaces:** `prepare_source(source: SourceMesh, fingerprint: str, field_spacing: float, contact_spacing: float) -> PreparedSource`; `build_chart(source: SourceMesh, spacing: float, node_budget: int = 100000) -> SurfaceChart`; `build_contact_field(source: SourceMesh, spacing: float, sample_budget: int = 2097152) -> ContactField`; `FieldBuffers(chart: SurfaceChart, device: str)`; `deposit_attached(pool: ParticlePool, prepared: PreparedSource, buffers: FieldBuffers) -> None`.

- [ ] Write `test_field_config_bounds` and `test_legacy_config_defaults`. Add `solver_backend='LEGACY'|'FIELD'` (default LEGACY), `display_mode` option POINTS, `field_spacing=.001` in [0.00005,0.02] m, `contact_spacing=.002` in [0.00005,0.05] m, `field_viscosity=1e-6` in [0,0.01] m²/s, `surface_tension=.072` in [0,1] N/m, `resample_target=0` integer [0,capacity]. Assert bool/nonfinite/out-of-range rejection and RNA endpoint round trips. FIELD requires POINTS; old configs retain exact defaults.
- [ ] Write `test_subspacing_anchor_deposition`, `test_chart_fold_and_edge_locality`, `test_preparation_budgets_and_release`. On a 0.1 m triangle with radius 0.0001 m, assert `abs(deposited_volume-attached_volume)/attached_volume <= 1e-5`, including all vertices/edges and two islands. Opposite folded sheets get zero cross-sheet deposit; total chart area differs <=1%; nodes <=100000 and samples <=2097152. Mirrored extraction preserves outward orientation. Nonmanifold edges are explicit barriers, and unsupported resolution is reported. Two retains and three matching releases free once; injected allocation failure leaves no owned arrays.

  ```python
  assert abs(deposited_volume - attached_volume) <= attached_volume * 1e-5
  assert unrelated_sheet_volume == 0
  assert node_count <= 100000 and sample_count <= 2097152
  ```

- [ ] Run `python -m pytest tests/test_gpu_field_config.py -q` and `CUDA(test_field_preparation.py)`; expect missing FIELD settings/preparation/transfer failures.
- [ ] Implement deterministic uniform midpoint refinement per original triangle with a global refinement level chosen to fit the cap. Retain analytic original-barycentric-to-child lookup, so every valid anchor locates a child triangle and deposits normalized barycentric volume/momentum to its three nodes regardless of radius. Use fixed three-contribution-per-slot buffers, sort by node/slot/local-weight key, then fixed-order segmented block reductions with float64 sums; avoid million-way contended node atomics and race-dependent float32 accumulation. Share nodes only along actual mesh edges; split fans at nonmanifold/crease barriers, use global face IDs throughout. Publish areas, nonnegative edge weights and gradients/Laplacian coefficients. Contact preparation chooses bounded grid dimensions, stores conservative distance interpolation error bounds and marks cells containing competing sheets ambiguous; preparation may use exact BVH work offline. Report requested/effective spacing and all barrier/ambiguity counts. Correct negative-determinant evaluated transforms when extracting winding, with legacy motion regressions; retain `SourceMesh.evaluated_triangle_ids` through degenerate filtering for later attribute mapping. Make source sampling gravity/mask refreshable independently of immutable collision arrays.
- [ ] Run both focused commands plus `CUDA(test_source.py)`, `Blender(test_gpu_source.py)` and `Blender(test_gpu_settings.py)`; expect PASS and compatible legacy source behavior. Commit `feat: prepare bounded surface charts and contact fields`.

### Task 2: Bounded field dynamics, wetness and volume reductions

**Files:** Modify `flumen/gpu/{field_solver,state,solver}.py`; create `tests/cuda/test_field_dynamics.py`.

**Interfaces:** `evolve_field(prepared: PreparedSource, buffers: FieldBuffers, config: FlowConfig, dt: float) -> FieldStep`; `sample_field(prepared: PreparedSource, buffers: FieldBuffers, face: int, bary: tuple[float,float]) -> tuple[float, tuple[float,float,float]]` is a CPU diagnostic wrapper over the same CUDA interpolation. Field kernels operate on chart nodes/edges, never particle neighbors.

- [ ] Write `test_transfer_momentum_and_flat_film`, `test_incline_drainage_and_viscous_decay`, `test_capillary_response_and_step_limit`, `test_wetness_physical_time`. Assert volume and momentum transfer relative errors <=1e-5; constant thickness has zero pressure/capillary acceleration; gravity accelerates downhill and resistance approaches its analytic terminal speed; viscosity decreases velocity differences without increasing kinetic energy. A height bump has restoring capillary acceleration, finite state and <=64 field substeps. Equivalent physical drying intervals agree within 1e-6; repeated zero-dt calls change neither wetness nor liquid. Excessive stability demand produces a typed failure before particle state advances.

  ```python
  assert volume_relative_error <= 1e-5 and momentum_relative_error <= 1e-5
  assert kinetic_energy_after <= kinetic_energy_before + 1e-12
  assert 1 <= step.substeps <= 64
  np.testing.assert_allclose(wetness_a, wetness_b, atol=1e-6, rtol=0)
  ```

- [ ] Run `CUDA(test_field_dynamics.py)`; expect absent evolution/reduction behavior.
- [ ] Implement thickness `h_i=V_i/A_i`, deposited tangent velocity `u_i=M_i/V_i`, water density 1000 kg/m³, and tangent acceleration `g_T - |g| grad(h) + (surface_tension/1000) grad(Laplacian(h))`. Use chart-local area-weighted gradients and physical Laplacian `sum_j(w_ij*(h_j-h_i))/A_i` with nonnegative symmetric edge weights, viscosity via eight bounded implicit Jacobi iterations per substep, and analytic resistance/damping. Cap pressure and capillary accelerations using the existing repulsion/cohesion controls, count activation, and report the approximation. Select <=64 steps from chart minimum edge, transport/wave and capillary stability bounds; never discard elapsed time when the bound is exceeded. Derived deposits own no ledger volume. Update wetness from prior interval coverage with exact physical-time exponential deposition/drying once; initialize coverage at the start without advancing wetness. Replace million-way global stats atomics with fixed block partial reductions and float64 final sums; preserve existing stats meanings.
- [ ] Run the focused command plus `CUDA(test_solver.py)` and `CUDA(test_surface_fields.py)`; expect PASS. Record numerical coefficients/stability diagnostic in validation docs. Commit `feat: evolve conservative bounded surface fields`.

### Task 3: Local particle transport and bounded aggregation

**Files:** Create `flumen/gpu/{field_motion,field_aggregate}.py`, `tests/cuda/test_field_motion.py`; modify `flumen/gpu/{solver,state,emission}.py`, `scripts/benchmark_particle_solver.py`.

**Interfaces:** `advance_field_particles(pool: ParticlePool, prepared: PreparedSource, buffers: FieldBuffers, config: FlowConfig, dt: float) -> FieldStep`; `aggregate_field_particles(pool: ParticlePool, prepared: PreparedSource, config: FlowConfig) -> None`; `resample_particles(pool: ParticlePool, prepared: PreparedSource, target: int) -> None`. FIELD `FlowSolver` owns field buffers and retains preparation; it allocates no `InteractionBuffers` or live geometry buffers.

- [ ] Write `test_chart_walk_matches_exact_contact`, `test_thin_sheet_sweep_and_fallback_overflow`, `test_dense_merge_and_resampling`, `test_field_replay_births_and_reuse`. Across edges/islands/fold barriers, assert proper global face/bary anchors, no unrelated-sheet transfer and nonpenetration within the contact error bound. Compare smooth-plane motion to exact BVH within 1e-6 m. High-speed sweeps cannot tunnel through a thin sheet. An overfull fallback queue raises an explicit error without committing half an interval. Coincident 10000-particle clusters stay finite, use <=8 candidate checks per particle and conserve volume/momentum <=1e-5. Resampling adds no emitted volume; IDs stay unique. Rate 3 plus coating 5 gives 8/11/14 births at frames 1/2/3; request 10**9 at capacity 5 remains bounded. Replay/retirement/slot reuse reproduces IDs and positions within 1e-6 m.

  ```python
  np.testing.assert_allclose(replayed_positions, sequential_positions, atol=1e-6, rtol=0)
  assert emitted_volume_after_resampling == emitted_volume_before_resampling
  assert len(np.unique(live_ids)) == len(live_ids)
  assert abs(live_volume + removed_volume - emitted_volume) <= emitted_volume * 1e-5
  ```

- [ ] Run `CUDA(test_field_motion.py)`; expect absent FIELD transport/aggregation.
- [ ] Gather the field once per completed interval and advect attached anchors by a maximum eight-edge adjacency walk, parallel-transporting tangent velocity across valid faces. Use existing adhesion/normal-turn rules for detachment. For free ballistic sweeps, conservative distance bounds prove clear motion or run at most eight contact-field steps; ambiguous/unfinished cases enter a deterministic compact exact-BVH fallback queue capped at 65536. No fallback overflow may pass acceptance. Stage particle, field/wetness and ledger writes until contact validation succeeds, then commit; age/retirement, wetness and ledger advance once. Aggregate using sorted `(state, provenance, spatial cell, ID)` keys and up to eight adjacent candidates, reciprocal ownership, same-sheet/visibility checks and the existing maximum merged radius. Unmerged excess remains live and counted, never discarded. Optional explicit resampling fills vacant slots by deterministic volume splits of larger particles with identical velocity and local anchors, preserving volume/momentum; allocate new IDs from the same exhaustion-checked `pool.next_id` stream as emission, but report splits separately from physical births. Re-deposit post-transport/birth state for diagnostics without another wetness interval.
- [ ] Run focused plus `CUDA(test_emission.py)`, `CUDA(test_motion.py)`, `CUDA(test_merge.py)` and `CUDA(test_connected_motion.py)`; expect PASS. Extend the solver benchmark with `--backend FIELD --distribution attached|free|mixed|dense`, preparation time, stages and owned array bytes. Run million-particle solver-only characterization for all four distributions sequentially; label it feasibility, not viewport acceptance. Commit `feat: transport field particles with bounded local contacts`.

### Task 4: Full-count GPU point preview

**Files:** Create `flumen/gpu_point_display.py`, `tests/cuda/test_point_stream.py`, `tests/blender/test_gpu_points.py`; modify `flumen/gpu/{state,solver}.py`, `flumen/gpu_runtime.py`, `flumen/gpu_ui.py`.

**Interfaces:** `FlowSolver.point_snapshot(limit: int | None = None) -> PointBatch`; `create_point_display(host: bpy.types.Object) -> None`; `update_point_display(host: bpy.types.Object, batch: PointBatch) -> None`; `release_point_display(host: bpy.types.Object) -> None`. Display settings `point_size=2.0` pixels [1,16], `point_color` RGBA and `display_limit=0` (all) live outside numerical configuration.

- [ ] Write `test_point_stream_compacts_only_xyzr` asserting correct active order/radius, empty shapes, full count, deterministic optional subset and no auxiliary readback. Write `test_point_draw_is_read_only`, `test_point_ownership_and_allocation_failure`: two draws preserve accepted/live/wetness/frame; changing color/size preserves solver identity; no per-frame mesh vertices or sphere instances are created; two hosts and two viewport areas do not share buffers; delete/unregister twice releases owned handlers/batches once, including partial creation failure.

  ```python
  assert batch.xyzr.dtype == np.float32 and batch.xyzr.flags.c_contiguous
  assert batch.xyzr.shape == (batch.displayed_count, 4)
  assert batch.displayed_count == batch.live_count  # limit=None
  assert solver.current_frame == frame_before_draw
  ```

- [ ] Run `CUDA(test_point_stream.py)` and `Blender(test_gpu_points.py)`; expect missing point-stream/display APIs.
- [ ] Implement a fixed GPU vec4 gather plus pinned contiguous copy. Lazily allocate the old full snapshot auxiliary arrays only when requested; point preview copies only `shown*16` bytes. Use Blender's built-in POINT_UNIFORM_COLOR shader initially with a bulk position GPU vertex buffer, clip/depth-aware POST_VIEW handler and per-region view matrix. Confirm the installed 5.2 API signatures by a small probe before coding. Respect host/collection visibility and Blender GPU state restoration. Cache batches by integer-frame identity, never call seek/emission from a draw handler. Expose simulated/displayed counts and separate readback/upload/draw timings. Global handler ownership must survive closed or changed viewport areas.
- [ ] Run focused commands and `Blender(test_gpu_host.py)`; expect PASS. Execute a short actual 1080p million-point draw probe. Measure copies before considering CUDA/graphics interop; only add guarded interop if an identified copy bottleneck requires it, with its own lifecycle tests and recorded supported graphics backend. Commit `feat: draw full GPU particle streams in the viewport`.

### Task 5: Physical reset, preparation reuse and compatible Blender controls

**Files:** Modify `flumen/gpu_runtime.py`, `flumen/gpu_properties.py`, `flumen/gpu_operators.py`, `flumen/gpu_ui.py`, `flumen/gpu/{config,solver}.py`; create `tests/blender/test_gpu_field_host.py`.

**Interfaces:** `physical_key(config: FlowConfig) -> tuple` in `gpu/config.py`; `reset_host(host)` replays with retained preparation when its evaluated fingerprint and field/contact spacing match. `create_gpu_host(source, scene, display_mode='DROPS', solver_backend='LEGACY')` preserves existing calls; UI adds a distinct Field Particle Preview action.

- [ ] Write `test_physical_reset_reuses_preparation`, `test_cosmetic_edits_are_inert`, `test_source_edit_invalidates_contacts`, `test_field_host_lifecycle`. Gravity/rate/time-scale edits replace particle state while preserving preparation identity; changes to world transform/evaluated vertices/triangles/spacing replace preparation. Color/material/visibility/offline quality preserve current frame, ledger and contacts. Same-frame evaluation is inert; backward/skipped seeks replay all intervals. Duplicate/reset/undo/load/delete release only owned references and preserve a second host/user material. Opening old Drops/Connected files keeps their backend and controls unchanged.

  ```python
  assert reset_solver.prepared is original_preparation  # physical edit only
  assert edited_source_solver.prepared is not original_preparation
  assert solver_after_material_edit is solver_before_material_edit
  np.testing.assert_array_equal(ledger_after_material_edit, ledger_before_material_edit)
  ```

- [ ] Run `Blender(test_gpu_field_host.py)`; expect reset-reuse/new-backend/lifecycle failures.
- [ ] Separate physical, static-preparation and display signatures. FIELD physical keys exclude offline reconstruction quality and point appearance; legacy keys retain fields that affect existing topology/physics. Source geometry fingerprint includes evaluated world geometry/winding, face/island convention and preparation version. Refresh emission mask/gravity bounds without rebuilding mesh/contact arrays. Dirty physical edits require explicit Reset/Recompute and reuse compatible preparation. UI labels pairwise backend limits, field/contact effective spacing, fallback/limited counts and optional display subset. Do not silently replace existing file modes; recommend FIELD for new particle editing only after Task 6 evidence. Integrate point cleanup into all existing load/undo/delete/unregister paths.
- [ ] Run the focused command plus all `Blender(test_gpu_*.py)` and `Blender(test_connected_water.py)`; expect PASS. Commit `feat: reuse static contacts across particle tuning`.

### Task 6: Performance and reference-motion evidence before bake work

**Files:** Create `scripts/benchmark_particle_viewport.py`, `examples/create_field_water_demo.py`, `tests/test_particle_preview_report.py`; modify `scripts/connected_viewport.py`, `docs/CONNECTED_WATER_VALIDATION.md`, `docs/IMPLEMENTATION_STATUS.md`; retain reports/stills in `artifacts/`.

**Interfaces:** `validate_particle_preview_report(report: dict) -> list[str]` in new `flumen/gpu/preview_report.py`; benchmark CLI `--distribution attached|free|mixed|dense --capacity 1000000 --warmup 120 --frames 600 --output PATH`. Driver advances one integer interval for each completed measurement draw and never catches up by skipping.

- [ ] Write `test_rejects_capacity_only_or_subset`, `test_rejects_short_wrong_size_and_tail_latency`, `test_rejects_contact_overflow_or_ledger_error`, `test_stages_and_memory_are_distinct`. A valid synthetic report has 600 measured draws after 120 warmup, 1920x1080 throughout, minimum live/displayed 1000000, mean >=30, p95 <=33.3, no skipped intervals, no unresolved contacts, relative ledger <=1e-5. Each single altered field must fail; total-device VRAM cannot substitute for owned allocations. Synthetic report tests do not prove actual performance.

  ```python
  assert validate_particle_preview_report(valid_report) == []
  assert validate_particle_preview_report(subset_report)
  assert validate_particle_preview_report(short_or_overflow_report)
  ```

- [ ] Run `python -m pytest tests/test_particle_preview_report.py -q`; expect absent validator.
- [ ] Implement full-count 1080p native draw timing using the proven driver, safe closed-area cleanup, raw interval samples and separate solver/field/contact/aggregation/readback/upload/draw costs. Record configuration, physical dt, source triangles, counts, field/contact spacing, warmup, GPU/backend/driver, tracked owned arrays and isolated preparation/allocation delta separately from whole-device VRAM. Use lifetime 1000, kill height -10000 and explicit conservative resampling when aggregation reduces count; report split counts and all births/removals. Generate 180-frame stationary drainage captures with opaque points and velocity/height diagnostics. Quantify downhill displacement, concentration contrast/channel persistence, growing drop volumes and detachment events; human visual inspection compares extracted stationary-bust reference frames.
- [ ] Run the focused test, then the four complete viewport distributions sequentially and a separate dense solver stress. Inspect raw counts and timings; record PASS/FAIL for each without extrapolation. If any gate fails, profile the failing stage, make only justified bounded optimizations with regression tests, and repeat affected runs. Never reduce count/disable required physics to manufacture a pass. Record an unmet target explicitly; bake work may continue as an independent deliverable, but FIELD cannot be promoted as meeting that target. Commit `perf: measure million-particle preview and drainage`.

### Task 7: Explicit validated particle cache baking

**Files:** Create `flumen/particle_cache.py`, `flumen/gpu_bake.py`, `tests/test_particle_cache.py`, `tests/blender/test_gpu_particle_bake.py`; modify `flumen/gpu/{state,solver}.py`, `flumen/gpu_operators.py`, `flumen/gpu_ui.py`.

**Interfaces:** `CacheHeader(schema_version: int, source_fingerprint: str, physical_settings: dict, physical_dt: float, start_frame: int, end_frame: int, capacity: int, chart_fingerprint: str)`; `CachedFrame(frame: int, arrays: dict[str,np.ndarray], wetness: np.ndarray, counters: np.ndarray, ledger: np.ndarray, next_id: int)`; `FlowSolver.cache_snapshot() -> CachedFrame`; `CacheWriter(path: Path, header: CacheHeader)`, `.write_static(arrays: dict[str,np.ndarray])`, `.write(frame: CachedFrame)`, `.finish()`, `.cancel()`; `CacheReader(path: Path)`, `.validate(source_fingerprint: str, physical_settings: dict)`, `.read_static() -> dict[str,np.ndarray]`, `.read(frame: int) -> CachedFrame`; `estimate_cache_bytes(header: CacheHeader, static_bytes: int = 0) -> int`. Static arrays are world vertices/triangles/islands, chart nodes/triangles/areas/normals/global-face and barycentric provenance, plus required source attribute arrays.

- [ ] Write `test_cache_roundtrip_and_size`, `test_incomplete_corrupt_and_mismatch_rejected`, `test_no_cuda_required_to_read`. Preserve every active particle's position/velocity/normal/age/volume/state/island/global face/bary/IDs/path/limited flags and all physical accounting exactly; persist immutable world source/chart data once. Assert no pointers/object dtype/pickle, finite arrays and validated lengths/dtypes; reader rejects truncated/missing frames, checksum/source/settings mismatches and incomplete manifest. Cancellation cannot be opened as complete. Writer refuses existing output, reader bounds allocations against header/file size; size estimate includes per-frame arrays and fixed data.
- [ ] Write `test_modal_bake_cancel_and_retry` and `test_editing_writes_no_cache`. Interactive frame evaluation creates no disk frames; baking uses a separate solver with shared immutable preparation and leaves the live host state unchanged; cancel releases it, marks partial output and permits a new destination; success includes every integer frame and restores UI/frame state after failure.

  ```python
  np.testing.assert_array_equal(reader.read(frame).arrays['volume'], snapshot.arrays['volume'])
  assert actual_cache_bytes <= estimate_cache_bytes(header, static_bytes)
  assert live_solver.current_frame == live_frame_before_bake
  # CacheReader(cancelled_path) and corrupt/mismatched cache validation must raise ValueError.
  ```

- [ ] Run `python -m pytest tests/test_particle_cache.py -q` and `Blender(test_gpu_particle_bake.py)`; expect missing cache/bake APIs.
- [ ] Implement versioned JSON manifest plus allow_pickle=False NumPy frame files, SHA256 records, temporary-file atomic replacement and complete marker only after all frames validate. Serialize source/chart arrays and world-unit convention; the Blender bake adapter captures evaluated UV/material/corner data aligned through `SourceMesh.evaluated_triangle_ids` and corrected winding. Physical settings exclude materials/mesh quality. Warn with the computed size estimate in the bake dialog before starting, use an explicit user-chosen destination and bounded one-frame memory. Modal timer supports cancel between frames; no mandatory full-million frame duplication in the editing path. Finished cache is immutable; source changes require a new bake.
- [ ] Run both focused commands plus `Blender(test_bake.py)`; expect PASS. Exercise a short actual million-particle cache bake/read, record time/size/peak memory, and verify no live state change. Commit `feat: bake validated particle caches with cancellation`.

### Task 8: Offline connected meshing and CUDA-free playback

**Files:** Create `flumen/offline_mesher.py`, `flumen/water_types.py`, `flumen/gpu_baked_display.py`, `tests/test_offline_water_mesh.py`, `tests/blender/test_gpu_baked_water.py`; modify `flumen/gpu/state.py`, `flumen/gpu_bake.py`, `flumen/gpu_operators.py`, `flumen/gpu_ui.py`, `flumen/gpu_materials.py`, `examples/create_field_water_demo.py`.

**Interfaces:** `iter_mesh_tiles(reader: CacheReader, frame: int, options: MeshOptions) -> Iterator[WaterGeometry]`; `mesh_cached_frame(reader: CacheReader, frame: int, options: MeshOptions, destination: Path, cancel: Callable[[],bool]) -> dict` returns counts/error/resolution/timing diagnostics; `create_baked_water(cache_path: Path, mesh_path: Path, scene: bpy.types.Scene) -> bpy.types.Object`; `update_baked_water(host: bpy.types.Object, frame: int) -> None`. Neither cache reading, meshing nor finished playback initializes the live CUDA solver.

- [ ] Write `test_refined_film_volume_and_dry_holes`, `test_free_neck_breakup_and_tile_seams`, `test_folded_surface_and_wetness_provenance`, `test_mesh_cancel_and_budget`. Flat patches reconstruct volume within 5%, finer spacing converges; dry regions stay open, separate sheets never bridge. Current touching drops form a neck and separated states remove it without history ghosts; shared tile-boundary keys prevent duplicate surfaces/cracks. Report unresolved/subresolution volume instead of losing it. Halving mesh spacing changes no cached particle/ledger/wetness arrays. Bounded 32³-cell tiles plus halo do not accumulate the whole domain in memory; cancellation never marks a mesh sequence complete.
- [ ] Write `test_baked_playback_without_cuda_and_material_edit` and `test_baked_source_attributes_and_cleanup`. Monkeypatch live CUDA initialization to raise; mesh playback and offline render still work. Material/quality edits preserve the particle cache. Wet proxy maps original global face/bary data to UVs, source material slots and supported point/corner attributes; unsupported attributes are reported. Delete/load releases only baked-owned objects/materials and leaves the source unchanged.

  ```python
  assert abs(mesh_volume - represented_volume) <= represented_volume * 0.05
  assert duplicate_boundary_faces == 0 and cracks_on_tile_boundary == 0
  assert max_live_tile_cells <= 34**3  # 32-cell core plus one-cell halo on each side
  assert cached_frame_hash_after_meshing == cached_frame_hash_before_meshing
  ```

- [ ] Run `python -m pytest tests/test_offline_water_mesh.py -q` and `Blender(test_gpu_baked_water.py)`; expect missing offline APIs.
- [ ] Move only the existing `MeshBatch` and `WaterGeometry` dataclasses to CPU-only `water_types.py`, re-exporting them from `gpu/state.py` to preserve imports; bracket with existing mesh tests. Implement bounded CPU/NumPy tiled meshing from immutable cached particles using original anchors with radius/quality-defined normalized support, independently refined attached charts, and compact anisotropic free-drop fields. Reuse existing volume normalization/clip/marching-tetrahedra mathematics rather than requiring live Warp buffers. Include complete support halos, global vertex/edge keys and deterministic ownership of tile cells; stream indexed arrays to per-frame files with validated complete manifests. Apply contact-sheet barriers to free field support. Report particle volume, represented/excluded/unresolved volume, signed mesh volume/error, spacing, time and peak memory. Bake a retained wet source proxy and current connected geometry; GPU point preview stays separate. Cache mesh frames by integer frame for portable Blender playback/render.
- [ ] Run both focused commands and `Blender(test_connected_water.py)`; expect PASS. Produce and inspect both opaque and water 180-frame offline clips/stills against the supplied stationary reference; record drainage/channel/drop/neck/wetness outcomes and visual limitations. Record actual bake cost; no meshing FPS gate. Commit `feat: mesh cached water offline and play baked frames`.

### Task 9: Whole-increment verification, independent review and package

**Files:** Modify `README.md`, `docs/{IMPLEMENTATION_STATUS,CONNECTED_WATER_VALIDATION}.md`, `flumen/{__init__.py,blender_manifest.toml}`, `pyproject.toml`, `scripts/{smoke_test_blender,smoke_gpu_package}.py`, `tests/blender/test_smoke_contract.py`; add `docs/PARTICLE_PREVIEW_AND_BAKE.md`, update durable checkpoints. Use the existing recursive `scripts/build_extension.py`.

**Interfaces:** `validate_field_output(solver: FlowSolver) -> None` and `validate_baked_output(host: bpy.types.Object) -> None` in `scripts/smoke_test_blender.py` raise AssertionError on an empty/nonconservative result; existing `validate_output(obj)` keeps its legacy contract. The isolated parent smoke runner exercises these against staged sources.

- [ ] Document exact UI workflow: Create Field Particle Preview, emission/radius versus field/contact spacing, Reset/Recompute reuse, counts/limitations, explicit cache bake, independent mesh bake, portable render. Keep old modes documented. Include verified hardware/distribution performance, raw report links, numerical approximation, memory/disk/bake costs and reference comparisons. State any unmet target plainly.
- [ ] Add `test_empty_field_and_baked_output_rejected` and staged-package smoke assertions for FIELD preparation/continuous births/full point batch, cancellation/cache validation, offline mesh playback with live CUDA disabled, legacy Drops/Connected and resource cleanup. Run `Blender(test_smoke_contract.py)` RED before extending existing scripts; expect missing validators. GREEN must reject zero-live field state/empty baked geometry and pass real emitted/reconstructed fixtures.
- [ ] Run `python -m pytest -q`, the full actual CUDA suite and full Blender suite (omit focused selectors). Confirm counts/results in logs, no silent skips, no concurrent GPU jobs. Promote FIELD as recommended only if required numerical, viewport and visual evidence supports it; otherwise deliver an explicitly experimental checkpoint with failed gates visible.
- [ ] Invoke `superpowers:requesting-code-review` and the Native final reviewer required by `superpowers:executing-plans`: one fresh independent reviewer on the most capable available model examines the entire increment from `09f47ac69e3b5caae1930f03327b58e61479bf1e`, approved specs, tests and measured evidence. Existing foundation review does not substitute. Fix confirmed Important/Critical findings in one pass with regressions, rerun affected checks and required full suites; record minor deferrals and all rulings. Do not start a repeat review cycle.
- [ ] Once fixes are committed, synchronize version 0.0.4, stage a fresh package with the pinned Warp wheel, run extension validation and both existing smoke runners against the staged package. Also verify finished mesh playback in clean Blender without live CUDA dependencies. Package locally with SHA256 and reproduction commands; no push/publication. Commit `feat: package particle preview and offline water bake` only after fresh verification.

  ```powershell
  python scripts/build_extension.py artifacts/extension-stage-particles-v4 --wheel-dir .gpu-wheels
  & $FlumenBlender --background --factory-startup --command extension validate artifacts/extension-stage-particles-v4
  & $FlumenBlender --background --factory-startup --python-exit-code 1 --python scripts/smoke_test_blender.py -- --package artifacts/extension-stage-particles-v4
  python scripts/smoke_gpu_package.py artifacts/extension-stage-particles-v4
  ```

- [ ] Reconcile the old and new scratch ledgers into a durable checkpoint with completed/failed gates, reviewer disposition and deferred limitations. Delete only the finished plans' scratch directories after verified commits; if required work remains, preserve scratch and do not mark complete. Provide the user local package, demo/cache/mesh locations and actual measured outcomes.

## Self-review and handoff

Coverage checked against the approved spec: stationary preparation/ownership and subspacing transfer (Task 1); field forces/conservation/wetness (2); local free/attached contact, bounded aggregation/resampling, replay/emission and solver-only probes (3); compact full-count draw and measured copy path (4); physical/cosmetic reset and lifecycle (5); actual per-distribution viewport gates and reference motion (6); explicit immutable/cancellable cache (7); finer offline connected geometry and CUDA-free playback (8); exhaustive evidence, review and package (9). All five Review Focus inputs have named tests in their owning tasks. Shared signatures/settings agree across tasks; no new dependency or unrelated refactor is planned.

Spec approval is recorded. Plan review is pending; the previously selected Native method is preserved. No FIELD implementation or performance success is claimed by this document.
