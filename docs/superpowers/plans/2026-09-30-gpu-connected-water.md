# GPU Connected Water Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bead-only reference demo with connected draining water, merging drops, short live necks and retained wetness, while measuring the complete real-time GPU/Blender cost.

**Architecture:** Extend the bounded CUDA pool with synchronous neighbor interactions and conservative pair merging. Reconstruct attached liquid on a stationary surface proxy and free liquid in bounded local sampling bricks; maintain wetness separately from live liquid. Blender owns controls, generated display objects and bulk updates, while Warp owns time-dependent numerical state.

**Tech Stack:** Windows x64, Blender 5.2, embedded Python, NVIDIA Warp 1.17.0, NumPy supplied by Blender, pytest for pure helpers, unittest for actual CUDA/Blender tests; FFmpeg for captured viewport clips.

**Spec:** [Approved connected-water design](../specs/2026-09-30-gpu-connected-water-design.md). Read this plan and that spec before execution.

## Global Constraints

- Target Windows x64, Blender 5.2, Warp 1.17.0 and the RTX 5050; verify runtime versions during execution.
- Stationary collision surfaces, one Blender unit equals one meter, identity unparented host; retain existing Geometry Nodes workflows and native bakes.
- Preserve old scenes: `display_mode='DROPS'`, `interactions_enabled=False`, `time_scale=1.0`, `initial_coating_count=0` when new properties were not saved.
- Physical interval = `fps_base / fps * time_scale`. Movement, age and drying use that interval; births remain once per eligible integer frame.
- Initial coating counts as emitted volume. Merge consumption does not count as removed volume. Relative volume-ledger error <=1e-5; position replay tolerance 1e-6 m.
- Interactions read the same prior substep state. Maximum 64 shared substeps and 64 accepted neighbors; deterministic selection and reported overflow.
- Starting approximation controls: radius scale 4, cohesion acceleration limit 2 m/s², repulsion acceleration limit 20 m/s², surface damping 10 s⁻¹, merge distance 0.25 times pair-radius sum, maximum merged attached radius 4 times nominal radius.
- Maximum 100,000 proxy vertices, 2,097,152 volumetric samples, 250,000 total liquid output vertices and 500,000 total liquid triangles. Report coarsening, overflow and unrepresented volume; reuse allocations.
- Thickness-volume integral within 5% of represented attached volume. Wetness stays in [0,1] and does not enter the fluid ledger.
- Water material: IOR 1.333, roughness 0.05. Wetness deposition rate 5 s⁻¹ while covered, drying rate 0.1 s⁻¹.
- No CPU solver fallback, installation during registration, per-particle Blender objects, serialized CUDA pointers, or unbounded trajectory geometry.
- Offline GPU rendering/baking and moving/deforming surfaces remain unsupported. Do not advertise the moving-hand reference as matched.
- Visual acceptance uses a fixed-camera 180-frame sequence on an approximately 0.3 m stationary Suzanne with approximately 20,000 evaluated triangles, 8192 slots, 4096 initial coating particles and 64 births/frame.
- Performance acceptance: 600 completed 1920x1080 Eevee viewport draws after 120 warmup frames, connected reconstruction and interactions enabled, >=30 FPS average and p95 <=33.3 ms. Also measure solid geometry separately. No dropped simulated steps.

## Review Focus

1. Sparse or coarse input meshes and narrow reconstruction support must not lose liquid silently or render an unrelated uniform shell (Tasks 2, 5, 6).
2. Close folded regions of the same mesh, duplicate positions and overfull neighborhoods must not form false bridges, race-dependent merges or NaNs (Tasks 3, 4, 7).
3. Multi-island sources must use one face-index convention through births, attachment changes and reconstruction (Task 2).
4. Reloaded 0.0.3 files, duplicate hosts, failed allocations and undo must preserve legacy behavior and release only owned state/objects (Tasks 1, 8, 10).
5. Same-frame redraws, slow time scales, backward/skipped seeks and shader edits must not deposit wetness or emit twice (Tasks 1, 5, 8).

## Files and interfaces

Extend `flumen/gpu/{config,source,state,emission,motion,solver}.py` at their existing responsibilities. New numerical modules: `topology.py` (stationary proxy and surface support), `neighbors.py` (deterministic local queries), `interaction.py` (prior-state forces), `merge.py` (reciprocal ownership), `surface.py` (thickness/wetness), `surface_mesh.py` (attached contour geometry), `free_mesh.py` (local implicit free-water surface). No numerical module imports `bpy`.

New Blender modules: `gpu_water_display.py` (bulk connected meshes and owned wetness proxy) and `gpu_materials.py` (water/wet surface materials). Keep `gpu_display.py` as the Drops baseline. Extend runtime, properties, operators and panel; no general repository refactor.

Shared types, defined by owning tasks:

- Task 1 extends `FlowConfig` with `display_mode`, `interactions_enabled`, `time_scale`, `initial_coating_count`, `interaction_radius_scale`, `cohesion_acceleration`, `repulsion_acceleration`, `surface_damping`, `merge_distance_scale`, `maximum_merged_radius_scale`, `reconstruction_scale`, `wetness_deposit_rate`, `wetness_drying_rate`.
- Task 2 adds CPU world geometry and local-to-global face maps to `SourceMesh`; all stored particle face IDs become global source-mesh IDs. `SurfaceTopology` owns bounded proxy geometry, normals, vertex area weights, original-face mapping, island IDs and connected local-support tables, with `close()`.
- Task 2 extends `DisplayBatch` with normals `(N,3)`, velocities `(N,3)`, volumes `(N,)`, states `(N,)`, global faces `(N,)`, islands `(N,)`; IDs and positions retain their existing meanings. Empty batches have correctly shaped arrays.
- Task 3 `NeighborBuffers` owns fixed `(capacity,64)` partner indices, counts and overflow counters. `InteractionBuffers` owns previous-position/velocity snapshots, forces and shared-step reduction, with `close()`.
- Task 5 `SurfaceState` owns GPU thickness, wetness, deposition normalization and diagnostics for one `SurfaceTopology`, with `close()`. `SurfaceBatch` contains contiguous proxy thickness/wetness arrays and represented/unrepresented attached-volume totals.
- Task 6 `MeshBatch` contains contiguous float32 vertices/normals, int32 triangles and geometry diagnostics. Empty output is valid. `WaterGeometry` contains attached/free `MeshBatch` values and shared counts/budget diagnostics; the combined budgets apply across both.
- Extend `FrameStats` with defaulted fields: `interaction_ms`, `reconstruction_ms`, `neighbor_overflow`, `merged_pairs`, `proxy_vertices`, `water_vertices`, `water_triangles`, `coarsening_factor`, `unrepresented_volume`, `rendered_volume_error`. Defaults preserve old callers/report readers.
- `FlowSolver` owns optional topology, interaction and surface resources; retains `seek()`, `snapshot()`, `reset()`, `close()`. Add `surface_snapshot() -> SurfaceBatch | None` and `water_snapshot() -> WaterGeometry | None`, both read-only for simulation time.

## Execution and evidence

Preserve the already selected **Native** execution method: implement tasks sequentially in the isolated workspace, then obtain one independent whole-branch review and fix confirmed Important/Critical findings. Do not dispatch per-task implementation agents. Reuse the current worktree only after the worktree skill verifies its clean state and branch ownership; keep the user's PDF/video and extracted reference frames untouched.

Before code changes, run the baseline suites and record the committed base. Each task has a failing test, implementation, passing focused tests and a commit. CUDA failure cannot be reported as a skipped success. Pure tests: `python -m pytest -q`.

CUDA command (append `-- 'test_filename.py'` for a focused file):

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/run_cuda_tests.py
```

Blender command (append `-- --pattern 'test_filename.py'`):

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe' --background --factory-startup --python-exit-code 1 --python scripts/run_blender_tests.py
```

### Task 1: Compatible controls, coating births and physical time

**Files:** Modify `flumen/gpu/{config,emission,solver}.py`, `flumen/gpu_properties.py`; create `tests/test_gpu_connected_config.py`, `tests/cuda/test_connected_emission.py`; extend `tests/blender/test_gpu_settings.py`.

**Interfaces:** `emit_batch(pool, source, config, frame, requested: int) -> None` owns the existing bounded allocation; `emit()` retains its signature and uses this helper. Initial coating is a separate batch invoked once at the simulation start, before that frame's normal emission. Use the same source mask, deterministic ID stream and volume ledger for both. `FlowSolver.dt` stores the scaled physical interval.

- [ ] Write `test_legacy_defaults` asserting Drops/disabled/1.0/0; `test_connected_control_validation` covers Boolean type, modes, nonfinite values, invalid scales/counts and exact Blender endpoint round trips. Bounds: time scale [0.01,2], initial count [0,2**31-1], interaction radius scale [1,16], acceleration limits [0,100], damping/deposit/drying [0,1000], merge distance scale [0,1], merged radius scale [1,8], reconstruction scale [0.5,4]. Reject bool counts; require actual bool for `interactions_enabled`.
- [ ] Write `test_coating_and_frame_births`: capacity 32, full mask, coating 5 and rate 3 yield accepted/live 8,11,14 at frames 1,2,3. Repeating frame 3 is inert; seeking 0 then 3 reproduces IDs/positions. A coating request 10**9 at capacity 5 stays capacity-bounded and records rejection without backlog.
- [ ] Write `test_time_scale_uses_physical_seconds`: a surviving first batch ages `2*1.001/30*0.5` at frame 3; counts equal the 1.0 scale run. Verify freefall at equal physical elapsed time. Extend RNA endpoint tests to new Float/Bool/Enum properties.
- [ ] Run the new pure/CUDA/settings tests and record missing-setting/behavior failures.
- [ ] Implement the controls, strict validation and Boolean/enum RNA handling without relaxing existing float-endpoint normalization. Keep display mode changes reset-driven; shader/material edits remain outside numerical config.
- [ ] Rerun focused tests and old emission/solver tests; commit `feat: add coating emission and GPU time controls`.

### Task 2: Global anchors and bounded surface topology

**Files:** Modify `flumen/gpu/{source,motion,state}.py`; create `flumen/gpu/topology.py`, `tests/cuda/test_surface_topology.py`; extend `tests/blender/test_gpu_source.py`.

**Interfaces:** `build_topology(source: SourceMesh, support_radius: float, target_edge: float, vertex_budget=100000) -> SurfaceTopology`. Preserve world geometry after degenerate-face removal; store concatenated island local-to-global maps and offsets. `snapshot()` retains the three old display fields and adds the new fields defined above.

- [ ] Write `test_global_face_ids_survive_island_projection_and_reattachment` on a two-island mesh whose second island has different local/global face indices. Sample, move and recapture a particle; its face/barycentric anchor must evaluate on the correct global triangle.
- [ ] Write `test_topology_support_rejects_fold_shortcut`: use close sheets joined by a distant strip (one island); support cannot cross the gap. Also cover mirrored winding, disconnected sheets, coarse triangles and all-degenerate rejection.
- [ ] Write `test_proxy_budget_and_area`: deterministic subdivision toward `target_edge=1.5*radius*reconstruction_scale`, <=100000 vertices, finite unit normals/positive area weights, total area within 1% of source, reported effective resolution/coarsening. A coarse proxy or unsupported footprint must produce diagnostic volume rather than invented support.
- [ ] Run new topology/source tests before implementation.
- [ ] Build CPU adjacency once at reset; refine conservatively within budget. Local support uses connected surface neighborhoods, normal compatibility and bounded geodesic checks, not only island/Euclidean distance. Publish GPU support tables. Correct every write of `data.face` to global IDs using the local-to-global map. No position/motion rewrite in the Drops path.
- [ ] Verify nonempty and empty extended snapshots have aligned IDs/state/normal/volume arrays; rerun old motion/source tests; commit `feat: preserve surface anchors for connected water`.

### Task 3: Deterministic neighborhoods and bounded forces

**Files:** Create `flumen/gpu/neighbors.py`, `interaction.py`, `tests/cuda/connected_fixtures.py`, `tests/cuda/test_interaction.py`; extend pool diagnostics/storage in `state.py`.

**Interfaces:** `build_neighbors(pool, source, topology, config, buffers) -> None`; `compute_forces(pool, config, neighbors, scratch) -> None`. Test helper `make_fixture(vertices, triangles, positions, states=None, volumes=None, velocities=None, config=None) -> tuple[SourceMesh, ParticlePool, DeviceInfo]` creates real CUDA fixtures; each test closes owned resources.

- [ ] Write `test_prior_state_symmetry`: two equal-volume attached particles at separations 0.5,1.5 and 3 nominal radii have finite equal/opposite interaction accelerations, separate at overlap and attract at short range. Force magnitudes respect 2 m/s² cohesion and 20 m/s² repulsion caps. Damping reduces tangent relative speed without normal acceleration.
- [ ] Write `test_coincident_and_overfull_neighbors`: coincident coordinates never divide by zero; 70 eligible neighbors select the nearest 64 ordered by `(distance, particle_id)` regardless of slot permutation, and report overflow. Inactive/foreign-state/foreign-surface particles are excluded.
- [ ] Write `test_same_island_fold_and_back_side_are_not_neighbors`, using Task 2's connected-fold topology. Zero strengths yield no added force; no force changes volume or active IDs.
- [ ] Run `test_interaction.py` to observe failure.
- [ ] Use Warp `HashGrid` for candidate lookup and fixed neighbor storage. Apply connectivity/visibility checks before deterministic nearest selection. Read snapshots only; accumulate tangent cohesion, pressure/spacing repulsion and damping into a separate force array. Do not mutate neighbor velocities while computing forces.
- [ ] Rerun interaction/topology tests; commit `feat: add bounded GPU surface interactions`.

### Task 4: Reciprocal merging and synchronized integration

**Files:** Create `flumen/gpu/merge.py`, `tests/cuda/test_connected_motion.py`, `tests/cuda/test_merge.py`; modify `motion.py`, `solver.py`, `state.py`.

**Interfaces:** `merge_pairs(pool, source, topology, config, neighbors) -> None`; `advance_coupled(pool, source, topology, config, scratch, dt: float) -> None`. Add a forced single-substep path to the integration kernel; keep `advance()` unchanged for disabled interactions. Merge proposal/commit kernels use the same pre-merge state.

- [ ] Write `test_reciprocal_merge_volume_and_momentum`: positions separated by less than 0.25 times summed radii merge exactly once; lower particle ID survives, volume sums, velocity is volume-weighted, removed ledger does not increase. A nonreciprocal three-particle chain cannot consume a particle twice. Attachment-state/surface mismatches and oversized merged drops remain separate.
- [ ] Write `test_synchronous_steps_and_zero_dt`: dt=0 changes nothing; all particles see identical prior-state boundaries; one frame advances age exactly once despite detach/reattach. Shared step reduction includes gravity plus capped interaction acceleration and clamps at 64 with travel diagnostics.
- [ ] Write replay tests with interactions, merges, emission and slot reuse: forward/skipped/backward evaluation agrees within 1e-6 m and ledger relative error <=1e-5. Neighbor slot permutation cannot change partner ownership.
- [ ] Run merge/connected-motion tests to capture failure.
- [ ] Per shared substep, snapshot, query neighbors, compute/apply bounded forces, integrate one common physical interval, then select/commit reciprocal merges. Project an attached merged anchor onto the same permitted surface; reject the merge if projection fails. Use max source age for the survivor so merging cannot extend lifetime indefinitely. Initialize/terminate obsolete display links by surviving identity, not reused slot.
- [ ] Rerun old and new CUDA motion/emission/solver tests; commit `feat: integrate conservative connected GPU water`.

### Task 5: Conservative attached thickness and independent wetness

**Files:** Create `flumen/gpu/surface.py`, `tests/cuda/test_surface_fields.py`; modify `solver.py`, `state.py`.

**Interfaces:** `build_surface(topology, config, device) -> SurfaceState`; `update_surface(surface, pool, source, config, dt: float) -> None`; `snapshot_surface(surface) -> SurfaceBatch`. Solver updates thickness after completed frame integration/births; wetness advances once per completed physical interval, never during snapshots/redraws.

- [ ] Write `test_normalized_thickness_volume`: deposit attached volumes onto a flat supported proxy; `sum(thickness*vertex_area)` matches represented volume within 5%. Foreign/free particles deposit nothing. Unsupported particles contribute to `unrepresented_volume`; all arrays remain finite and nonnegative.
- [ ] Write `test_sheet_support_and_dry_regions`: dense adjacent particles give connected nonzero coverage; an uncovered region stays zero. Same-island folds and back faces do not receive deposits.
- [ ] Write `test_wetness_seconds_and_replay`: coverage with deposit rate 5 s⁻¹ raises wetness toward one; no coverage with drying 0.1 s⁻¹ lowers it. Equivalent physical time at 24/48 FPS and time scales 0.05/0.5/1 agrees within 1e-5. Same-frame seek/readback changes nothing; reset/backward replay reconstructs thickness/wetness.
- [ ] Run new tests before implementation.
- [ ] Normalize each particle's topology-supported kernel weights using vertex area; deposit only live volume into the thickness field. Wetness uses exact exponential deposit/dry updates in physical seconds, bounded [0,1], outside the liquid ledger. Initial thickness is available at time zero; initial wetness is zero until an interval elapses. Keep bounded support buffers and report footprint overflow.
- [ ] Rerun surface/solver tests, including retained allocation sizes across 10,000 zero-emission updates; commit `feat: reconstruct attached water thickness and wetness`.

### Task 6: Connected attached geometry with real boundaries

**Files:** Create `flumen/gpu/surface_mesh.py`, `tests/cuda/test_surface_mesh.py`; extend `state.py` for `MeshBatch`/`WaterGeometry` and `solver.py` readback.

**Interfaces:** `build_attached_mesh(topology, surface, config, geometry_buffers) -> MeshBatch`; reusable geometry buffers own capped GPU output and count/scan arrays. Surface contour threshold is 1e-5 m thickness; measure/report the volume excluded by that threshold.

- [ ] Write `test_connected_patch_and_dry_hole`: a prescribed thickness patch produces a connected raised surface with a boundary skirt, zero dry-region faces, finite smooth normals and no duplicate solid/source output. A dry hole stays open rather than being bridged by a full shell.
- [ ] Write `test_volume_and_locality`: closed patch mesh signed volume agrees with represented thickness volume within 5% on a flat fixture; close unrelated surfaces never gain connecting triangles. Curved fixtures remain on the supported side of the source.
- [ ] Write `test_mesh_budgets_are_explicit`: count before writing; combined output must stay <=250000 vertices/500000 triangles. Reduced available budgets return a diagnostic instead of silently partial geometry or stale success. Empty/expired thickness yields empty liquid geometry.
- [ ] Run surface-mesh tests before implementation.
- [ ] Clip proxy triangles at the thickness contour, construct the raised top and appropriate boundary/base closure, interpolate normals, and use count/scan/write CUDA passes. Share boundary samples consistently across adjacent triangles to avoid cracks. Geometry is derived output and never changes particle volume or motion.
- [ ] Rerun field/mesh tests and inspect a neutral patch preview; commit `feat: display connected surface-water geometry`.

### Task 7: Smooth free drops and current-state neck breakup

**Files:** Create `flumen/gpu/free_mesh.py`, `tests/cuda/test_free_mesh.py`; modify solver geometry assembly and geometry diagnostics.

**Interfaces:** `build_free_mesh(pool, source, config, available_vertex_budget: int, available_triangle_budget: int, buffers) -> MeshBatch`. Attached/free output shares Task 6's global budget; no retained path histories. `water_snapshot()` rebuilds geometry only when numerical state or reconstruction controls change.

- [ ] Write `test_isolated_drop_and_live_neck`: an isolated free particle gives a closed smooth surface; nearby aligned current particles give one connected neck. Separating beyond support or expiring a particle removes that neck on the next evaluation. No old trajectory remains visible.
- [ ] Write `test_elongation_is_bounded`: velocity-derived anisotropy has aspect ratio <=4, finite zero-velocity behavior and volume-normalized kernels. A separated distant cloud does not allocate the entire intervening AABB.
- [ ] Write `test_sampling_and_output_caps`: total volumetric samples <=2097152 and output respects remaining global mesh budgets. Record actual resolution, coarsening and reconstructed signed-volume error; count-pass overflow cannot produce silent partial output. Include coincident particles and a close solid barrier.
- [ ] Run free-mesh tests to capture failure.
- [ ] Build local sampling bricks only around particle support, with halo samples evaluated from a shared field. Use stable int64 brick keys, Warp radix sorting/run-length encoding and bounded prefix scans. Start with 4x4x4 cells per brick (5x5x5 samples); cap at 16777 bricks so samples remain <=2097152. Duplicate halo values agree at shared coordinates. Validate key-domain range explicitly.
- [ ] Evaluate a normalized compact anisotropic field from current free particles. Start sampling pitch at `0.5*nominal_radius*reconstruction_scale`. Use kernel `(1-q²)^3` for q<1, support radius twice the volume-derived particle radius, and volume-preserving anisotropy along velocity: axial scale in [1,4], transverse scales `1/sqrt(axial_scale)`. With kernel normalization `315/(64*pi*support_radius³)`, isovalue 0.3460693359375 yields the nominal isolated spherical radius in the continuous field. Use a fixed marching-tetrahedra decomposition with consistent winding/gradient normals; each brick owns only its 4x4x4 cells. Solid proximity clips unsupported crossing necks. Increase sample pitch by factors of two when count passes exceed budgets, report coarsening, and stop with an explicit geometry error after four attempts. No CPU particle solver or blanket filled-domain allocation.
- [ ] Rerun attached/free mesh tests, including budget sharing and full expiry; commit `feat: reconstruct free drops and breaking necks`.

### Task 8: Blender connected-water output, controls and lifecycle

**Files:** Create `flumen/gpu_water_display.py`, `gpu_materials.py`, `tests/blender/test_connected_water.py`; modify `gpu_runtime.py`, `gpu_properties.py`, `gpu_operators.py`, `gpu_ui.py`, and registration only as required.

**Interfaces:** `create_water_display(host, topology) -> None`; `update_water_display(host, geometry: WaterGeometry, fields: SurfaceBatch) -> None`; `release_water_display(host) -> None`; `create_water_material() -> bpy.types.Material`. Extend `create_gpu_host(source, scene, display_mode='DROPS')` so old callers retain their behavior; the UI operator requests Connected mode and enables interactions explicitly.

- [ ] Write `test_legacy_and_connected_hosts`: a default old/API host remains Drops/disabled; Connected creates one owned liquid display and one owned wetness proxy, with no per-particle objects. Source mesh/material is byte/value unchanged. Bulk updates yield visible connected geometry and proper attributes.
- [ ] Write `test_duplicate_undo_delete_and_failure`: duplicate hosts own separate buffers/meshes/wetness; reset/load/undo/delete/unregister releases owned resources and does not delete source/user objects. Inject one allocation/construction failure and verify recovery leaves no orphan owned objects/runtimes.
- [ ] Write `test_shader_edits_and_same_frame_are_inert`: material/color changes preserve solver, wetness and accepted births; repeated draws deposit nothing. Physics/mode/proxy-resolution/wetness controls require Reset and display a clear reason. Reconstruction overflow/coarsening is visible in diagnostics, not only console output.
- [ ] Run focused Blender tests before implementation.
- [ ] Generate host-owned connected mesh and wetness proxy with source geometry copied once. Use `foreach_set` for frame updates and persistent smooth material nodes; no source-to-host feedback dependency. IOR=1.333, roughness=0.05, with wetness mapped to roughness/darkening independently. Preserve explicit live-preview-only render rejection.
- [ ] Rerun all Blender GPU tests and the pure settings suite; commit `feat: expose connected GPU water in Blender`.

### Task 9: Reference sequence, visual evidence and full-frame performance

**Files:** Create `examples/create_connected_water_demo.py`, `scripts/capture_connected_water.py`, `scripts/benchmark_connected_water.py`, `docs/CONNECTED_WATER_VALIDATION.md`; modify `scripts/gpu_report.py` and `tests/test_gpu_report.py` to identify the rendered mode/engine and reject wrong-fixture acceptance.

**Interfaces:** `create_connected_demo(view='WATER', emission_end=120) -> tuple[scene, host]`; geometry view uses a neutral opaque material without changing numerical state. Capture supports `--view WATER|GEOMETRY --frames 180 --output PATH`. Benchmark supports `--view WATER|GEOMETRY --output PATH` with 120 warmup and 600 measured draws; reuse the established window-size/framebuffer synchronization approach.

- [ ] Write report tests: a Drops/solid/sphere fixture cannot pass Connected/Eevee acceptance; missing completed draws, wrong 1920x1080 dimensions or skipped simulation steps fail. A correctly labeled 600-draw report passes only with mean >=30 FPS and p95 <=33.3 ms.
- [ ] Run report tests before adding the new acceptance logic.
- [ ] Create the ~0.3 m Suzanne fixture with ~20000 evaluated triangles, 8192 slots, coating 4096, rate 64/frame, time scale 0.5, fixed camera, dark background and large reflection lights. Emission stops at frame 120 for the 180-frame drainage/wetness clip; the separate benchmark emits through frame 720. Save all tuned controls and actual source/particle/mesh counts.
- [ ] Capture each completed viewport frame after drawing; use FFmpeg to encode 180-frame water and neutral-geometry MP4s. Capture frames 1/30/60/90/120/180 and compare to the supplied bust frames. Do not use offline unbaked rendering or claim differing source geometry is a pixel match.
- [ ] Inspect both clips for all five approved visual criteria: draining connected coating, persistent uneven channels, merging plus narrowing/broken necks, retained/decaying wetness, and connected geometry in both material views. Record passing evidence by frame/time. A bead field, static painted coating or historical tube output fails this gate even if tests pass.
- [ ] Run the 600-draw Eevee benchmark, then a separate solid geometry benchmark without concurrent GPU jobs. Include synchronized solver/interaction/reconstruction/transfer/update/draw cost, p95/max/mean, dropped-frame count, volume error, live count, output resolution and memory. Profile a missed budget at the dominant stage; repeat only after a concrete change. Preserve actual measurements even if the real-time gate fails.
- [ ] Commit the demo generator, capture/benchmark scripts, reports and comparison stills. Keep large MP4/demo ZIP deliverables out of source history; link their local paths in the validation record. Commit `perf: validate connected water against the references`.

### Task 10: Reviewed package and delivery

**Files:** Modify `flumen/__init__.py`, `flumen/blender_manifest.toml`, `pyproject.toml`, `.gitignore`, `README.md`, `docs/GPU_SETUP.md`, `docs/IMPLEMENTATION_STATUS.md`, `scripts/smoke_test_blender.py`, `scripts/smoke_gpu_package.py`; keep the verified Warp wheel/license/hash unchanged. Add package tests only for new meaningful connected-mode smoke behavior. Ignore the new ZIP, MP4/capture-frame folders and saved demo paths while retaining generators, reports and selected stills in source history.

**Interfaces:** Extension version 0.0.4; recursive builder remains `build_extension(stage_dir, wheel_dir=None)`. Smoke test imports staged sources plus the extracted pinned wheel, creates Connected water, advances frames and verifies real nonempty water geometry and a balanced ledger.

- [ ] Add a staged Connected smoke assertion and record its expected failure against a 0.0.3 stage; preserve existing legacy/Drops smoke checks. Add a save/reload old-property fixture confirming unsaved new settings resolve to legacy defaults and no source mutation.
- [ ] Update version/docs to the actual final behavior and measured results, including any failed visual/FPS gate. Do not label the overall reference goal complete unless all visual and performance criteria passed.
- [ ] Run full pure Python, actual CUDA and full Blender suites on the final source; capture counts and output. Build a fresh wheel-bundled stage, validate with Blender, run staged smoke, build ZIP and verify every Python entry byte-matches canonical source. Deliver `flumen-0.0.4.zip`, saved demo, both 180-frame preview clips and validation instructions.
- [ ] Obtain one independent whole-branch review with the five Review Focus areas, approved spec/plan and execution evidence. Reviewer must distinguish confirmed bugs from GPU/performance judgments they did not rerun. Reproduce and fix Important/Critical findings in one focused pass, then rerun affected suites and rebuild/check the package.
- [ ] Commit final fixes/evidence, preserve the execution rulings and deferred minors, and use the finishing skill for integration. The user previously chose commit/push for the foundation; do not infer authorization to publish a new release or merge the visual increment.

## Self-review and handoff

All approved spec areas have owners: compatibility/emission/time (1), anchors/locality/proxy (2), forces/bounds (3), merging/common time (4), volume/thickness/wetness (5), attached/free surfaces and budgets (6–7), Blender lifecycle/materials (8), visual/full-frame evidence (9), packaging/review (10). Each Review Focus line has named fixtures in its owning tasks.

The implementation remains unstarted until the user reviews this plan. Preserve Native execution once the plan is accepted; there is no need to choose the method again.
