# GPU surface flow and continuous emission

Status: approved by the user on 2026-09-30; implementation planning authorized.

## Outcome and scope

The user wants Flumen to approach `m2-vp9-res_1080p.mp4`, run in real time using a GPU solver, and offer continuous particle emission by frame. They accepted the recommendation to use NVIDIA Warp/CUDA with Blender for controls and display.

The reference shows connected surface water, rivulets, detached drops, and a moving hand. Faster isolated particles alone do not satisfy that visual goal. Deliver the work in two increments: (1) a measured GPU particle foundation with continuous emission; (2) connected surface-water appearance and interactions. This document specifies increment 1 and the data boundaries needed by increment 2. It does not declare the overall visual goal complete after increment 1.

Working scope: stationary collision meshes first, following the recommended staged approach. Moving/deforming surfaces are a later subsystem requiring surface correspondence and inherited velocity. Target hardware is the detected Windows machine with RTX 5050, 8151 MiB reported VRAM, driver 591.86, and the project's Blender 5.2 environment.

## Architecture

Add a separate GPU Flow workflow alongside the existing static and Geometry Nodes workflows. Existing node groups, socket identifiers, and native bakes retain their current behavior. A new host owns GPU settings and a runtime identifier; it does not masquerade as a Geometry Nodes simulation cache.

Data flow:

```text
Blender evaluated stationary mesh -> triangulation + source distribution
                                  -> Warp collision mesh on CUDA
Blender timeline + emitter settings -> bounded GPU particle pool
                                   -> attached/free integration + collision
                                   -> compact display attributes
                                   -> Blender point mesh + instanced drops
```

Separate responsibilities:

- `flumen/gpu/`: device initialization, particle state, emission, kernels, collision, and metrics. Numerical code does not import `bpy`.
- Blender adapter: evaluated mesh extraction, host properties, frame coordination, display transfer, reset, and runtime cleanup.
- UI/operators: create GPU Flow, reset, emission controls, and device/timing diagnostics.
- Benchmark and tests: numerical checks, actual CUDA execution, Blender lifecycle checks, and measured playback.

Warp imports are lazy. A missing dependency or unavailable CUDA device produces an actionable GPU Flow error while leaving existing Flumen workflows usable. GPU Flow must not silently run on CPU. Pin a stable Warp wheel compatible with Blender's embedded Python and the installed GPU driver after testing; do not install packages automatically during add-on registration. Document an explicit development setup and package the tested Windows wheel for distribution, with applicable license notices and extension validation.

## Particle state and motion

Allocate a fixed-capacity GPU pool containing position, velocity, volume, age, active flag, attached/free state, normal, collision face/barycentric anchor, surface island, particle ID, and path ID. Slots may be reused; particle IDs may not be reused during a run. Keep counters and emitted/removed volume separately from slots.

Start with the current motion model: tangent gravity and analytic linear drag while attached, adhesion-controlled detachment, ballistic free flight, collision clearance, and speed-controlled reattachment. Use Warp mesh queries against a persistent collision acceleration structure, built once per reset. Preserve island/locality checks so particles do not jump across disconnected close surfaces. Thin sheets and grazing contact require explicit collision fixtures; nearest-point projection alone is insufficient.

Choose bounded adaptive substeps from actual frame duration and travel limits. Integrate each particle once per substep even when its state changes. Report travel limiting and actual substeps; do not hide instability by silently slowing simulated time. Preserve the current meter-based world-coordinate convention and identity host transform requirement.

Increment 1 does not add particle merging or a physical thin-film solver. Keeping volume, anchors, and path identity explicit allows subsequent cohesion, merging, and surface appearance without relying on display geometry as simulation state.

## Continuous emission contract

Expose mode (`Burst` or `Continuous`), burst count, particles per frame, emission start/end frames (inclusive), seed, source height mask, drop radius, lifetime in seconds, kill height, and maximum live particles. Use the collision surface as emitter in this increment, reusing the existing height-mask concept. Defer a separate emitter object.

- On the simulation start frame, initialize state at time zero and emit the eligible batch without advancing it.
- For each subsequent integer frame, advance existing particles through the elapsed frame interval, retire expired/out-of-bounds particles, then emit that frame's batch at age zero. New particles begin moving during the next interval.
- Re-evaluating the same frame never emits or integrates twice. Substeps do not multiply the emission count.
- Burst emits only at the configured emission start frame; continuous mode emits on every eligible integer frame. Zero rate emits nothing.
- Sample triangles by eligible surface area with barycentric positions and deterministic seed/frame/ID inputs. Use bounded sampling attempts; an empty or very narrow mask reports fewer accepted particles instead of hanging or allocating unbounded candidates.
- Reuse retired slots first. If capacity is full, drop excess requested births and count them; do not replace live particles or accumulate a deferred burst.
- Record requested, accepted, and capacity-rejected births separately from source-sampling rejection. Only accepted births contribute to emitted volume.
- Require `initial volume + emitted volume = live volume + removed volume` within documented floating-point tolerance. Define initial volume as zero for the new backend and count all accepted births, including the first batch, as emitted volume.

Particles per frame intentionally changes particles per second when scene FPS changes. Physics still uses seconds, with `dt = fps_base / fps`. FPS or state-driving setting changes invalidate the run and require reset; material edits do not.

Proposed defaults: continuous mode, 64 particles/frame, 8192 maximum live particles, 4-second lifetime, scene frame range for emission, and the existing gravity/radius/source-mask defaults. These are initial settings, not a performance claim.

## Blender lifecycle and display

Keep one runtime per host. Live playback processes every integer frame, including skipped frames, so viewport scheduling cannot change emission counts. Backward seeks reset and replay deterministically from the simulation start. Seeking is potentially slower than sequential playback and is reported separately. Uncached subframe evaluation is unsupported in this increment and must report that limitation rather than alter state inconsistently.

Changing collision geometry/transforms invalidates the stationary snapshot. Use dependency-graph change notifications plus a snapshot check at reset; avoid rebuilding the entire collision mesh on every frame. Guard callbacks against recursive host updates. Host deletion, undo, file loading, and add-on unregister release corresponding runtimes and handlers. Reloaded files reconstruct state through reset/replay; CUDA pointers are never serialized.

Transfer compact active particle positions, radii, and IDs once per displayed frame, using bulk mesh updates. Start with an ordinary point mesh and instanced low-resolution drops so viewport and Blender rendering share visible output. Avoid per-particle Python object creation and realized sphere meshes in the update loop. Include GPU synchronization, GPU-to-CPU transfer, mesh updates, and viewport redraw in measurements; zero-copy graphics interop is not assumed.

Live GPU playback requires the add-on and tested Warp dependency. Portable baked export is a subsequent delivery, separate from existing native Geometry Nodes bakes. Reject unsupported out-of-order/background rendering paths clearly; sequential live preview is the supported acceptance workflow for increment 1.

## Performance and acceptance

Define real time as at least 30 FPS sequential viewport playback after initialization/JIT, with full frame time p95 at or below 33.3 ms for the acceptance fixture. Use a 1920x1080 solid viewport, a stationary approximately 20,000-triangle sphere, 30 FPS scene timing, 64 births/frame, 4-second lifetime, 8192-slot capacity, and a 600-frame run after compilation. Disable playback frame dropping and confirm simulated frame count and emitted volume, so dropped simulation steps cannot manufacture a pass.

Measure synchronized solver time, transfer time, Blender update time, total viewport frame time, live count, actual substeps, volume error, and memory. Report compilation/setup separately and include median, p95, and maximum. Background evaluation timing alone cannot establish viewport FPS. Existing repository timings are historical baseline evidence, not new measurements on this hardware.

Also exercise 2048 and 32768 capacity settings and a sustained capacity-saturation run. Those characterize scaling rather than impose an unsupported universal FPS guarantee. If the acceptance fixture misses its budget, profile the dominant stage and report the measured limit before adding the visual increment; do not relabel simulation-only timing as real-time playback.

Required correctness evidence:

- Actual CUDA device execution and clear missing-device/dependency failures.
- Attached motion, free fall, detachment, reattachment, thin-sheet locality, finite state, and equivalent physical elapsed time at 24/48 FPS.
- Exact eligible-frame birth counts when source/capacity allow; inclusive emission boundaries; zero rate; repeated-frame idempotence; capacity recycling; unique IDs; bounded allocation; deterministic reset/replay within numerical tolerance.
- Emission-aware volume accounting and no doubled integration on transitions.
- Two independent hosts; settings invalidation; source edits; deletion/undo/load/unregister cleanup; background import of pure numerical modules.
- Existing Python and Blender regression suites remain passing; packaged extension validates and smoke-tests with its dependency.

## Follow-on visual increment

After the foundation passes, design bounded neighbor interactions and volume-conserving merging, connected rivulet display, and a surface water/wetness representation. Compare fixed-camera sequences against the stationary bust portion of the reference. A wetness shader alone does not represent flowing liquid thickness. Measure its additional display and interaction costs within the same total frame budget. Moving-hand matching remains a separate surface-transport milestone.

## Decision record and references

Warp/CUDA was selected because it supplies GPU kernels and mesh-query primitives while allowing Blender to remain the authoring environment. Extending only the existing graph does not supply the requested dedicated GPU backend; raw Blender compute shaders would require more custom collision infrastructure. Compatibility and performance remain execution gates, not promises based solely on framework capability.

- Local reference: `m2-vp9-res_1080p.mp4`; sampled frames: `artifacts/reference-contact-sheet.jpg`.
- Current baseline: `docs/IMPLEMENTATION_STATUS.md`, `flumen/build_simulation.py`, `flumen/sim_source.py`, `flumen/sim_output.py`.
- NVIDIA Warp: https://github.com/NVIDIA/warp
- Warp mesh collision example: https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_mesh.py

Written-spec approval was received on 2026-09-30. The implementation plan is the next workflow checkpoint.
