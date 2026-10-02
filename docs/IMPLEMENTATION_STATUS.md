# Implementation status — connected development and legacy releases

## Connected development checkpoint, 2026-10-02

The development branch now has GPU surface interactions, conservative merging,
initial coating/time controls, attached/free mesh reconstruction and persistent
wetness. Both 180-frame clips and comparison stills exist, but the visual
reference gate still fails: coating/channels are too coarse and patchy.

The user now permits offline final meshing and requires interactive particle
simulation at one million particles. The Surface Field preview meets the
1080p viewport target on the RTX 5050: attached, free, mixed and dense runs
each complete 120 warmup plus 600 measured full-count draws at 34-47 FPS with
p95 23-31 ms (attached has about 2 ms margin; background GPU apps must be
closed). The visual drainage comparison against the reference fails: the
coating drains far too slowly and forms no distinct rivulets.
The revised [design](superpowers/specs/2026-10-01-million-particle-preview-design.md)
and plan are approved. Tasks 1-6 (preparation, field dynamics, transport,
point drawing, preparation reuse, measured gates) are complete; drip release,
thin-film wall drag and fixed field substeps were added by user ruling. The
particle cache and offline baking (Tasks 7-9) remain unfinished. The
[saved checkpoint](superpowers/checkpoints/2026-10-01-particle-scale.md) records
the resume steps. No version 0.0.4 package or release is delivered.

Verified checkpoint: 90 Python, 80 CUDA and 68 Blender tests. Raw results,
reproduction and limitations are in
[Connected validation](CONNECTED_WATER_VALIDATION.md). The earlier 0.0.3
measurements below are historical Drops evidence, not proof for this increment.

## GPU Flow, 2026-09-30

The separate Warp 1.17.0 CUDA backend provides stationary surface particles,
continuous/burst emission, bounded slot recycling, analytic tangent drag,
detachment, approximate ray/proximity collision, deterministic integer-frame
replay, volume accounting, and Blender instanced preview output.

Verified on Blender 5.2.0 LTS, embedded Python 3.13.13, NVIDIA RTX 5050
(8151 MiB), driver 591.86. Warp reports CUDA Toolkit 12.9 / driver API 13.1.

The actual 1920x1080 solid viewport run completed 600 sequential draws at
**56.59 FPS average, 16.70 ms median, 25.02 ms p95**. It used 8192 slots,
64 births/frame, 4-second lifetime and a 102-segment/100-ring sphere. Across
frames 1-720 (120 warmup + 600 measured), all 46,080 requested births were
accepted; the final state had 7,743 live particles, no travel-limited particles,
and zero observed volume-ledger discrepancy. Peak process working set was
672 MiB. Whole-device VRAM usage was 6211 MiB including other applications;
this is not the solver's isolated allocation.

The measurement waits for each POST_PIXEL draw and synchronizes graphics using
a framebuffer readback before requesting the next frame. CUDA is synchronized
before display transfer. Wall-clock throughput includes scheduling and redraw;
initialization/JIT is excluded. Raw samples, exact controls and hardware metadata
are in `artifacts/benchmark-gpu-viewport.json`. Background-only scaling results
are in `artifacts/benchmark-gpu-{2048,8192,32768}.json` and never count as proof
of viewport FPS. The 30 FPS acceptance gate passed for this fixture, not for
arbitrary meshes, shaders, or particle counts.

Final verification: 38 Python tests, 17 actual CUDA tests, and 52 Blender tests
passed. The independent review identified one UI endpoint rounding bug; a
Blender regression reproduced seven affected endpoints before the adapter fix.
All endpoints now round-trip, while invalid/nonfinite settings remain rejected.
The final staged package passed manifest validation and CUDA/legacy smoke tests.
The viewport benchmark explicitly disables Blender's startup splash.

GPU Flow is a particle foundation. Connected films, merging/cohesion, persistent
rivulets, deforming collision surfaces, and portable GPU bake export remain
unimplemented. GPU hosts are hidden from offline rendering by default. Use the
legacy workflow for native Geometry Nodes baking.

## Legacy 0.0.2 / M0-M1 evidence

Verified locally on 2026-09-25 with **Blender 5.2.0 LTS**, build `fbe6228777e7`, on Windows 11 and an Intel Core i7-13700 (24 logical CPUs).

## Delivered

M0 repairs actual static propagation and preserves interface IDs, values, links, and drivers across rebuilds. Candidate validation evaluates seeds, downhill trails, and render output on an isolated fixture before replacing the previous implementation.

M1 adds a separate identity water host, one-time bounded source particles, tangent gravity with analytic linear resistance, bounded support/adhesion, detachment, approximate free-drop collision and reattachment, volume removal accounting, radius-derived drop meshes, reset, and native packed baking. All temporal updates run in Geometry Nodes. Named simulation helpers require matching ownership/schema/interface signatures.

## Executed checks

| Check | Result |
|---|---|
| Pure Python suite | 20 passed |
| Actual Blender graph suite | 42 passed; final run 19.211 s |
| Seed allocation regression | Reproduced 70,711 raw points for budget 32 before fix; conservative full-area density cap passes |
| Static rebuild | Values, links, drivers, save/reload, behavioral rejection and rollback pass |
| Numerical/playback gates | Drag, free fall, 24/48 fps, zero dt, empty state, locality, support loss, contact and volume removal pass |
| Cache/bake gates | Seed replay; changed gravity after reset; two independent packed hosts; reload and unordered frame reads; downstream material edits pass |
| Staged package smoke | Static generated geometry and advancing animated render state pass |
| Extension manifest and ZIP | Blender validation passes; ZIP and mirror Python files byte-match canonical source; no bytecode |
| Paired node API inspection | Simulation/Repeat dynamic sockets and native bake properties inspected successfully |
| Independent review | Four Important findings and one minor fixed; regression suite passes |
| Hosted Blender CI | Configured manual workflow; not executed. Requires verified official archive URL and SHA256 |

Bake tests use isolated temporary files. Windows sandbox denied access to those directories, so the final suites were run with approved escalation. No Python packages are required inside Blender.

## Performance evidence

Each row is a separate sequential **240-frame, 24 fps** run. Minimum substeps=8; actual adaptive counts range from 9 to 47/48. Source Start/Softness=0 seeds almost the full sphere to approach the requested budget. Lifetime=100 and Kill Height=-1000 avoid removals in this timing fixture.

| Particle budget | Actual maximum | Triangles | Median ms | p95 ms | Max ms | Peak process MiB |
|---|---|---|---|---|---|---|
| 512 | 512 | 1,980 | 112.5 | 542.1 | 601.0 | 273.7 |
| 512 | 512 | 20,022 | 227.7 | 1028.3 | 1097.2 | 294.5 |
| 2,048 | 2,048 | 1,980 | 125.1 | 599.3 | 643.2 | 355.9 |
| 2,048 | 2,048 | 20,022 | 232.0 | 1330.3 | 1348.1 | 398.6 |

Timings include simulation, realized drop output, and one diagnostic vertex; they exclude metric readback and image rendering. First-frame compilation/setup costs are reported separately in each JSON. Peak memory includes Blender's process baseline and simulation cache. Later settled frames are cheaper than active flow, so median alone does not describe the worst playback cost.

**The proposed 100 ms/frame preview target was not achieved in this matrix.** All four runs recorded zero relative volume-ledger discrepancy at measured float precision and zero travel-limited particles. These sphere benchmarks do not prove accuracy or performance for arbitrary meshes. Numerical removal and high-speed clamp behavior are covered separately by runtime tests.

Raw reports: [512 / 1,980](../artifacts/benchmark-m1-512-2000.json), [512 / 20,022](../artifacts/benchmark-m1-512-20000.json), [2,048 / 1,980](../artifacts/benchmark-m1-2048-2000.json), [2,048 / 20,022](../artifacts/benchmark-m1-2048-20000.json).

## Visual evidence and artifacts

[Open the six-scene demo](../artifacts/Flumen_M1_Demo.blend), with packed frames **1-72**, or install [the extension ZIP](../artifacts/flumen-0.0.2.zip). Scene geometry, cameras, material display, and controls are generated by `examples/create_validation_scenes.py`.

Inspected Workbench previews: [sphere](../artifacts/Flumen_M1_Preview.png), [bottle rim](../artifacts/M1_Bottle_rim.png), [Suzanne](../artifacts/M1_Suzanne.png), [slope](../artifacts/M1_Slope.png), [opposing sheets](../artifacts/M1_Opposing_sheets.png), and [concave fold](../artifacts/M1_Concave_fold.png). Each scene's baked render mesh was checked after reopening the demo. The previews show isolated beads, surface retention, and detached drops. Beads collect at the concave valley and around facial features; there is no continuous coating or merging. The sphere and bottle's lowest drops can leave the camera crop as they fall.

## Limits and compatibility

- Collision sources must remain stationary. Transform/deformation/topology animation and moving the identity host are unsupported. M3 correspondence has not been implemented.
- Collision uses a radius-extended center ray plus proximity clearance, not a swept sphere or physical liquid solver. Sharp edges and grazing contacts remain approximate.
- A narrow source mask can seed substantially fewer particles than the budget. Constant-height geometry is rejected during setup with guidance to tilt it or change Gravity; zero density deliberately emits nothing.
- Travel clamping trades accuracy for bounded motion; inspect `sf_step_limited` and Diagnostics when increasing velocity, frame duration, or complexity.
- Clear only the affected host's native bake after state-driving changes, return to its start frame, and replay/rebake. No cross-version cache compatibility is promised.
- Static 0.0.1 interface IDs remain stable. Animated schema 1 is new in 0.0.2. There is no destructive animated rebuild operator.
- M2 merging, persistent wetness, bounded animated trails and film display are deferred. This is a procedural adaptation of the paper's phenomena, not its FLIP/APIC implementation.

The detailed [plan](superpowers/plans/2026-09-24-geometry-nodes-drips.md) and execution ledger record scope decisions. No Git repository was present; the source baseline archive and ledger are retained instead of a fabricated commit history.
