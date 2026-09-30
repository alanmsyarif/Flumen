# GPU connected water — proposed visual increment

Status: approved by the user on 2026-09-30. Implementation planning authorized.

## Intended result

The original goal remains a real-time GPU water effect resembling the supplied
paper and video, with optional continuous emission by frame. The delivered 0.0.3
particle foundation passes its performance tests but does not meet that visual
goal. This increment must visibly close the gap: an initially connected layer
drains into uneven channels, gathers into larger drops, stretches near shedding
points, and breaks into free drops. Persistent wetness remains after water drains.

Target stationary surfaces first, as in the approved GPU design. The moving hand
requires subsequent surface correspondence and inherited velocity; this delivery
must not be described as a match for that animated portion of the reference.
Keep Windows x64, Blender 5.2, Warp 1.17.0 and the RTX 5050 as the target system.

## Reference investigation

Read both pages of the local `siggraph2019_drips-v2.pdf` and inspected the local
1920×1080, 30 FPS, 6.941-second `m2-vp9-res_1080p.mp4`.

- At 0.8 seconds, the hand shows connected surface patches, thick borders,
  narrow stretched necks and detached drops. Its overlay says Time Scale 0.05.
- At 2.5 seconds, the bust shows an initial coating and downward draining fronts.
  Its overlay says Particle Sep 0.0008, Time Scale 0.5, Contact Angle 70,
  Viscosity Mult 10 and Surface Tension 0.072. Treat these as reference controls,
  not values that can be copied into the current different numerical model.
- The paper uses a FLIP/APIC pipeline, pressure/curvature surface tension,
  near-surface viscosity and contact-angle extrapolation of the fluid SDF.
  It reports 0.4–1 mm voxels, 10–100 substeps per frame, 5–20 million particles
  for a face, and overnight computation. It does not establish real-time playback.

The current implementation has independent particle integration in
`flumen/gpu/motion.py`, no interaction kernel, and instanced level-1 icospheres
in `flumen/gpu_display.py`. Its snapshot contains only position, radius and ID.
The demo uses blue solid shading, no water material, and no initial connected
layer. These explain the gap; increasing particle count or shader gloss alone
does not supply the missing water behavior.

## Options and recommendation

1. **Recommended: interacting GPU particles plus surface-aware reconstruction.**
   Reuse the existing CUDA collision/emission/lifecycle foundation. Add bounded
   surface interactions, live liquid thickness reconstruction, and independent
   wetness. This aims for the visible phenomena under the real-time budget,
   without claiming the paper's physical solver.
2. **Appearance-only channels and wetness.** Cheaper, but historical tubes do not
   retract or break like water and would conceal the missing interaction model.
   This is insufficient as the completion criterion.
3. **Implement the paper's grid/particle solver.** Closest to its physical method,
   but pressure, viscosity, SDF extrapolation and high-resolution reconstruction
   form a separate research-scale solver. The paper gives no basis for promising
   its fidelity at 30 FPS on this GPU.

## Simulation changes

Keep the existing particle volume ledger authoritative. Add spatial neighbor
queries on CUDA with distance, attachment state, island, normal and local-surface
visibility checks. Euclidean proximity alone must not connect opposite sides of
skin, close disconnected sheets, or unrelated parts of a folded surface.

Provide bounded short-range cohesion and velocity damping while attached.
Use pressure/spacing repulsion to prevent attraction from collapsing all water
into a point. Resolve close pair merging in two passes: reciprocal partner
selection with particle-ID tie-breaking, then one owner writes the merged state.
Sum volume and preserve volume-weighted velocity; retire the consumed slot
without recording its volume as removed. Do not merge unrelated attachment
states. Keep the survivor ID and invalidate obsolete display links.

Interactions use a shared substep boundary and read previous positions/velocities
into scratch buffers. The foundation's independent per-thread stepping cannot
be reused unchanged for coupled forces. Bound substeps to 64 and keep existing
travel-limit diagnostics. Cap accepted neighbors at 64 with a reported overflow
counter and deterministic selection; overloaded neighborhoods must not silently
produce a different result based on CUDA iteration order.

Add a time-scale control: physical interval = fps_base / fps × time_scale.
Age, drying, interaction and movement all use this interval. Emission remains
once per eligible integer frame; slowing physics deliberately does not reduce
births per displayed frame. Keep 1.0 as the compatibility default; the visual
demo may use 0.5. Add an optional initial coating batch, emitted once at the
simulation start and counted as emitted volume, alongside continuous emission.
Its default is zero, preserving existing scenes. The reference demo starts with
a 4096-particle coating and 64 births/frame, subject to the existing capacity.

## Liquid geometry and wetness

Extend compact readback with attachment state, surface normal, velocity, volume
and stable identity. Implement a Connected Water display mode; preserve Drops
as a diagnostic/performance baseline. Existing files retain their stored mode.

For attached water, reconstruct a local thickness field from current live volume
on a stationary surface proxy. Normalize each particle's deposition weights over
its supported area; test the resulting thickness-volume integral. Use the field
to generate a connected raised water surface with boundaries, allowing dense
regions to form sheets and sparse draining regions to form channels. Dry regions
must not render a uniform water shell. Surface locality applies to reconstruction
as well as forces. Reconstruction radius must not bridge unrelated geometry.

For free water, reconstruct smooth drops and short necks from current particles,
with bounded kernel elongation. Do not draw unlimited historical trajectories as
water. Neck links disappear when their live support separates or expires.
Reconstruction may approximate visible volume, but must expose its error and
cannot add volume back into simulation. Wetness and active liquid are distinct.

Maintain a bounded surface wetness field in [0,1], deposited by attached water
and decaying in physical seconds. Wetness modifies roughness/color after active
liquid drains. It is not a thickness or liquid-volume reservoir. Use a generated
surface proxy owned by the host; do not modify the user's source mesh/material
or introduce a dependency cycle from source back to host.

Use smooth water normals and a neutral, transparent water material with IOR
1.333 and roughness 0.05. Supply an Eevee preview scene with a dark background,
large reflection lights and consistent camera framing. Also include a neutral
opaque geometry view so attractive shading cannot hide disconnected beads.

Reconstruction budgets: at most 100,000 surface proxy vertices, 2,097,152
volumetric samples when used, 250,000 output vertices and 500,000 triangles.
Reuse buffers. Report budget/coarsening/overflow; never truncate unnoticed or
allocate an unbounded domain around every falling drop. Preserve the particle
baseline when the user explicitly selects Drops; do not silently use it while
claiming Connected Water passed.

## Controls and compatibility

Separate physics controls requiring Reset from appearance controls that can be
edited live. New simulation controls: interactions enabled, cohesion strength,
interaction radius, surface damping, merge threshold, time scale and initial
coating count. New display controls: Drops/Connected Water, reconstruction scale,
wetness deposit/drying, and water material. Proposed Connected Water starting
values are: interaction radius 4× nominal particle radius, cohesion acceleration
limit 2 m/s², repulsion acceleration limit 20 m/s², surface damping 10 s⁻¹,
merge distance 0.25× the sum of pair radii, and maximum merged attached radius
4× nominal radius. These are approximation controls, not the paper's SI surface
tension/viscosity coefficients. Connected Water enables interactions; existing
Drops scenes retain interactions disabled. Shader roughness defaults to 0.05;
wetness drying defaults to 0.1 s⁻¹. Deposition raises local wetness toward one
with a 5 s⁻¹ rate while liquid covers the proxy. Numerical fixtures validate these
starting values before any demo-specific tuning is recorded.

State-driving controls, proxy resolution and wetness deposition/drying require
Reset; shader edits and output shading do not. Duplicate hosts must own separate
proxy/wetness state. Delete, undo, reload, reset and unregister release all new
buffers and generated owned objects. Backward seeks reset/replay all fields.
Retain strict finite/range validation, integer-frame semantics and no CPU fallback.

Offline portable baking remains outside this increment. Save reproducible live
demo settings and export a measured viewport comparison clip. Do not enable
ordinary offline rendering of an unbaked live sequence or claim it is supported.

## Acceptance: visual and numerical evidence

Use an approximately 0.3 m tall stationary Suzanne fixture with about 20,000
evaluated triangles, 8192 slots and the initial-coating/continuous preset.
This provides a reproducible surface when the reference head asset is unavailable;
different source geometry prevents a pixel-identical comparison.

Deliver the saved demo, a fixed-camera 180-frame preview clip, reference/current
comparison images, numerical results and full viewport timing. Visual acceptance
requires all of the following in that sequence, not merely a still screenshot:

1. A connected initial water patch drains and its boundary moves down the surface.
2. Several uneven channels remain connected over multiple frames and collect water.
3. Close drops combine into larger drops; the neck near a shedding point narrows
   and loses its connection as a drop separates.
4. Wetness remains after active liquid drains, then decays when emission stops.
5. Neutral geometry and water-material views both show continuous water features;
   zoomed geometry must not remain a field of independent beads.

Numerical tests: reciprocal merging preserves volume and momentum, no pair is
consumed twice, symmetric interactions use the same prior state, dt=0 is inert,
nearby unrelated sheets remain separate, bounded forces keep state finite,
and initial/continuous emission keeps the ledger balanced. Require replay within
the existing 1e-6 m tolerance and relative volume-ledger error <=1e-5. Verify
wetness timing at equivalent elapsed time at 24/48 FPS and different time scales.
Reconstruction must have no NaNs, no unsupported cross-surface links, and a
surface thickness integral within 5% of its represented attached volume.

Performance acceptance: at least 30 FPS average and p95 <=33.3 ms across 600
completed 1920×1080 Eevee viewport draws after 120 warmup frames, with the visual
mode and its interactions enabled. Measure solver, reconstruction, transfer,
Blender update and draw; count all simulated frames and report actual live count,
mesh resolution and memory. Also measure the solid geometry view separately.
The earlier sphere/Drops benchmark is not evidence for this visual-mode gate.
If the gate fails, report the measured tradeoff rather than calling it real time.

## Source references

- User's local PDF: `C:/Users/user/Documents/Flumen/siggraph2019_drips-v2.pdf`.
- Publisher copy: https://www.wetafx.co.nz/assets/Uploads/PDFs/siggraph2019_drips-v2.pdf
- User's local video: `C:/Users/user/Documents/Flumen/m2-vp9-res_1080p.mp4`.
- Extracted frames: `artifacts/reference-hand-0p8.png` and
  `artifacts/reference-bust-2p5.png` in the main project.
- Warp 1.17.0 CUDA neighbor and reconstruction examples:
  https://github.com/NVIDIA/warp/blob/v1.17.0/warp/examples/core/example_sph.py
  and https://github.com/NVIDIA/warp/blob/v1.17.0/warp/examples/core/example_marching_cubes.py
- Prior approved foundation: `2026-09-30-gpu-flow-design.md` in this directory.
