# Million-particle interactive editing — revised design

Status: **draft for user review**. This supersedes the real-time *final mesh*
gate of the approved 2026-09-30 design, following the user's 2026-10-01 direction.
It does not claim that the new backend or a particle bake already exists.

## Intended result

The user wants water resembling the supplied thin-film/drips references, with
continuous births per frame and responsive particle simulation while adjusting
the scene. Final meshing may run offline during a later bake. The interactive
target includes **one million live particles** on the RTX 5050; a large capacity
with only a few thousand active particles does not satisfy that target.

Keep stationary collision surfaces for this increment. Preserve old Drops,
existing Geometry Nodes bakes, current CUDA water experiments, frame-based
emission, scaled physical time, deterministic replay and volume accounting.
The moving hand and a full reproduction of the paper's FLIP/APIC method remain
outside this stationary-surface increment.

## What the measured checkpoint establishes

The current algorithm searches particle neighborhoods and integrates collision
contacts repeatedly per shared step. Storing only 64 neighbors bounds memory,
but does not bound candidate traversal when particles concentrate.

With meshing and drawing completely excluded, one million independent particles
achieved 22.31 FPS at minimum eight substeps and 25.05 FPS at minimum one after
the collision fast-path optimization. The dense interacting probe still took
about 20 seconds for its final interval. Cohesion, repulsion and damping were
enabled; merging was disabled to preserve the exact million live count.
These are short feasibility probes, not full viewport acceptance runs.

[Validation and raw evidence](../../CONNECTED_WATER_VALIDATION.md) include the
baseline and optimized reports. Merely deferring the existing mesher, hiding
particles, disabling interactions or reducing their count cannot establish the
requested interactive fluid performance.

## Recommended architecture

Use a **GPU surface-field solver with advected particles** for attached water.
Resolve thin-film interaction forces on a bounded source-local field, then
sample that field to move the million particles. Particle count controls
sampling detail; it does not multiply a nearest-neighbor solve on every substep.
Use lightweight GPU point drawing while editing and a separate cache/meshing
workflow for the final surface.

This changes the numerical approximation. The current pairwise solver remains
available for comparison and compatibility; its measured limits are labeled.
The field backend must prove conservation, surface locality and the intended
drainage behavior before becoming the recommended default.

### Stationary contact preparation

Prepare mesh adjacency, global face provenance and source-local reconstruction
coordinates once. Attached advection uses its known triangle and adjacency
walks across edges, with a bounded fallback queue for unusual contacts. Free
particles use a precomputed collision distance/contact field, with exact BVH
checks for ambiguous thin or folded surfaces.

Every contact sample carries enough face/island provenance to reject an
unrelated nearby sheet. Do not use a coarse distance field alone to transport
liquid across folded surfaces. Unsupported geometry or exhausted contact
resolution produces an explicit diagnostic. Static-source edits rebuild these
fields; no precomputed contacts survive a changed source fingerprint.

### Field interaction and particle transport

Scatter attached particle volume and momentum to a source-local surface field
with normalized weights. Compute gravity-driven tangent flow, height/pressure
gradients, viscosity and capillary/cohesive response on that field. Subcycle
the smaller field where stability requires it, rather than rebuilding a
million-particle neighbor list at each substep. Gather velocities to particles
and advect through the surface contact representation.

Particles remain the owners of liquid volume. Derived field deposits are not a
second liquid reservoir; transfer must not duplicate mass. Drainage, detachment,
coalescence and retirement conserve the particle ledger. Persistent wetness is
a separate state, updated once per completed physical interval and unaffected
by redraws or meshing. Free-drop motion/contact and volume-conserving aggregation
must use bounded work; the dense case cannot quietly fall back to all-pairs
searches. A million free particles is measured separately from attached water.

Transfer uses each particle's source anchor and normalized local interpolation
weights. It must represent a particle even when its radius is smaller than the
field-node spacing; the current nearest-footprint sampling failure is not an
acceptable shortcut. Surface adjacency, rather than Euclidean proximity alone,
defines field communication across folded regions.

Start with at most 100,000 surface nodes and 2,097,152 contact-field samples,
fixed particle buffers and bounded field iteration/contact-walk counts. Avoid
allocating the old 64-entry neighbor arrays for every field-backend particle.
Report required/effective resolution and unsupported contacts when those caps
are insufficient. Measure backend-owned GPU allocations separately from the
whole-device VRAM readings that include unrelated applications.

Field resolution and particle radius are separate controls. Coarse interactive
fields introduce an explicit approximation; a finer final mesh does not imply
that coarse simulated motion has become more accurate.

Physical controls restart/replay particle state using cached static contacts;
they do not rebuild unchanged source contact data. Display style, visibility,
materials and offline mesh quality do not reset the particle simulation.

### Particle viewport

Draw particles as GPU points, with a lightweight position/radius stream and no
per-frame Blender surface-mesh rebuild or instanced sphere geometry. All
particles remain in the numerical solve. The full-count benchmark draws all
one million particles; an optional display subset must show its displayed count
and cannot pass that benchmark.

Blender provides a built-in point shader suitable for the first drawing path.
[Blender 5.2 GPU shader documentation](https://docs.blender.org/api/5.2/gpu.shader.html)
The first implementation may use pinned readback and bulk GPU upload; measure
those copies explicitly. Add graphics/CUDA interop only if the measured copy
cost prevents acceptance, with a supported-backend check and complete resource
cleanup. Do not assume interop is portable across Blender graphics backends.

### Cache and offline meshing

Interactive editing does not write an obligatory disk cache or rebuild final
geometry. An explicit bake records integer-frame particle states, volume,
identity/contact data, physical settings, wetness and the source fingerprint.
Serialize data, never CUDA pointers. Bake can run slower than real time and
must support cancellation without marking a partial cache complete.

Final meshing reads that immutable cache. It can refine the attached film and
free-droplet field more finely than the current live proxy, processing bounded
tiles and streaming output when necessary. A user can change mesh quality or
material without rerunning particle motion. Cache validation rejects mismatched
sources/settings and missing frames. Disk usage is estimated before baking;
one million full particle states per frame can produce a large cache.

## Acceptance and evidence

1. On the documented RTX 5050, complete 120 warmup plus 600 measured 1920×1080
   particle viewport draws, with at least 1,000,000 live simulated and displayed
   particles throughout the measured run, average >=30 FPS and p95 <=33.3 ms.
   Record actual counts, scene/settings, source triangles, GPU stages, transfers,
   draw costs and memory. No skipped simulation intervals or hidden decimation.
   The primary stationary fixture is approximately 0.3 m Suzanne with 15,000–
   25,000 evaluated triangles, 0.1 mm nominal particle radius and time scale 0.5
   at 30 FPS. Record any particle reseeding/aggregation needed to maintain count;
   it must conserve existing liquid rather than silently create extra volume.
2. Measure attached, free and mixed distributions separately. Include a dense
   concentrated-flow stress case; isolated uniformly spaced particles cannot
   establish clustered-water performance. Any distribution that misses the
   target is reported as a limitation, not generalized into a pass.
3. Particle ledger relative error <=1e-5; replay positions within 1e-6 m at
   equivalent settings. Finite state, bounded work/storage, deterministic births
   and safe slot reuse. Field transfers conserve represented volume; unresolved
   volume is reported. Grid or contact overflow cannot silently lose water.
4. Qualitative motion shows draining coating, uneven persistent channels and
   growing/releasing drops. Offline meshing shows connected geometry and current
   neck breakup, with retained/decaying wetness. Use both opaque and water views
   and compare against the supplied stationary bust reference.
5. Repeated draws and shader edits do not advance physics or emit/deposit twice.
   Reset, duplicate, undo, load and delete release only owned resources.
6. Baked playback and meshing are independent of live simulation state. Finished
   mesh playback/rendering needs no CUDA solver. A corrupt,
   partial or incompatible cache is rejected. Final meshing has no real-time
   requirement; its time, memory, resolution and reconstruction error are stated.

**This is a target to verify, not a promise that the RTX 5050 already achieves it.**
If the field backend misses the measured budget, report the stages and limits
before recommending a reduced count, fidelity or stronger GPU.

## Implementation order after review

First prove conservative source-local field transfer, local contact traversal
and a million-particle solver-only budget. Next add and measure full-count point
drawing. Then validate drainage/aggregation against the reference, implement the
explicit cache and offline mesher, and package the verified result. Keep each
stage usable and measured; do not spend time polishing a final mesh while the
particle solver still misses the interactive budget.

The detailed implementation plan is written after this revised design is
reviewed. Continue the already selected Native execution method.
