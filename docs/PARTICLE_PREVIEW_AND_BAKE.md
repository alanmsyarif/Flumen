# Field particle preview and offline water bake (0.0.4, experimental)

Flumen 0.0.4 adds a **Surface Field** backend that previews up to one million
water particles live in the viewport on a stationary mesh. You then bake the
particles to disk and mesh them offline into renderable water. The backend is
**experimental**. The 1080p viewport speed gates pass, but the drainage comparison
against the reference footage fails with the default settings (see
[Validation status](#validation-status)). The older workflows remain available. **Create GPU Flow** now defaults to
Connected Water display; **Create Animated Flow** and **Build Flumen** are unchanged.

Requirements: Windows, an NVIDIA GPU with CUDA, Blender 5.2, and the extension
zip, which bundles Warp 1.17.0. Set Scene Unit Scale to 1.0 (one unit = one meter).
The collision mesh must stay still.

## Workflow

1. Select the mesh. Open **3D View > Sidebar > Flumen > GPU Flow** and choose
   **Field Particle Preview**. This creates a host object that draws points. The
   water is emitted from the selected mesh's own surface, not from another object.
2. Set the emission and sizing controls:
   - **Particles Per Frame**, **Emission Start/End**, **Capacity**, **Lifetime**:
     births and the size of the live pool.
   - **Radius**: the water volume per particle. Total water = count × (4/3)πr³.
   - **Field Spacing** / **Contact Spacing**: the resolution of the surface field
     and of the contact grid. These are separate from Radius. Finer spacing costs
     more preparation memory and time.
   - **Resistance**, **Field Viscosity**, **Surface Tension**: control how the film
     drains. Wall drag 3ν/h² slows thin films automatically.
   - **Contact Hysteresis** (default 0 = off): dry surface holds thin fronts until
     enough water gathers, so thick spots break through first and later water
     follows the wet tracks, forming rivulets. 0.05–0.1 is a useful range; 0.07
     was used for the rivulet stills.
3. Play from the start frame.
   - Display-only edits never need a reset.
   - Physical edits mark the host dirty. **Reset GPU Flow** then replays from the
     start and reuses the stationary preparation (chart, contact grid) as long as
     the geometry and the field/contact spacings are unchanged.
   - FIELD-specific controls appear in the panel: Pressure Cap, Capillary Cap,
     Surface Damping, merge scales and wetness rates. Diagnostics show the
     effective field spacing (with its coarsening against the requested value),
     force-cap hits (pressure/capillary cap activations) and the interval Courant number.
4. Display settings change only the drawing, never the simulation:
   - **Preview Style**: **Points** is fastest. **Water** draws a screen-space
     smoothed surface, viewport only.
   - **Display Limit** caps the drawn points with an even subset. The panel shows
     simulated and displayed counts separately.
5. **Bake Particle Cache** writes frames start..end to a new folder. The dialog
   estimates disk use, and the folder must not already exist. The bake runs as a
   cancellable modal job, does not change the live preview, and a cancelled or
   corrupt cache is rejected.
6. **Mesh Particle Cache** meshes the cache on the CPU. You set the drop grid
   spacing, film spacing, film smoothing, the film thickness cap (excess becomes
   pendant drops), drop shape (PCA or velocity), and **Wet Sheen**. Wet Sheen is a
   cosmetic thin coat where water has passed; its volume is reported separately.
   The result plays back as a mesh sequence with a wet-proxy source. Playback and
   rendering never initialize CUDA, so a `.blend` file plus its cache and mesh
   folders renders on a machine without a GPU solver. Creating baked water turns
   on Render > Lock Interface, because playback rewrites mesh data on frame change.
   The add-on's Mesh button uses one worker process. The 8-worker timings below
   come from `scripts/render_offline_clip.py --workers 8`.
7. Render the result:
   - **EEVEE** works.
   - **Cycles** with **Motion Blur** (shutter about 1 frame) also streaks falling
     drops into streams. The mesh carries a per-vertex `velocity` attribute in m/s,
     which EEVEE ignores.

Script equivalent: `scripts/render_offline_clip.py` (bake, mesh, render). Key options
are `--count`, `--water-scale`, `--contact-hysteresis`, `--water-engine cycles` and
`--shutter`.

## Measured performance (RTX 5050 laptop, Blender 5.2, Windows)

The 1080p solid viewport has 1,000,000 particles simulated and drawn. Each gate
runs 120 warmup frames plus 600 measured frames. The gate is mean ≥ 30 FPS and
p95 ≤ 33.3 ms. Background GPU apps (for example DaVinci Resolve) must be closed.

| Distribution | Defaults: FPS / p95 ms | Contact hysteresis 0.07 |
|---|---|---|
| attached | 34.3 / 31.2 | 35.0 / 30.6 |
| free | 45.6 / 23.6 | 44.7 / 23.7 |
| mixed | 42.1 / 25.3 | 41.9 / 24.7 |
| dense | 47.2 / 23.0 | 47.5 / 21.8 |
| attached, Water style | 33.3 / 32.0 | — |

Raw reports are in `artifacts/particle-viewport-*.json`. The attached run has
about 2 ms of margin. Other GPUs, meshes and spacings will differ.

Costs:
- **Cache:** about 88 bytes per particle per frame (88 MB per million particles
  per frame).
- **Bake time:** a 1M-particle bake runs at about 1.4 s per frame. A 250k-particle,
  180-frame bake takes about 2 min (cache about 4 GB).
- **Meshing:** a 250k-particle, 180-frame clip with 0.4 mm drops and a 2 mm film
  takes about 21 min on 8 worker processes.

## Numerical approximation

- **Attached water:** particles deposit volume and momentum onto a surface field.
  The field integrates gravity, hydrostatic and capillary pressure, lubrication
  wall drag, viscosity and drip release (an underside film detaches when
  h·(ĝ·n) exceeds the capillary length). Thickness is frozen within each interval
  and substeps are fixed. The interval Courant number is reported, not enforced.
- **Free drops:** follow ballistic motion with local contact against the source.
- **Conservation:** the volume ledger (emitted = live + removed) holds to round-off.
- **Wetness:** a surface record of where water has passed. With hysteresis on,
  pinned fronts do not wet.
- **Offline meshing:** the film is a conservative shell over a refined source
  lattice, floated 10 µm off the source to avoid z-fighting. Drops use PCA
  anisotropic kernels and tiled marching tetrahedra. Volume diagnostics are
  written per frame: represented, meshed, pooled, sheen, cropped and
  sub-resolution volume.

## Validation status

- **Pass:**
  - viewport speed gates (above)
  - conservation and CUDA-free playback (tests and staged smoke)
- **Fail at defaults: drainage versus the reference bust.** The fixture's thin
  coat is dominated by wall drag and drains about 2 cm in 3 s.
- **With 8× water (`--water-scale 2`), resistance 60 and contact hysteresis 0.07,**
  the bust forms a central rivulet down the muzzle with dry stripes beside it
  ([still](../artifacts/rivulet-pinning-h07.png)). A full final clip with all
  look settings was not rendered (stopped at user request).
- **Not supported:**
  - moving or deforming collision surfaces
  - GPU solving during final render (bake first)
  - non-NVIDIA GPUs

Detailed evidence is in [Connected validation](CONNECTED_WATER_VALIDATION.md).
