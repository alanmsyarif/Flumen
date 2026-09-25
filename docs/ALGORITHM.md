# Animated M1 algorithm

The collision object is evaluated in the separate identity host's coordinate space, realized, and triangulated. One-time source points use gravity-relative normalized height, a hard or soft threshold, and a bounded density. Flat height ranges produce no sources.

A Simulation Zone stores particles, elapsed seconds, initial/removed volume, the next path ID, and substeps. Each frame uses a Repeat Zone with a speed/gravity travel estimate and 1-64 substeps (default minimum 8). Original attached/free states are split before updates; state transitions cannot integrate a particle twice in one substep.

For an attached particle, sample and normalize the supporting normal on its stored mesh island. Project gravity and velocity onto the tangent plane. Integrate linear drag analytically, including a small-resistance series to avoid numerical cancellation. A sampled `sf_resistance` attribute overrides the scalar control. Project the candidate onto that island, reject excessive normal turns, and detach when support is lost or outward gravity exceeds the artistic Adhesion Acceleration threshold.

For a free particle, integrate ballistic position and velocity. Raycast along displacement extended by its volume-derived radius; nearest-surface queries provide radius clearance, never reattachment on their own. A low inward-speed front hit attaches; other impacts remove inward velocity with zero restitution. This center-ray approximation is not a swept-sphere solver. Grazing edges and sharp thin features remain limitations.

Commit next-state fields to temporary named attributes before changing their dependencies. Retain volume and stable particle IDs; allocate new path IDs on transitions. Remove particles only through lifetime or kill-height rules and accumulate their volume in the ledger. Travel clamping sets a sticky diagnostic flag. Zero time steps preserve state.

Render only the water particles, as realized icospheres with radius `(3 * volume / (4 * pi)) ** (1/3)`. Attached centers are offset from the anchor by that radius. Source geometry remains separately visible. Material application occurs outside simulation state. M1 has no merging, wetness, or accumulated animated trails.

Native Blender caches and bakes own temporal evaluation. No Python frame handlers, external solvers, or background simulation process are used. Source and host motion are unsupported.

---

# Static Path Algorithm

## 1. Automatic source

Normalize the surface's position along `Up = -normalize(Gravity)`.

```text
h(p) = dot(p, Up)
height01 = remap(h, min(h), max(h), 0, 1)
```

A smooth threshold around `Source Start` creates the initial source/coating region.

## 2. Seed points

`Distribute Points on Faces` uses:

```text
density = Seed Density * source_mask
```

Each seed stores a stable `sf_id` and starts at `sf_age = 0`.

## 3. Tangent gravity

For current surface normal `N` and normalized gravity `G`:

```text
G_t = G - dot(G, N) * N
D = normalize(G_t)
```

On a horizontal top where gravity is parallel to the normal, `G_t` approaches zero. Randomness can break the symmetry, but v0.0.1 does not yet have curvature-aware fallback logic.

## 4. Stable perturbation

A deterministic random vector is generated from path ID, user seed, and Repeat Zone iteration. Its normal component is removed before blending with the tangent-gravity direction.

```text
R_t = R - dot(R, N) * N
D = normalize(D + randomness * R_t)
```

## 5. Integration

```text
candidate = p + D * StepLength
```

The candidate is moved to the closest location on the target surface.

## 6. Trail recording

The projected tip receives:

```text
sf_age = iteration + 1
```

and is joined to the previous trail point cloud.

## 7. Curve reconstruction

At the end:

```text
Points to Curves
Curve Group ID = sf_id
Weight         = sf_age
```

The resulting curves are offset slightly along `sf_normal`, given a radius, and converted to preview tubes.
