# Architecture

`flumen/` is canonical Python source. Python constructs and validates editable Geometry Nodes graphs; frame updates run entirely in Blender nodes. `scripts/build_extension.py` stages the same modules for distribution.

## Static paths

`SF_SurfaceFlow` is a stable outer interface pointing to an owned implementation child. Build reuses it. Rebuild constructs a candidate, evaluates seeds/trails/render geometry on an isolated slope, then swaps the active branch. Existing socket identifiers, modifier values, drivers, and parent links remain intact. A failed construction or validation preserves the previous implementation. Unused owned children are removed only after success.

Static `sf_age` is a path iteration index. Repeat Zone samples become curves grouped by `sf_id`; the render output includes the original surface and tube geometry.

## Animated M1

```text
stationary collision object -> realization/triangulation -> one-time SF_Source
                                                          |
                                             Simulation Zone state
                                                          |
                                             bounded Repeat Zone
                                                /             \
                                      SF_AttachedStep      SF_FreeStep
                                                \             /
                                             transition IDs + ledger
                                                          |
                                                SF_Output drops
```

An unparented identity host owns each modifier cache; collision geometry remains in a separately visible source object. The graph exposes Geometry, Particles, and Diagnostics outputs. Root and helper groups use `sf_owned`, schema version 1, and an interface signature; incompatible same-named simulation assets fail setup instead of being silently reused. Failed root construction removes only groups created by that transaction.

Particle state includes stable IDs, age in seconds, position, velocity, volume, attached/free state, normal, anchor, island, path ID, and blocked/travel-limit flags. Simulation state also carries elapsed time, initial/removed volume, next path ID, and actual substeps. Each integrator captures next attributes before updating fields they depend on. Original substep state selects the two branches, preventing double integration at transitions.

Schema 1 is the first animated release. Existing static interfaces remain compatible with 0.0.1. No cross-version simulation bake compatibility is promised; after changing simulation code/schema, keep old baked files intact and create a fresh setup. Animated force rebuild is intentionally rejected. Material-only changes are downstream of state; input/geometry changes require clearing only the affected host's native bake and replaying from its start frame.

See [Algorithm](ALGORITHM.md) for integration details and [Testing](TESTING.md) for executed behavior gates. M2 merging/wetness/trails and M3 moving surfaces are not present.
