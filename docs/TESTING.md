# Testing

Run pure tests with `python -m pytest -q`. Pytest excludes `tests/blender/`; Blender runtime tests use unittest and need no pytest installation inside Blender.

```text
blender --background --factory-startup --python-exit-code 1 --python scripts/run_blender_tests.py
blender --background --factory-startup --python-exit-code 1 --python scripts/run_blender_tests.py -- --pattern test_free.py
blender --background --factory-startup --python-exit-code 1 --python scripts/smoke_test_blender.py
```

The full runtime suite checks actual generated geometry, analytic displacement/drag, island and fold rejection, detachment and swept collision, grazing clearance, empty states, volume removal, frame rates, independent hosts, settings reset, rebuild preservation, failure rollback, and packed baking. Bake tests reopen saved files and compare randomly accessed states to sequential playback. Temporary scenes/caches belong exclusively to each test.

Smoke checks require real generated static trails and animated render geometry with advancing age. They save nothing unless `--save PATH` is provided. `--package STAGE_DIRECTORY` imports the staged extension, registers its operators, and exercises both workflows.

## Benchmarks and visual fixtures

```text
blender --background --factory-startup --python-exit-code 1 --python scripts/benchmark_blender.py -- --particles 512 --faces 2000 --frames 240 --fps 24 --substeps 8 --output artifacts/benchmark-m1-512-2000.json
blender --background --factory-startup --python-exit-code 1 --python examples/create_validation_scenes.py
```

Benchmark the 512/2,048 particle budget x 2,000/20,000 triangle target matrix in separate, sequential Blender processes. Reports contain actual seeded count and triangles, actual adaptive substep distribution, volume error, first-frame cost, median/p95/max frame time, and process peak memory (including Blender baseline). Timings include simulation, realized drop geometry, and one diagnostic vertex; metric readback and rendering are excluded. The face count is a target, and random seeding can produce fewer particles than the budget. Benchmarks set Source Start/Softness to zero to seed almost the full sphere and approach the requested particle budget; other controls use defaults except Lifetime=100 and Kill Height=-1000.

The scene generator creates six scenes with cameras and materials, packs frames 1-72, and renders a Workbench preview. Visually inspect contact, departure, clipping, and the distinction between particles and a liquid film.

## CI

`.github/workflows/python-tests.yml` runs pure tests on pushes and pull requests. `.github/workflows/blender-tests.yml` is manually dispatched: it requires a verified official Linux archive URL and SHA256 as dispatch inputs, checks that checksum and exact Blender version 5.2.0, runs runtime/staged checks, and uploads the extension. This hosted workflow has not been executed here; official download endpoints were inaccessible during local work. Local evidence uses the installed Windows Blender 5.2.0 LTS build `fbe6228777e7`.
