# Contributing

This is an R&D repository. Keep algorithmic changes isolated and measurable.

Before opening a pull request:

1. run `python -m pytest`
2. run Blender smoke test on 5.2 LTS
3. visually test sphere, bottle, and head
4. avoid bundling unrelated node-graph cleanup with algorithm changes
5. document new named attributes in `docs/ARCHITECTURE.md`

Node groups and attributes use the `SF_` / `sf_` prefixes.
