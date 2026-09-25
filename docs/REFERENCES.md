# References

## Primary visual/technical inspiration

Alexey Stomakhin, Andrew Moffat, Gary Boyle. **A Practical Guide to Thin Film and Drips Simulation.** SIGGRAPH 2019 Talks. DOI: 10.1145/3306307.3328141.

The paper studies close-up water interaction with characters and discusses surface tension, viscosity, contact-angle adhesion, and moving collision geometry using a FLIP/APIC pipeline.

Flumen uses the paper as a *phenomena reference*, not as an implementation specification. The static workflow integrates paths; M1 integrates attached and free particles on stationary collision geometry. It does not reproduce the paper's thin-film solver.

## Blender features used by the MVP

- Geometry Nodes Repeat Zone
- Geometry Proximity
- Sample Nearest Surface
- Distribute Points on Faces
- Points to Curves
- named attributes

Primary paper: https://www.wetafx.co.nz/assets/Uploads/PDFs/siggraph2019_drips-v2.pdf (local reference: `siggraph2019_drips-v2.pdf`). M1 also uses Simulation Zones, Raycast, and native packed simulation baking.
