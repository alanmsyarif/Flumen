"""CPU-only water geometry containers shared by live and offline meshing."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MeshBatch:
    vertices: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @classmethod
    def empty(cls,**diagnostics):
        return cls(np.empty((0,3),np.float32),np.empty((0,3),np.float32),
                   np.empty((0,3),np.int32),diagnostics)


@dataclass
class WaterGeometry:
    attached: MeshBatch
    free: MeshBatch
    diagnostics: dict = field(default_factory=dict)
