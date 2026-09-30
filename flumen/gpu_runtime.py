"""Blender mesh extraction and GPU host lifecycle."""
import numpy as np


def extract_source(obj, depsgraph) -> tuple:
    if obj is None or obj.type != 'MESH':
        raise ValueError('Select a collision mesh')
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = np.empty(len(mesh.vertices)*3,dtype=np.float32)
        mesh.vertices.foreach_get('co',vertices)
        vertices = vertices.reshape(-1,3)
        matrix = np.asarray(evaluated.matrix_world,dtype=np.float32)
        vertices = vertices @ matrix[:3,:3].T + matrix[:3,3]
        faces = np.empty(len(mesh.loop_triangles)*3,dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices',faces)
        faces = faces.reshape(-1,3)
        parents = list(range(len(vertices)))
        def root(a):
            while parents[a] != a:
                parents[a] = parents[parents[a]]
                a = parents[a]
            return a
        for a,b,c in faces:
            ra = root(int(a))
            parents[root(int(b))] = ra
            parents[root(int(c))] = ra
        islands = np.asarray([root(int(face[0])) for face in faces],dtype=np.int32)
        return np.ascontiguousarray(vertices), faces, islands
    finally:
        evaluated.to_mesh_clear()
