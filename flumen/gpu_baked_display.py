"""CUDA-free playback of offline-meshed water frames, plus a retained wet source proxy."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import bpy
from bpy.app.handlers import persistent
import numpy as np
from .particle_cache import CacheReader
from .gpu_materials import create_water_material, create_wet_material
from .gpu_water_display import _write_mesh

_SUPPORTED = {'position', 'material_index', 'sharp_face', 'sharp_edge'}


def _manifest(mesh_path):
    path = Path(mesh_path)/'manifest.json'
    if not path.is_file(): raise ValueError(f'No mesh sequence at {mesh_path}')
    try: manifest = json.loads(path.read_text(encoding='utf8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error: raise ValueError('Mesh manifest is corrupt') from error
    if manifest.get('status') != 'complete': raise ValueError(f"Mesh sequence is not complete ({manifest.get('status')})")
    return manifest


def wet_proxy(water):
    proxy = water.get('sf_baked_proxy')
    return proxy if isinstance(proxy, bpy.types.Object) else None


def create_baked_water(cache_path, mesh_path, scene, source=None):
    """Create playback objects for a complete mesh sequence of a complete particle cache."""
    reader = CacheReader(cache_path)
    manifest = _manifest(mesh_path)
    if manifest.get('source_fingerprint') != reader.header.source_fingerprint:
        raise ValueError('Mesh sequence was made from a different particle cache')
    static = reader.read_static()
    water = bpy.data.objects.new('Flumen Baked Water', bpy.data.meshes.new('Flumen Baked Water'))
    water.data['sf_baked_mesh_data'] = True
    water.data.materials.append(create_water_material())
    water['sf_baked_water'] = True
    water['sf_baked_cache'] = str(cache_path); water['sf_baked_mesh'] = str(mesh_path)
    frames = sorted(int(f) for f in manifest['frames'])
    water['sf_baked_start'], water['sf_baked_end'] = frames[0], frames[-1]
    scene.collection.objects.link(water)
    proxy = bpy.data.objects.new('Flumen Baked Wet Surface', bpy.data.meshes.new('Flumen Baked Wet Surface'))
    proxy.data['sf_baked_mesh_data'] = True
    _write_mesh(proxy.data, static['source_vertices'], static['source_triangles'])
    uv_name = source.data.uv_layers.active.name if source is not None and source.data.uv_layers.active else 'UVMap'
    proxy.data.uv_layers.new(name=uv_name).data.foreach_set('uv', static['corner_uv'].reshape(-1))
    proxy.data.attributes.new('sf_wetness', 'FLOAT', 'CORNER')
    if source is not None and len(source.data.materials):
        # Slot 0 becomes an owned wet copy; other slots keep the source materials, unchanged.
        proxy.data.materials.append(create_wet_material(source))
        for material in list(source.data.materials)[1:]: proxy.data.materials.append(material)
        indices = np.minimum(static['material_index'], len(proxy.data.materials)-1)
        proxy.data.polygons.foreach_set('material_index', indices.astype(np.int32))
    if source is not None:
        uv_names = {layer.name for layer in source.data.uv_layers}
        unsupported = sorted(a.name for a in source.data.attributes
                             if not a.name.startswith('.') and a.name not in _SUPPORTED | uv_names)
        proxy['sf_baked_unsupported'] = ', '.join(unsupported)
    proxy['sf_baked_owner'] = water; water['sf_baked_proxy'] = proxy
    scene.collection.objects.link(proxy)
    update_baked_water(water, scene.frame_current)
    return water


def _load_frame(water, frame):
    folder = Path(water['sf_baked_mesh'])
    record = _manifest(folder)['frames'].get(str(frame))
    if record is None: raise ValueError(f'Frame {frame} is missing from the mesh sequence')
    path = folder/record['file']
    if not path.is_file() or path.stat().st_size != record['bytes']: raise ValueError(f'{record["file"]} is missing or truncated')
    data = path.read_bytes()
    if sha256(data).hexdigest() != record['sha256']: raise ValueError(f'{record["file"]} checksum mismatch')
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def update_baked_water(host, frame: int) -> None:
    """Show the cached mesh for an integer frame, clamped to the baked range. Never touches CUDA."""
    frame = min(max(int(frame), host['sf_baked_start']), host['sf_baked_end'])
    arrays = _load_frame(host, frame)
    count = len(arrays['attached_vertices'])
    vertices = np.concatenate([arrays['attached_vertices'], arrays['free_vertices']])
    triangles = np.concatenate([arrays['attached_triangles'], arrays['free_triangles']+count])
    _write_mesh(host.data, vertices, triangles)
    proxy = wet_proxy(host)
    if proxy is not None:
        proxy.data.attributes['sf_wetness'].data.foreach_set('value', arrays['wet_corner'].reshape(-1))
        proxy.data.update()
    host['sf_baked_frame'] = frame
    host['sf_baked_error'] = ''


def purge_baked():
    """Release baked-owned proxies, meshes and materials whose water object is gone."""
    for proxy in [o for o in bpy.data.objects if 'sf_baked_owner' in o and o.get('sf_baked_owner') is None]:
        mesh = proxy.data; materials = list(mesh.materials)
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh.users == 0: bpy.data.meshes.remove(mesh)
        for material in materials:
            if material and material.users == 0 and material.get('sf_gpu_wet_material'): bpy.data.materials.remove(material)
    for mesh in [m for m in bpy.data.meshes if m.get('sf_baked_mesh_data') and m.users == 0]:
        materials = list(mesh.materials)
        bpy.data.meshes.remove(mesh)
        for material in materials:
            if material and material.users == 0 and (material.get('sf_gpu_water_material') or material.get('sf_gpu_wet_material')):
                bpy.data.materials.remove(material)


@persistent
def on_baked_frame(scene, depsgraph=None):
    purge_baked()
    for water in [o for o in scene.objects if o.get('sf_baked_water')]:
        try: update_baked_water(water, scene.frame_current)
        except (OSError, ValueError) as error: water['sf_baked_error'] = str(error)
