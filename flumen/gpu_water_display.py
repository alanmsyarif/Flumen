"""Bulk live water output and exclusively host-owned wetness geometry."""
import bpy
import numpy as np
from .gpu_materials import create_water_material,create_wet_material


def _write_mesh(mesh,vertices,triangles,normals=None):
    mesh.clear_geometry()
    n=len(vertices); t=len(triangles)
    if n:
        mesh.vertices.add(n)
        mesh.vertices.foreach_set('co',np.asarray(vertices,np.float32).reshape(-1))
    if t:
        mesh.loops.add(3*t); mesh.polygons.add(t)
        mesh.loops.foreach_set('vertex_index',np.asarray(triangles,np.int32).reshape(-1))
        mesh.polygons.foreach_set('loop_start',np.arange(t,dtype=np.int32)*3)
        mesh.polygons.foreach_set('loop_total',np.full(t,3,np.int32))
        mesh.polygons.foreach_set('use_smooth',np.ones(t,bool))
    mesh.update()
    # Shared vertices let Blender compute smooth normals from the live shape.
    # Importing per-vertex custom normals through RNA dominates frame time.


def _remove_proxy(proxy):
    mesh=proxy.data
    materials=list(mesh.materials)
    bpy.data.objects.remove(proxy,do_unlink=True)
    if mesh.users==0: bpy.data.meshes.remove(mesh)
    for material in materials:
        if material and material.users==0 and material.get('sf_gpu_wet_material'):
            bpy.data.materials.remove(material)


def release_water_display(host):
    for proxy in list(bpy.data.objects):
        if proxy.get('sf_gpu_wet_owned') and proxy.get('sf_gpu_owner')==host:
            _remove_proxy(proxy)
    if 'sf_gpu_wet_proxy' in host: del host['sf_gpu_wet_proxy']
    if host.get('sf_gpu_water_display'):
        if host.data.users>1: host.data=host.data.copy()
        host.data.clear_geometry(); host.data.update()


def purge_orphan_water_displays():
    for proxy in list(bpy.data.objects):
        if proxy.get('sf_gpu_wet_owned') and proxy.get('sf_gpu_owner') is None:
            _remove_proxy(proxy)


def create_water_display(host,topology):
    release_water_display(host)
    if host.data.users>1: host.data=host.data.copy()
    for modifier in list(host.modifiers):
        if modifier.type=='NODES' and modifier.node_group and modifier.node_group.get('sf_gpu_display'):
            tree=modifier.node_group; host.modifiers.remove(modifier)
            if tree.users==0: bpy.data.node_groups.remove(tree)
    host['sf_gpu_water_display']=True
    if host.flumen_gpu.material is None:
        host.flumen_gpu.material=create_water_material()
    from .gpu_display import set_material
    set_material(host,host.flumen_gpu.material)
    mesh=bpy.data.meshes.new('Flumen Wetness Proxy')
    proxy=None
    try:
        # Tiny outward displacement prevents coplanar depth conflicts.
        points=topology.vertices+topology.normals*2.e-6
        _write_mesh(mesh,points,topology.triangles,topology.normals)
        mesh.attributes.new('sf_wetness','FLOAT','POINT')
        mesh.materials.append(create_wet_material(host.flumen_gpu.source))
        proxy=bpy.data.objects.new('Flumen Wetness',mesh)
        proxy.color=host.flumen_gpu.source.color[:]
        proxy['sf_gpu_wet_owned']=True; proxy['sf_gpu_owner']=host
        proxy.hide_render=True; proxy.hide_select=True
        for collection in host.users_collection: collection.objects.link(proxy)
        host['sf_gpu_wet_proxy']=proxy
    except Exception:
        if proxy is not None: _remove_proxy(proxy)
        elif mesh.users==0:
            materials=list(mesh.materials); bpy.data.meshes.remove(mesh)
            for material in materials:
                if material and material.users==0 and material.get('sf_gpu_wet_material'):
                    bpy.data.materials.remove(material)
        raise


def update_water_display(host,geometry,fields):
    if host.data.users>1: host.data=host.data.copy()
    a,f=geometry.attached,geometry.free
    # Free output is a GPU triangle stream. Share exact coincident samples for
    # smooth native viewport normals without moving the reconstructed surface.
    if len(f.vertices):
        free_vertices,inverse=np.unique(f.vertices,axis=0,return_inverse=True)
        free_triangles=inverse[f.triangles].astype(np.int32)
    else:
        free_vertices=f.vertices; free_triangles=f.triangles
    vertices=np.concatenate((a.vertices,free_vertices))
    triangles=np.concatenate((a.triangles,free_triangles+len(a.vertices)))
    _write_mesh(host.data,vertices,triangles)
    proxy=host.get('sf_gpu_wet_proxy')
    if proxy is None or proxy.get('sf_gpu_owner')!=host:
        raise RuntimeError('Wetness output missing. Reset GPU Flow.')
    if len(proxy.data.vertices)!=len(fields.wetness):
        raise RuntimeError('Wetness proxy changed. Reset GPU Flow.')
    proxy.data.attributes['sf_wetness'].data.foreach_set('value',fields.wetness)
    proxy.data.update()
    host['sf_gpu_geometry_error']=geometry.diagnostics.get('error','')
    host['sf_gpu_coarsening']=geometry.diagnostics.get('coarsening_factor',1.)
    host['sf_gpu_unrepresented_volume']=geometry.diagnostics.get('unrepresented_volume',0.)
