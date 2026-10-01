import importlib
import unittest
import numpy as np
from flumen.gpu.config import FlowConfig
from flumen.gpu.device import require_cuda
from flumen.gpu.source import build_source
from flumen.gpu.topology import build_topology
from flumen.gpu.surface import build_surface


def signed_volume(mesh):
    p=mesh.vertices.astype(np.float64)[mesh.triangles]
    return float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)


class SurfaceMeshTests(unittest.TestCase):
    def make(self):
        self.assertIsNotNone(importlib.util.find_spec('flumen.gpu.surface_mesh'),'Connected mesh implementation missing')
        cfg=FlowConfig(display_mode='CONNECTED')
        device=require_cuda()
        source=build_source([[0,0,0],[.01,0,0],[.01,0,.01],[0,0,.01]],[[0,1,2],[0,2,3]],[0,0],cfg,device)
        topology=build_topology(source,.004,.003)
        surface=build_surface(topology,cfg,device)
        api=importlib.import_module('flumen.gpu.surface_mesh')
        buffers=api.GeometryBuffers(len(topology.triangles),device.alias)
        for resource in (source,topology,surface,buffers): self.addCleanup(resource.close)
        return cfg,topology,surface,buffers,api

    def mesh(self,f):
        cfg,t,s,b,api=f
        return api.build_attached_mesh(t,s,cfg,b)

    def test_connected_patch_and_volume(self):
        f=self.make()
        f[2].thickness.assign(np.full(len(f[1].vertices),.0001,dtype=np.float32))
        mesh=self.mesh(f)
        self.assertGreater(len(mesh.triangles),0)
        self.assertTrue(np.isfinite(mesh.vertices).all())
        np.testing.assert_allclose(np.linalg.norm(mesh.normals,axis=1),1,atol=1e-6)
        self.assertAlmostEqual(signed_volume(mesh),1e-8,delta=5e-10)
        # Welding by shared coordinates is enough to verify a closed patch.
        _,inverse=np.unique(np.round(mesh.vertices,8),axis=0,return_inverse=True)
        triangles=inverse[mesh.triangles]
        edges=np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        self.assertTrue((counts==2).all())

    def test_dry_hole_and_expiry(self):
        f=self.make(); t=f[1]
        height=np.where(np.linalg.norm(t.vertices-[.005,0,.005],axis=1)<.002,0.,.0001).astype(np.float32)
        f[2].thickness.assign(height)
        mesh=self.mesh(f)
        centers=mesh.vertices[mesh.triangles].mean(axis=1)
        self.assertFalse((np.linalg.norm(centers[:,[0,2]]-[.005,.005],axis=1)<.0005).any())
        f[2].thickness.zero_()
        self.assertEqual(len(self.mesh(f).triangles),0)

    def test_mesh_budget_failure_is_explicit(self):
        f=list(self.make())
        f[2].thickness.assign(np.full(len(f[1].vertices),.0001,dtype=np.float32))
        f[3]=f[4].GeometryBuffers(len(f[1].triangles),require_cuda().alias,vertex_budget=3,triangle_budget=1)
        self.addCleanup(f[3].close)
        mesh=self.mesh(f)
        self.assertEqual(len(mesh.vertices),0)
        self.assertIn('budget',mesh.diagnostics['error'].lower())

    def test_shared_patch_vertices_fit_the_actual_budget(self):
        f=list(self.make())
        f[2].thickness.assign(np.full(len(f[1].vertices),.0001,np.float32))
        limit=2*len(f[1].vertices)
        f[3]=f[4].GeometryBuffers(len(f[1].triangles),require_cuda().alias,vertex_budget=limit)
        self.addCleanup(f[3].close)
        mesh=self.mesh(f)
        self.assertGreater(len(mesh.triangles),0)
        self.assertLessEqual(len(mesh.vertices),limit)
        self.assertAlmostEqual(signed_volume(mesh),1e-8,delta=5e-10)

    def test_disconnected_curved_surfaces_do_not_bridge(self):
        f=self.make(); cfg,_,_,_,api=f
        vertices=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float32)*.01
        triangles=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
        # Two distinct curved components separated by a small gap.
        points=np.concatenate((vertices,vertices+[.021,0,0]))
        faces=np.concatenate((triangles,triangles+6))
        device=require_cuda()
        source=build_source(points,faces,[0]*8+[1]*8,cfg,device)
        topology=build_topology(source,.004,.006)
        surface=build_surface(topology,cfg,device)
        buffers=api.GeometryBuffers(len(topology.triangles),device.alias)
        for resource in (source,topology,surface,buffers): self.addCleanup(resource.close)
        surface.thickness.assign(np.full(len(topology.vertices),.0001,np.float32))
        mesh=api.build_attached_mesh(topology,surface,cfg,buffers)
        self.assertGreater(len(mesh.vertices),0)
        centers=mesh.vertices[mesh.triangles].mean(axis=1)
        self.assertFalse(((centers[:,0]>.0101)&(centers[:,0]<.0109)).any())
        # Output is exactly base or outward displacement of a supported vertex.
        expected=np.concatenate((topology.vertices,topology.vertices+.0001*topology.normals))
        for block in np.array_split(mesh.vertices,16):
            self.assertLess(np.min(np.linalg.norm(block[:,None]-expected,axis=2),axis=1).max(),1e-7)
