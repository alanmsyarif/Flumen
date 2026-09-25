"""Inspect paired zone sockets and native bake properties in the installed Blender."""
import bpy

NODE_TYPES = ["GeometryNodeRepeatInput", "GeometryNodeRepeatOutput",
              "GeometryNodeSimulationInput", "GeometryNodeSimulationOutput",
              "GeometryNodeSampleNearestSurface", "GeometryNodeRaycast",
              "GeometryNodeProximity", "GeometryNodePointsToCurves"]
tree = bpy.data.node_groups.new("SF_API_INSPECT", "GeometryNodeTree")
try:
    nodes = {kind: tree.nodes.new(kind) for kind in NODE_TYPES}
    for family in ("Repeat", "Simulation"):
        start=nodes["GeometryNode"+family+"Input"]
        end=nodes["GeometryNode"+family+"Output"]
        start.pair_with_output(end)
        items=end.repeat_items if family=="Repeat" else end.state_items
        items.new('FLOAT','Inspection State')
    for kind,node in nodes.items():
        print(kind)
        for direction,sockets in [('INPUT',node.inputs),('OUTPUT',node.outputs)]:
            for socket in sockets:
                print(direction,socket.name,socket.identifier,socket.bl_idname,
                      'available=',socket.enabled and not socket.is_unavailable)
    for cls in (bpy.types.NodesModifier,bpy.types.NodesModifierBake):
        print(cls.__name__,[(p.identifier,p.type) for p in cls.bl_rna.properties
                            if 'bake' in p.identifier or cls==bpy.types.NodesModifierBake])
finally:
    bpy.data.node_groups.remove(tree)
