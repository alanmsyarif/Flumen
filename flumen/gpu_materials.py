"""Display-only liquid and wet-surface materials."""
import bpy


def create_water_material():
    material=bpy.data.materials.new('Flumen Connected Water')
    material['sf_gpu_water_material']=True
    material.use_nodes=True
    shader=material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value=(.94,.98,1.,1.)
    shader.inputs['IOR'].default_value=1.333
    shader.inputs['Roughness'].default_value=.05
    shader.inputs['Transmission Weight'].default_value=1.
    material.diffuse_color=(.15,.5,.65,1.)
    return material


def create_wet_material(source):
    original=source.data.materials[0] if len(source.data.materials) else None
    material=original.copy() if original else bpy.data.materials.new('Flumen Wet Surface')
    material.name='Flumen Wet Surface'
    material['sf_gpu_wet_material']=True
    material.use_nodes=True
    nodes,links=material.node_tree.nodes,material.node_tree.links
    shader=next((n for n in nodes if n.bl_idname=='ShaderNodeBsdfPrincipled'),None)
    if shader is None:
        # A source shader without Principled remains intact in this owned copy.
        return material
    attribute=nodes.new('ShaderNodeAttribute'); attribute.attribute_name='sf_wetness'
    rough=shader.inputs['Roughness']
    dry_rough=float(rough.default_value)
    rough_source=rough.links[0].from_socket if rough.is_linked else None
    difference=nodes.new('ShaderNodeMath'); difference.operation='SUBTRACT'
    if rough_source: links.new(rough_source,difference.inputs[0])
    else: difference.inputs[0].default_value=dry_rough
    difference.inputs[1].default_value=.05
    factor=nodes.new('ShaderNodeMath'); factor.operation='MULTIPLY'
    links.new(difference.outputs[0],factor.inputs[0]); links.new(attribute.outputs['Fac'],factor.inputs[1])
    result=nodes.new('ShaderNodeMath'); result.operation='SUBTRACT'
    if rough_source: links.new(rough_source,result.inputs[0])
    else: result.inputs[0].default_value=dry_rough
    links.new(factor.outputs[0],result.inputs[1]); links.new(result.outputs[0],rough)
    color=shader.inputs['Base Color']
    color_source=color.links[0].from_socket if color.is_linked else None
    darken=nodes.new('ShaderNodeMixRGB'); darken.blend_type='MULTIPLY'
    darken.inputs[1].default_value=color.default_value[:]
    darken.inputs[2].default_value=(.55,.55,.55,1.)
    if color_source: links.new(color_source,darken.inputs[1])
    links.new(attribute.outputs['Fac'],darken.inputs[0]); links.new(darken.outputs[0],color)
    return material
