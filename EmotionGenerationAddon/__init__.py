bl_info = {
    "name": "ConfigurationGenerator",
    "author": "",
    "version": (1, 0),
    "blender": (2, 80, 0),  # require Blender 2.80 or newer
    "location": "View3D > Sidebar (N Panel) > Configuration Generator",
    "description": "Generates a configuration",
    "warning": "",
    "doc_url": "",
    "category": "Lighting",
}

import bpy
import os
import colorsys
import math
import importlib
from . import modelUtils
importlib.reload(modelUtils)

# A poll function to ensure only Light objects can be selected
def isLight(self, object):
    return object.type == 'LIGHT'

def wrapHue(hue):
    return hue % 1.0

def setupNoiseCompositor(scene, grainIntensity):
    if grainIntensity <= 0:
        return
    scene.render.use_compositing = True
    # Create compositor node tree if it doesn't exist
    if scene.compositing_node_group is None:
        tree = bpy.data.node_groups.new(name="Compositing", type='CompositorNodeTree')
        tree.interface.new_socket(
            name="Image",
            description="Color output",
            in_out='OUTPUT',
            socket_type='NodeSocketColor'
        )
        scene.compositing_node_group = tree
    else:
        tree = scene.compositing_node_group
    # Take render input
    renderLayersNode = None
    for node in tree.nodes:
        if node.bl_idname == 'CompositorNodeRLayers':
            renderLayersNode = node
            break
    
    if not renderLayersNode:
        renderLayersNode = tree.nodes.new('CompositorNodeRLayers')
        print(f"renderLayersNode: {renderLayersNode.bl_idname}")
        renderLayersNode.location = (-400, 200)
    
    # Create + setup white noise
    whiteNoiseNode = None
    for node in tree.nodes:
        if node.bl_idname == 'ShaderNodeTexWhiteNoise':
            whiteNoiseNode = node
            break
    
    if not whiteNoiseNode:
        whiteNoiseNode = tree.nodes.new('ShaderNodeTexWhiteNoise')
        print(f"whiteNoiseNode: {whiteNoiseNode.bl_idname}")
        whiteNoiseNode.location = (-400, -100)
        whiteNoiseNode.noise_dimensions = '3D'
    
    # Create alpha over node to combine noise
    alphaOverNode = None
    for node in tree.nodes:
        if node.bl_idname == 'CompositorNodeAlphaOver':
            alphaOverNode = node
            break
    
    if not alphaOverNode:
        alphaOverNode = tree.nodes.new('CompositorNodeAlphaOver')
        print(f"alphaOverNode: {alphaOverNode.bl_idname}")
        alphaOverNode.location = (0, 100)
    # Set grain intensity - 0.1 is max
    alphaOverNode.inputs[2].default_value = 0.1 * grainIntensity
    
    # Add render layers and noise into the alpha over
    if renderLayersNode.outputs['Image'] and alphaOverNode.inputs[0]:
        tree.links.new(renderLayersNode.outputs['Image'], alphaOverNode.inputs[0])
    if whiteNoiseNode.outputs['Color'] and alphaOverNode.inputs[1]:
        tree.links.new(whiteNoiseNode.outputs['Color'], alphaOverNode.inputs[1])

    # Map alpha over into output
    groupOutputNode = None
    for node in tree.nodes:
        if node.bl_idname == 'NodeGroupOutput':
            groupOutputNode = node
            break
    
    if not groupOutputNode:
        groupOutputNode = tree.nodes.new('NodeGroupOutput')
        print(f"groupOutputNode: {groupOutputNode.bl_idname}")
        groupOutputNode.location = (300, 100)

    if alphaOverNode.outputs['Image'] and groupOutputNode.inputs[0]:
        tree.links.new(alphaOverNode.outputs['Image'], groupOutputNode.inputs[0])

def generateConfig(context):
    """
    Generates parameters from emotion and applies them to the scene
    """
    scene = context.scene
    targetLight = scene.targetLight
    fillLight = scene.fillLight
    rimLight = scene.rimLight
    camera = scene.camera
    if not (targetLight and targetLight.type == 'LIGHT' and camera and camera.type == 'CAMERA'
            and fillLight and fillLight.type == 'LIGHT' and rimLight and rimLight.type == 'LIGHT'):
        return

    emotionSliders = scene.emotionProps
    inputVector = [emotionSliders.anger, emotionSliders.disgust, emotionSliders.fear,
                   emotionSliders.joy, emotionSliders.sadness, emotionSliders.surprise, emotionSliders.neutral]
    rawOutput = modelUtils.performInference(inputVector)
    parameters = modelUtils.mapRawOutput(rawOutput)

    camera.data.lens_unit = 'FOV'
    camera.data.angle = math.radians(parameters['fov'])
    setupNoiseCompositor(scene, parameters['grain'])
    
    targetRGB = colorsys.hsv_to_rgb(parameters['hue'], parameters['saturation'], 1.0)
    targetLight.data.color = targetRGB
    targetLight.data.energy = parameters['lightEnergy']
    if targetLight.data.type != 'SUN' and targetLight.data.node_tree:
        for node in targetLight.data.node_tree.nodes:
            if node.type == 'EMISSION':
                if "Strength" in node.inputs:
                    node.inputs["Strength"].default_value = targetLight.data.energy
                break
    
    # Fill light should be a complementary hue
    fillHue = wrapHue(parameters['hue'] + 0.5) 
    fillSat = parameters['saturation'] * 0.5 # Fill is usually less saturated
    fillRgb = colorsys.hsv_to_rgb(fillHue, fillSat, 1.0)
    fillLight.data.color = fillRgb
    # fill light should be set to have a 4:1 ratio
    fillLight.data.energy = targetLight.data.energy * 0.25
    if fillLight.data.type != 'SUN' and fillLight.data.node_tree:
        for node in fillLight.data.node_tree.nodes:
            if node.type == 'EMISSION':
                if "Strength" in node.inputs:
                    node.inputs["Strength"].default_value = fillLight.data.energy
                break

    # rim light should be punchy and powerful - generally a slightly different hue
    rimHue = wrapHue(parameters['hue'] + 0.1) 
    rimRgb = colorsys.hsv_to_rgb(rimHue, parameters['saturation'], 1.0)
    rimLight.data.color = rimRgb
    rimLight.data.energy = targetLight.data.energy * 0.7
    if rimLight.data.type != 'SUN' and rimLight.data.node_tree:
        for node in rimLight.data.node_tree.nodes:
            if node.type == 'EMISSION':
                if "Strength" in node.inputs:
                    node.inputs["Strength"].default_value = rimLight.data.energy
                break
    return {'FINISHED'}

class EmotionSliders(bpy.types.PropertyGroup):
    """
    This class holds the 7 slider values
    """
    # Define 7 floats. 
    # 'subtype="FACTOR"' forces the UI to look like a slider (0 to 1).
    anger: bpy.props.FloatProperty(
        name="Anger",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR'
    )
    disgust: bpy.props.FloatProperty(
        name="Disgust",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR'
    )
    fear: bpy.props.FloatProperty(
        name="Fear",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR'
    )
    joy: bpy.props.FloatProperty(
        name="Joy",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR'
    )
    sadness: bpy.props.FloatProperty(
        name="Sadness",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR'
    )
    surprise: bpy.props.FloatProperty(
        name="Surprise",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR'
    )
    neutral: bpy.props.FloatProperty(
        name="Neutral",
        default=0.0, min=0.0, max=1.0, subtype='FACTOR'
    )
    
class WM_OT_GenerateConfiguration(bpy.types.Operator):
    """Generates and applies scene parameters"""
    bl_idname = "wm.generate_configuration"
    bl_label = "Generate Scene"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.targetLight is not None

    def execute(self, context):
        generateConfig(context)
        return {'FINISHED'}

class WM_OT_LoadModel(bpy.types.Operator):
    bl_idname = "wm.load_model"
    bl_label = "Load Model"
    
    def execute(self, context):
        # Verify path exists
        path = context.scene.modelLoadingPath
        if os.path.exists(path):
            self.report({'INFO'}, "Model path verified")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Model not found")
            return {'CANCELLED'}

class VIEW3D_PT_ConfigurationGenerationPanel(bpy.types.Panel):
    """Creates a panel for all controls"""
    bl_label = "Emotion Configuration Panel"
    bl_idname = "VIEW3D_PT_emotion_configuration_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Emotion Configuration"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # add property selector for light source
        box = layout.box()
        box.label(text="Scene Setup")
        box.prop(scene, "targetLight")
        box.prop(scene, "fillLight")
        box.prop(scene, "rimLight")
        
        if scene.targetLight is None:
            layout.label(text="Please select a Target Light", icon='ERROR')
            return
        if scene.fillLight is None:
            layout.label(text="Please select a Fill Light", icon='ERROR')
            return
        if scene.rimLight is None:
            layout.label(text="Please select a Rim Light", icon='ERROR')
            return
        if len(set([scene.targetLight.data,scene.fillLight.data,scene.rimLight.data]))!=3:
            layout.label(text="All light sources must be different", icon='ERROR')

        layout.separator()
        box = layout.box()
        box.label(text="Emotion Generation")
        box.prop(scene, "modelLoadingPath")
        row = box.row()
        row.operator(WM_OT_LoadModel.bl_idname, icon='IMPORT')
        #anger, disgust, fear, joy, sadness, surprise, neutral

        layout.separator()
        layout.label(text="Emotion Inputs:")
        # Manually draw all sliders
        col = layout.column()
        for p in ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]:
            col.prop(scene.emotionProps, p)
        
        layout.separator()
        row = box.row()
        row.enabled = scene.modelLoadingPath != ""
        row.operator(WM_OT_GenerateConfiguration.bl_idname, icon="SHADING_RENDERED")

classes = (
    EmotionSliders,
    WM_OT_LoadModel,
    WM_OT_GenerateConfiguration,
    VIEW3D_PT_ConfigurationGenerationPanel,
)

def register_properties():
    bpy.types.Scene.emotionProps = bpy.props.PointerProperty(type=EmotionSliders)
    bpy.types.Scene.modelLoadingPath = bpy.props.StringProperty(
        name="Model Path",
        description="Configuration Generating Model",
        subtype='FILE_PATH',
        default=""
    )
    # Light properties
    bpy.types.Scene.targetLight = bpy.props.PointerProperty(
        name="Target Light", type=bpy.types.Object, poll=isLight
    )
    
    bpy.types.Scene.fillLight = bpy.props.PointerProperty(
        name="Fill Light", type=bpy.types.Object, poll=isLight
    )
    
    bpy.types.Scene.rimLight = bpy.props.PointerProperty(
        name="Rim Light", type=bpy.types.Object, poll=isLight
    )

def unregister_properties():
    del bpy.types.Scene.emotionProps
    del bpy.types.Scene.modelLoadingPath
    del bpy.types.Scene.targetLight
    del bpy.types.Scene.fillLight
    del bpy.types.Scene.rimLight

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_properties()

def unregister():
    unregister_properties()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

# can be run from terminal
if __name__ == "__main__":
    register()
