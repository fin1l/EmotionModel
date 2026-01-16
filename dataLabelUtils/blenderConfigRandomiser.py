bl_info = {
    "name": "Configuration Randomiser",
    "author": "",
    "version": (1, 0),
    "blender": (2, 80, 0),  # require Blender 2.80 or newer
    "location": "View3D > Sidebar (N Panel) > Random Lights",
    "description": "Randomizes the configuration",
    "warning": "",
    "doc_url": "",
    "category": "Lighting",
}

import bpy
import math
import random
import os
import json
import re # used for extracting numbers

# The expected format of lighting is a 3-point light set up with a key light, rim light, and fill light
# Use HSV instead of RGB for colour (more cinematic, hues, saturation, and value represent emotion better)
import colorsys

# want high value for all of them so do not bother randomising value
def wrapHue(hue):
    return hue % 1.0

def randomHueSat(hMin, hMax, sMin, sMax):
    return random.uniform(hMin, hMax), random.uniform(sMin, sMax)
# Use exp for light energy if it isn't a sun - otherwise use uniform
# This is due to perceptual difference in intensity, use different units
# Functions to define randomisation with an emotional bias
def joyParams(targetLightType):
    print("Happy params")
    # Bright + warm colors, normal FOV, low grain
    params = {}
    params['h'], params['s'] = randomHueSat(0.05, 0.15, 0.7, 1)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(2,4)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(800), math.log(1500)))
    params['fov'] = random.uniform(40, 60)
    params['grain'] = random.uniform(0.0, 0.1)
    return params

def acceptanceParams(targetLightType):
    print("Acceptance params")
    # Soft yellows + greens, normal FOV, low grain
    params = {}
    params['h'], params['s'] = randomHueSat(0.13, 0.23, 0.3, 0.55)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(2,4)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(800), math.log(1500)))
    params['fov'] = random.uniform(40, 60)
    params['grain'] = random.uniform(0.0, 0.1)
    return params

def sadParams(targetLightType):
    print("Sad params")
    # Cool + dark colours, normal/wide FOV, medium grain
    params = {}
    params['h'], params['s'] = randomHueSat(0.55, 0.65, 0.3, 0.6)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(0.1,0.5)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(100), math.log(300)))
    params['fov'] = random.uniform(50, 80)
    params['grain'] = random.uniform(0.1, 0.25)
    return params

def angerParams(targetLightType):
    print("Angry params")
    # Red colours, low FOV, high grain
    params = {}
    params['h'], params['s'] = randomHueSat(0.95, 1.05, 0.9, 1)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(0.8,1.5)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(500), math.log(1000)))
    params['fov'] = random.uniform(25, 35)
    params['grain'] = random.uniform(0.2, 0.4)
    return params

def fearParams(targetLightType):
    print("Fear params")
    # Sickly + dark colours, extreme FOV, high grain
    params = {}
    params['h'], params['s'] = randomHueSat(0.4, 0.55, 0.4, 0.7)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(0.05,0.2)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(50), math.log(200)))
    # randomly pick extreme
    if random.random()>0.5:
        params['fov'] = random.uniform(25, 35)
    else:
        params['fov'] = random.uniform(90, 100)
    params['grain'] = random.uniform(0.2, 0.4)
    return params

def disgustParams(targetLightType):
    print("Disgust params")
    # Sickly + dark colours, extreme FOV, high grain
    params = {}
    # Choose either sickly green or sickly purple
    if random.random() > 0.5:
        params['h'], params['s'] = randomHueSat(0.18, 0.28, 0.6, 0.9)
    else:
        params['h'], params['s'] = randomHueSat(0.75, 0.83, 0.4, 0.7)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(0.05,0.2)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(50), math.log(200)))
    # randomly pick extreme
    if random.random()>0.5:
        params['fov'] = random.uniform(25, 35)
    else:
        params['fov'] = random.uniform(90, 100)
    params['grain'] = random.uniform(0.2, 0.4)
    return params

def expectationParams(targetLightType):
    print("Expectation params")
    # Soft colours (low saturation), low FOV, low grain
    params = {}
    params['h'], params['s'] = randomHueSat(0.5, 0.7, 0.2, 0.4)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(2,4)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(1000), math.log(2000)))
    params['fov'] = random.uniform(30, 50)
    params['grain'] = random.uniform(0, 0.05)
    return params

def surpriseParams(targetLightType):
    print("Surprise params")
    # deep, cool colours, high FOV, low grain
    params = {}
    params['h'], params['s'] = randomHueSat(0.6, 0.75, 0.8, 1)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(3, 5)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(1500), math.log(3000)))
    params['fov'] = random.uniform(80, 100)
    params['grain'] = random.uniform(0, 0.1)
    return params

# A poll function to ensure only Light objects can be selected
def isLight(self, object):
    return object.type == 'LIGHT'

parameterGenerationFunctions = [fearParams, angerParams, joyParams, sadParams, acceptanceParams, disgustParams, expectationParams, surpriseParams, ]

def setupNoiseCompositor(scene, grain_intensity):
    if grain_intensity <= 0:
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
    
    # reset nodes
    #if tree.nodes:
    #    tree.nodes.clear()
    
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
    alphaOverNode.inputs[2].default_value = 0.1 * grain_intensity
    
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

def randomiseConfig(context):
    """
    Applies random values to the scene and returns parameters
    """
    scene = context.scene
    targetLight = scene.targetLight
    fillLight = scene.fillLight
    rimLight = scene.rimLight
    camera = scene.camera
    if not (targetLight and targetLight.type == 'LIGHT' and camera and camera.type == 'CAMERA'
            and fillLight and fillLight.type == 'LIGHT' and rimLight and rimLight.type == 'LIGHT'):
        return
    emotionChoice = math.floor(random.random() * 7)
    parameters = parameterGenerationFunctions[emotionChoice](targetLight.data.type)

    camera.data.lens_unit = 'FOV'
    camera.data.angle = math.radians(parameters['fov'])
    setupNoiseCompositor(scene, parameters['grain'])
    
    targetRGB = colorsys.hsv_to_rgb(wrapHue(parameters['h']), parameters['s'], 1.0)
    targetLight.data.color = targetRGB
    targetLight.data.energy = parameters['light_energy']
    if targetLight.data.type != 'SUN' and targetLight.data.node_tree:
        for node in targetLight.data.node_tree.nodes:
            if node.type == 'EMISSION':
                if "Strength" in node.inputs:
                    node.inputs["Strength"].default_value = targetLight.data.energy
                break
    
    # Fill light should be a complementary hue
    fillHue = wrapHue(parameters['h'] + 0.5) 
    fillSat = parameters['s'] * 0.5 # Fill is usually less saturated
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
    rimHue = wrapHue(parameters['h'] + 0.1) 
    rimRgb = colorsys.hsv_to_rgb(rimHue, parameters['s'], 1.0)
    rimLight.data.color = rimRgb
    rimLight.data.energy = targetLight.data.energy * 0.7
    if rimLight.data.type != 'SUN' and rimLight.data.node_tree:
        for node in rimLight.data.node_tree.nodes:
            if node.type == 'EMISSION':
                if "Strength" in node.inputs:
                    node.inputs["Strength"].default_value = rimLight.data.energy
                break
    
    # Circular hue can make it difficult for a model to predict colour
    # 0.0 is the same as 1.0 - this can be fixed by predicting sin + cos of hue and converting back
    # Conversion back is hue = math.atan2(hueSin, hueCos) * 2PI
    outputParams = {}
    outputParams['hueSin'] = math.sin(2 * math.pi * parameters['h'])
    outputParams['hueCos'] = math.cos(2 * math.pi * parameters['h'])
    outputParams['saturation'] = parameters['s']
    outputParams['light_energy'] = parameters['light_energy']
    outputParams['grain'] = parameters['grain']
    outputParams['fov'] = parameters['fov']
    return outputParams

# Only exists for testing
class WM_OT_RandomiseConfiguration(bpy.types.Operator):
    """Randomizes the scene to test parameters"""
    bl_idname = "wm.randomise_configuration"
    bl_label = "Randomise Scene (Test)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.targetLight is not None

    def execute(self, context):
        randomiseConfig(context)
        return {'FINISHED'}

class WM_OT_BatchGenerate(bpy.types.Operator):
    """Generates and renders a batch of images and a JSON file for data mapping"""
    bl_idname = "wm.batch_generate"
    bl_label = "Start Batch Generation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return scene.targetLight is not None and scene.batchOutputPath != ""

    def execute(self, context):
        scene = context.scene
        output_dir = bpy.path.abspath(scene.batchOutputPath)
        imageDirectory = os.path.join(output_dir, "images")
        jsonPath = os.path.join(output_dir, "labels.json")
        # Prioritise getting the most recent json label, fall back to existing image
        numberMatchingPattern = re.compile("\d+")
        dataList = []
        lastFileNum = -1
        if os.path.exists(jsonPath):
            # Assume the most recent image_id is the highest
            # This should be the case
            with open(jsonPath, 'r') as f:
                dataList = json.load(f)
            fileName = dataList[-1]["image_id"]
            lastFileNum = int(numberMatchingPattern.search(fileName)[0])
        if not os.path.exists(imageDirectory):
            os.makedirs(imageDirectory)
        else:
            # Find max image number existing
            for fileName in os.listdir(imageDirectory):
                if re.fullmatch("image(\d+)\.png", fileName):
                    lastFileNum = max(lastFileNum, int(numberMatchingPattern.search(fileName)[0]))
        # Set the starting index to count images from
        startInd = lastFileNum + 1
        
        
        self.report({'INFO'}, f"Generating {scene.batchImageCount} images")
        
        filepath = scene.render.filepath
        fileFormat = scene.render.image_settings.file_format

        scene.render.image_settings.file_format = 'PNG'

        for i in range(scene.batchImageCount):
            params = randomiseConfig(context)
            imageName = f"image{startInd+i}.png"
            scene.render.filepath = os.path.join(imageDirectory, imageName)
            bpy.ops.render.render(write_still=True)
            
            dataEntry = {
                "image_id": imageName,
                "parameters": params,
                "labels": {}
            }
            dataList.append(dataEntry)
            
            self.report({'INFO'}, f"Rendered {imageName}")

        with open(jsonPath, 'w') as f:
                json.dump(dataList, f, indent=2)
        scene.render.filepath = filepath
        scene.render.image_settings.file_format = fileFormat
        
        self.report({'INFO'}, f"Data generation complete and saved to {jsonPath}")
        return {'FINISHED'}

class VIEW3D_PT_RandomConfigurationPanel(bpy.types.Panel):
    """Creates a panel for all controls"""
    bl_label = "Random Configuration Panel"
    bl_idname = "VIEW3D_PT_random_configuration_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Randomised Configuration"

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

        box = layout.box()
        box.label(text="Batch Generation")
        
        box.prop(scene, "batchOutputPath")
        box.prop(scene, "batchImageCount")
        
        row = box.row()
        row.enabled = scene.batchOutputPath != ""
        row.operator(WM_OT_BatchGenerate.bl_idname)
        
        box = layout.box()
        box.label(text="Testing")
        box.operator(WM_OT_RandomiseConfiguration.bl_idname)

classes = (
    WM_OT_RandomiseConfiguration,
    WM_OT_BatchGenerate,
    VIEW3D_PT_RandomConfigurationPanel,
)

def register_properties():
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

    bpy.types.Scene.batchOutputPath = bpy.props.StringProperty(
        name="Output Folder",
        description="Folder for renders and data labelling",
        subtype='DIR_PATH',
        default=""
    )
    bpy.types.Scene.batchImageCount = bpy.props.IntProperty(
        name="Number of Images",
        description="Images generated per batch",
        default=10, min=1, max=10000
    )

def unregister_properties():
    del bpy.types.Scene.targetLight
    del bpy.types.Scene.fillLight
    del bpy.types.Scene.rimLight
    del bpy.types.Scene.batchOutputPath
    del bpy.types.Scene.batchImageCount

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
