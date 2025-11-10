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

# Use HSV instead of RGB for colour (more cinematic, hues, saturation, and vibrancy represent emotion better)
import colorsys

# want high vibrancy for all of them so do not bother randomising
def randomHSVLighting(hmin,hmax,smin,smax):
    return colorsys.hsv_to_rgb(random.uniform(hmin, hmax), random.uniform(smin, smax), 1)
# Use exp for light energy if it isn't a sun - otherwise use uniform
# This is due to perceptual difference in intensity, use different units
# Functions to define randomisation with an emotional bias
def happyParams(targetLightType):
    print("Happy params")
    # Bright + warm colors, normal FOV, low grain
    params = {}
    params['light_r'], params['light_g'], params['light_b'] = randomHSVLighting(0.05, 0.15, 0.7, 1)
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
    params['light_r'], params['light_g'], params['light_b'] = randomHSVLighting(0.55, 0.65, 0.2, 0.5)
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
    params['light_r'], params['light_g'], params['light_b'] = randomHSVLighting(0, 0.05, 0.9, 1)
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
    params['light_r'], params['light_g'], params['light_b'] = randomHSVLighting(0.4, 0.55, 0.3, 0.6)
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

def calmParams(targetLightType):
    print("Calm params")
    # Soft colours (low saturation), wide FOV, low grain
    params = {}
    params['light_r'], params['light_g'], params['light_b'] = randomHSVLighting(0, 1, 0.1, 0.3)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(2,4)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(1000), math.log(2000)))
    params['fov'] = random.uniform(70, 90)
    params['grain'] = random.uniform(0, 0.05)
    return params

def cozyParams(targetLightType):
    print("Cozy params")
    # deep, warm colours, low FOV, low/med grain
    # Ideally this uses a point light and not a sun for 'coziness'
    params = {}
    params['light_r'], params['light_g'], params['light_b'] = randomHSVLighting(0, 0.1, 0.8, 1)
    if targetLightType == 'SUN':
        params['light_energy'] = random.uniform(0.05, 0.3)
    else:
        params['light_energy'] = math.exp(random.uniform(math.log(150), math.log(400)))
    params['fov'] = random.uniform(30, 45)
    params['grain'] = random.uniform(0, 0.15)
    return params

def amazementParams(targetLightType):
    print("Amazement params")
    # deep, cool colours, high FOV, low grain
    params = {}
    params['light_r'], params['light_g'], params['light_b'] = randomHSVLighting(0.6, 0.75, 0.8, 1)
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

parameterGenerationFunctions = [happyParams, sadParams, angerParams, fearParams, calmParams, cozyParams, amazementParams]

def randomiseConfig(context):
    """
    Applies random values to the scene and returns parameters
    """
    scene = context.scene
    targetLight = scene.targetLight
    camera = scene.camera
    if not (targetLight and targetLight.type == 'LIGHT' and camera and camera.type == 'CAMERA'):
        return
    emotionChoice = math.floor(random.random() * 7)
    parameters = parameterGenerationFunctions[emotionChoice](targetLight.data.type)

    camera.data.lens_unit = 'FOV'
    camera.data.angle = math.radians(parameters['fov'])

    if scene.render.engine == 'CYCLES':
        scene.cycles.film_noise_intensity = parameters['grain']
        scene.cycles.use_film_noise = parameters['grain'] > 0
    elif scene.render.engine == 'EEVEE':
        scene.eevee.film_noise_intensity = parameters['grain']
        scene.eevee.use_film_noise = parameters['grain'] > 0

    targetLight.data.color = (parameters['light_r'],parameters['light_g'],parameters['light_b'])
    targetLight.data.energy = parameters['light_energy']
    if targetLight.data.type != 'SUN' and targetLight.data.node_tree:
        for node in targetLight.data.node_tree.nodes:
            if node.type == 'EMISSION':
                if "Strength" in node.inputs:
                    node.inputs["Strength"].default_value = parameters['light_energy']
                break
    
    return parameters

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
        if not os.path.exists(imageDirectory):
            os.makedirs(imageDirectory)
            
        dataList = []
        
        self.report({'INFO'}, f"Generating {scene.batchImageCount} images")
        
        filepath = scene.render.filepath
        fileFormat = scene.render.image_settings.file_format

        scene.render.image_settings.file_format = 'PNG'

        for i in range(scene.batchImageCount):
            params = randomiseConfig(context)
            imageName = f"image{i:04d}.png"
            scene.render.filepath = os.path.join(imageDirectory, imageName)
            bpy.ops.render.render(write_still=True)
            
            dataEntry = {
                "image_id": imageName,
                "parameters": params,
                "labels": {}
            }
            dataList.append(dataEntry)
            
            self.report({'INFO'}, f"Rendered {imageName}")

        jsonPath = os.path.join(output_dir, "labels.json")
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
        
        if scene.targetLight is None:
            layout.label(text="Please select a Target Light", icon='ERROR')
            return

        box = layout.box()
        box.label(text="Batch Generation")
        
        box.prop(scene, "batchOutputPath")
        box.prop(scene, "batchImageCount")
        
        row = box.row()
        row.enabled = scene.batchOutputPath != ""
        row.operator(WM_OT_BatchGenerate.bl_idname)
        
        box = layout.box()
        box.label(text="3. Test")
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
