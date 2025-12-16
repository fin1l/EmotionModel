import bpy
import os
import sys

def runBatchProcess():
    """
    Sets up the scene parameters and triggers the batch generation operator.
    """
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    else:
        args = []

    scene = bpy.context.scene
    # can technically skip as they should be saved
    currentDir = os.getcwd()
    if len(args):
        scene.batchOutputPath = os.path.join(currentDir, args[0])
    if not scene.batchOutputPath:
        scene.batchOutputPath = os.path.join(currentDir, "batchOutput")
    if len(args)>1:
        scene.batchImageCount = int(args[1])
    else:
        scene.batchImageCount = 5
    if not scene.targetLight:
        # scene.targetLight = bpy.data.objects.get("KeyLight")
        # scene.fillLight = bpy.data.objects.get("FillLight")
        # scene.rimLight = bpy.data.objects.get("RimLight")
        print("Error: Target Light is not set in the scene.")
        return

    # 3. Execute the Operator
    print(f"Starting batch generation of {scene.batchImageCount} images...")
    
    try:
        # We call the operator string defined in bl_idname
        bpy.ops.wm.batch_generate()
        print("Batch generation completed successfully.")
    except Exception as e:
        print(f"Failed to run batch generation: {e}")

if __name__ == "__main__":
    runBatchProcess()