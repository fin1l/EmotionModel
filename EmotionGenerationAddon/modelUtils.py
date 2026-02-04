import bpy
import sys
import os
import subprocess
import json
import math

def performInference():
    """
    Runs the inference as a background process
    """
    scene = bpy.context.scene
    # Get worker script from add on directory
    addOnDirectory = os.path.dirname(os.path.realpath(__file__))
    workerScript = os.path.join(addOnDirectory, "inferenceWorker.py")
    cmd = ["python", workerScript, scene.modelLoadingPath]

    if scene.useTextInput:
        inputText = scene.emotionTextInput
        cmd.append(inputText)
    else:
        emotionSliders = scene.emotionProps
        emotionInput = [emotionSliders.anger, emotionSliders.disgust, emotionSliders.fear,
                    emotionSliders.joy, emotionSliders.sadness, emotionSliders.surprise, emotionSliders.neutral]
        # Add the emotion inputs to the call
        for val in emotionInput:
            cmd.append(str(val))

    # Run the inference as a subprocess, getting the output from the console
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=addOnDirectory
        )
        workerOutput = result.stdout.strip()
        # Output from worker
        if result.returncode != 0:
            print(f"Worker Crash: {result.stderr}")
            return [0.0] * 6
            
        try:
            data = json.loads(workerOutput)
        except json.JSONDecodeError:
            print(f"Worker returned invalid JSON: {workerOutput}")
            return [0.0] * 6

        if data.get("status") == "success":
            return data["data"]
        else:
            print(f"Inference Error: {data.get('message')}")
            return [0.0] * 6

    except Exception as e:
        print(f"Subprocess Execution Failed: {e}")
        return [0.0] * 6

def mapRawOutput(rawOutput):
    #Using the constants from the data processing:
    #PARAMETER_MIN_MAX = {
    #    "hueSin": [-1, 1],
    #    "hueCos": [-1, 1],
    #    "saturation": [0.2, 1],
    #    "light_energy": [math.log(50), math.log(3000)],
    #    "grain": [0, 0.4],
    #    "fov": [25, 100]
    #}
    parameters = {}
    # Calculate hue
    hSin, hCos = rawOutput[0] * 2 - 1, rawOutput[1] * 2 - 1
    parameters["hue"] = math.atan2(hSin, hCos) / (2 * math.pi)
    if parameters["hue"] < 0: parameters["hue"] += 1
    parameters["saturation"] = rawOutput[2] * 0.8 + 0.2
    # Using log laws, log 3000 - log 50 = log 60
    parameters["lightEnergy"] = math.exp(rawOutput[3] * math.log(60) + math.log(50))
    parameters["grain"] = rawOutput[4] * 0.4
    parameters["fov"] = rawOutput[5] * 75 + 25
    return parameters
