import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import math

class ImprovedDeepEmotionModel(nn.Module):
    def __init__(self):
        super(ImprovedDeepEmotionModel, self).__init__()
        
        self.modelLayers = nn.Sequential(
            nn.Linear(7, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.3), # 30% dropout chance
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.3),

            # Step down: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.2), # Reduced dropout
            # Step down: 64 -> 32
            nn.Linear(64, 32),
            nn.ELU(),

            nn.Linear(32, 6),
            nn.Sigmoid()
        )
    
    def forward(self, inputTensor):
        return self.modelLayers(inputTensor)


def loadModel(modelName, device=torch.device("cpu")):
    if len(modelName)<4 or modelName[-4:] != ".pth":
        modelName = modelName + ".pth"
    newModel = ImprovedDeepEmotionModel().to(device)
    newModel.load_state_dict(torch.load(modelName))
    newModel.eval()
    return newModel

# Note: using a loaded model for predictions, use torch.no_grad() for resource saving
def performInference(loadedModel, emotionInput, device=torch.device("cpu")):
    inputTensor = torch.tensor([emotionInput], dtype=torch.float32).to(device)
    
    # Run inference (no gradient needed)
    with torch.no_grad():
        rawOutput = loadedModel(inputTensor)
    
    return rawOutput.cpu().tolist()[0]

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
