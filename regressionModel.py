import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

DATASETS_DIRECTORY = "./datasets/"
MODELS_DIRECTORY = "./models/"
EPOCH_COUNT = 50
BATCH_SIZE = 4

class EmotionConfigurationDataset(Dataset):
    def __init__(self, csvFile):
        self.emotionConfigurationFrames = pd.read_csv(csvFile)

    def __len__(self):
        return len(self.emotionConfigurationFrames)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        emotionVector = torch.tensor(
            self.emotionConfigurationFrames.iloc[index, :7].values, 
            dtype=torch.float32
        )
        sceneConfiguration = torch.tensor(
            self.emotionConfigurationFrames.iloc[index, 7:].values, 
            dtype=torch.float32
        )
        return {'emotion': emotionVector, 'configuration': sceneConfiguration}

class EmotionConfigurationModel(nn.Module):
    def __init__(self):
        super(EmotionConfigurationModel, self).__init__()
        self.modelLayers = nn.Sequential(
            nn.Linear(7, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )
    
    def forward(self, inputTensor):
        return self.modelLayers(inputTensor)

def saveModel(model, modelName):
    # Add file extension if not already included
    if len(modelName)<4 or modelName[-4:] != ".pth":
        modelName = modelName + ".pth"
    torch.save(model.state_dict(), MODELS_DIRECTORY + modelName)

# Note: using a loaded model for predictions, use torch.no_grad() for resource saving
# Need to check behaviour due to how AdamW optimiser operates:
# https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
def loadModel(modelName):
    if len(modelName)<4 or modelName[-4:] != ".pth":
        modelName = modelName + ".pth"
    newModel = EmotionConfigurationModel()
    newModel.load_state_dict(torch.load(MODELS_DIRECTORY + modelName))
    newModel.eval()
    return newModel

def trainModel(emotionConfigurationModel, emotionConfigurationDataset):
    trainLoader = DataLoader(
        emotionConfigurationDataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    lossCriterion = nn.MSELoss()
    #optimiser = optim.Adam(emotionConfigurationModel, lr=0.001)
    optimiser = torch.optim.AdamW(
        emotionConfigurationModel.parameters(), lr=0.001,
        weight_decay=0.01   # Critical for regularisation in small datasets
    )
    # Iterate over epochs, then over input-target pairings
    # Standard steps are:
    # Clear optimiser gradients
    # Forward pass through the model (find current prediction)
    # Calculate epoch loss based on output and target values
    # Backward pass to recalculate gradients (adjust for next epoch)
    # Update model weights
    emotionConfigurationModel.train()
    for epochNumber in range(EPOCH_COUNT):
        # calculate loss for each epoch using the lossCriterion
        epochLoss = 0.0
        for batchData in trainLoader:
            # Unpack the dictionary from the dataset
            inputParameters = batchData['emotion']
            targetValues = batchData['configuration']
            optimiser.zero_grad()
            # fwd pass
            predictedValues = emotionConfigurationModel(inputParameters)
            currentLoss = lossCriterion(predictedValues, targetValues)
            currentLoss.backward()
            optimiser.step()
            epochLoss += currentLoss.item()
        averageLoss = epochLoss / len(trainLoader)
        print(f"Epoch {epochNumber} loss: {averageLoss}")

def testModel(modelName, emotionInput):
    model = EmotionConfigurationModel()
    
    # Load model weights from file
    path = MODELS_DIRECTORY + modelName + ".pth"
    model.load_state_dict(torch.load(path))

    # Set model to evaluation mode
    model.eval()
    inputTensor = torch.tensor([emotionInput], dtype=torch.float32)
    
    # Run inference (no gradient needed)
    with torch.no_grad():
        rawOutput = model(inputTensor)
    
    return rawOutput.tolist()[0]

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

PERFORM_INFERENCE = True

if __name__ == "__main__":
    if PERFORM_INFERENCE:
        fearVector = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        fearOutput = testModel("emotionModelInitial", fearVector)
        print(mapRawOutput(fearOutput))
    else:
        csvPath = os.path.join(DATASETS_DIRECTORY, "trainingData.csv")
        
        if os.path.exists(csvPath):
            dataset = EmotionConfigurationDataset(csvPath)
            model = EmotionConfigurationModel()
            
            trainModel(model, dataset)
            
            saveModel(model, "emotionModelInitial")
        else:
            print(f"Error: CSV not found at {csvPath}")
