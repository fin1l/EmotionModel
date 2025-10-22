import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Plan is to make simple MLP model
# Start with 2 layers -> if underfitting then increase # of layers/nodes
# Use sigmoid function on output? -> make sure training data uses 0-1 range outputs
# Should likely use MSELoss for calculating error
# Adam is probably the best optimizer to use for the task with a learning rate ~0.01
DATASETS_DIRECTORY = "/datasets/"
MODELS_DIRECTORY = "/models/"
EPOCH_COUNT = 5

class EmotionConfigurationDataset(Dataset):
    def __init__(self, csvFile):
        self.emotionConfigurationFrames = pd.read_csv(csvFile)

    def __len__(self):
        return len(self.emotionConfigurationFrames)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        emotionVector = self.emotionConfigurationFrames.iloc[index, :7]
        sceneConfiguration = self.emotionConfigurationFrames.iloc[index, 7:]
        return {'emotion': emotionVector, 'configuration': sceneConfiguration}

class EmotionConfigurationModel(nn.Module):
    def __init__(self):
        super(EmotionConfigurationModel, self).__init__()
        # TODO: define model layers
        self.modelLayers = nn.Sequential()
    
    def forward(self, inputTensor):
        return self.modelLayers(inputTensor)

def saveModel(model, modelName):
    # Add file extension if not already included
    if len(modelName)<4 or modelName[-4:] != ".pth":
        modelName = modelName + ".pth"
    torch.save(model.state_dict(), MODELS_DIRECTORY + modelName)

# Note: using a loaded model for predictions, use torch.no_grad() for resource saving
def loadModel(modelName):
    if len(modelName)<4 or modelName[-4:] != ".pth":
        modelName = modelName + ".pth"
    newModel = EmotionConfigurationModel()
    newModel.load_state_dict(torch.load(MODELS_DIRECTORY + modelName))
    newModel.eval()
    return newModel

def trainModel(emotionConfigurationModel, emotionConfigurationDataset):
    lossCriterion = nn.MSELoss()
    adamOptimiser = optim.Adam(emotionConfigurationModel, lr=0.001)
    # Iterate over epochs, then over input-target pairings
    # calculate loss for each epoch using the lossCriterion
    # Standard steps are:
    # Clear optimiser gradients
    # Forward pass through the model (find current prediction)
    # Calculate epoch loss based on output and target values
    # Backward pass to recalculate gradients (adjust for next epoch)
    # Update model weights
    pass