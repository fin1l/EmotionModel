import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

DATASETS_DIRECTORY = "./datasets/"
MODELS_DIRECTORY = "./models/"
EPOCH_COUNT = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001
# Constants specify output
VERBOSE_TRAINING = False
# 0 - perform learning curve extrapolation
# 1 - learning rate search
# 2 - train model normally
EXECUTION_MODE = 2

class EmotionConfigurationDataset(Dataset):
    def __init__(self, csvFile, device=torch.device("cpu"), size=-1):
        self.emotionConfigurationFrames = pd.read_csv(csvFile)
        if size == -1:
            size = len(self.emotionConfigurationFrames)
        self.emotionVectors = torch.tensor(
            self.emotionConfigurationFrames.iloc[:size, :7].values, 
            dtype=torch.float32
        ).to(device)
        self.configurationVectors = torch.tensor(
            self.emotionConfigurationFrames.iloc[:size, 7:].values, 
            dtype=torch.float32
        ).to(device)

    def __len__(self):
        return len(self.emotionVectors)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return {
            'emotion': self.emotionVectors[index], 
            'configuration': self.configurationVectors[index]
        }

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

class DeepEmotionModel(nn.Module):
    def __init__(self):
        super(DeepEmotionModel, self).__init__()
        self.modelLayers = nn.Sequential(
            nn.Linear(7, 256),
            nn.ELU(),
            nn.Linear(256, 64),
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

def trainModel(emotionConfigurationModel, trainLoader, valLoader=None, epochCount = EPOCH_COUNT):
    lossCriterion = nn.MSELoss()
    optimiser = optim.AdamW(
        emotionConfigurationModel.parameters(), lr=LEARNING_RATE,
        weight_decay=0.01   # Critical for regularisation in small datasets
    )
    emotionConfigurationModel.train()
    # Track final validation loss
    finalValLoss = 0.0
    bestLoss = math.inf
    plateauedEpochs = 0
    PLATEAU_THRESHOLD = 5
    for epochNumber in range(epochCount):
        # calculate loss for each epoch using the lossCriterion
        epochLoss = 0.0
        for batchData in trainLoader:
            # Unpack the dictionary from the dataset
            inputParameters = batchData['emotion'].to(device)
            targetValues = batchData['configuration'].to(device)
            optimiser.zero_grad()
            # fwd pass
            predictedValues = emotionConfigurationModel(inputParameters)
            currentLoss = lossCriterion(predictedValues, targetValues)
            currentLoss.backward()
            optimiser.step()
            epochLoss += currentLoss.item()
        if epochNumber % 5 == 0:
            if VERBOSE_TRAINING:
                averageLoss = epochLoss / len(trainLoader)
                print(f"Epoch {epochNumber} loss: {averageLoss}")
        if valLoader:
            finalValLoss = validateModel(emotionConfigurationModel, valLoader, lossCriterion)
            if finalValLoss < bestLoss:
                bestLoss = finalValLoss
                plateauedEpochs = 0
            else:
                plateauedEpochs += 1
                if plateauedEpochs >= PLATEAU_THRESHOLD:
                    print(f"Early termination at {epochNumber} epochs")
                    print(f"Currently at {finalValLoss}, achieved {bestLoss}")
                    break
    return finalValLoss, bestLoss

def validateModel(model, valLoader, criterion):
    model.eval() 
    valLoss = 0.0
    with torch.no_grad(): 
        for batchData in valLoader:
            inputs = batchData['emotion']
            targets = batchData['configuration']
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valLoss += loss.item()
    
    avgLoss = valLoss / len(valLoader)
    return avgLoss

def testModel(modelName, emotionInput, baseModel = None, device=torch.device("cpu")):
    if baseModel:
        model = baseModel
    else:
        model = EmotionConfigurationModel()
    model.to(device)
    
    # Load model weights from file
    path = MODELS_DIRECTORY + modelName + ".pth"
    model.load_state_dict(torch.load(path))

    # Set model to evaluation mode
    model.eval()
    inputTensor = torch.tensor([emotionInput], dtype=torch.float32).to(device)
    
    # Run inference (no gradient needed)
    with torch.no_grad():
        rawOutput = model(inputTensor)
    
    return rawOutput.cpu().tolist()[0]

def learningCurveExtrapolation(fullTrainSet, fixedTestSet):
    print("Starting Learning Curve Extrapolation\n")
    # Increments to test in
    dataFractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    resultsSize = []
    resultsLoss = []
    resultsBestLoss = []
    
    # Use a fixed test loader for fair comparison across all runs
    testLoader = DataLoader(fixedTestSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    TRIALS_PER_POINT = 2

    for fraction in dataFractions:
        subsetSize = int(len(fullTrainSet) * fraction)
        if subsetSize < BATCH_SIZE: continue
        
        trialLosses = []
        trialBestLosses = []
        
        print(f"\nTraining with {subsetSize} samples ({fraction:.2f}):")
        
        for i in range(TRIALS_PER_POINT):
            # Create different seeded data
            subset, _ = random_split(fullTrainSet, [subsetSize, len(fullTrainSet) - subsetSize])
            subsetLoader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            
            # Create new model
            currentModel = DeepEmotionModel().to(device)
            
            # Train
            finalLoss, bestLoss = trainModel(currentModel, subsetLoader, valLoader=testLoader)
            trialLosses.append(finalLoss)
            trialBestLosses.append(bestLoss)
            print(f" Trial {i+1}: {finalLoss:.5f}")
            
        # Average the trials
        avgLoss = sum(trialLosses) / TRIALS_PER_POINT
        print(f"Average Loss: {avgLoss:.5f}")
        
        resultsSize.append(subsetSize)
        resultsLoss.append(avgLoss)
        resultsBestLoss.append(sum(trialBestLosses) / TRIALS_PER_POINT)
        
    return resultsSize, resultsLoss, resultsBestLoss

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

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIRECTORY):
        os.makedirs(MODELS_DIRECTORY)
    # Use GPU as device if possible
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Get full dataset
    fullDataset = EmotionConfigurationDataset(DATASETS_DIRECTORY + "trainingData.csv", device=device, size=-1)

    # Create initial 80-20 Train-Test split
    totalSize = len(fullDataset)
    print(f"Total dataset size: {totalSize}")
    trainSize = int(0.8 * totalSize)
    print(f"Training dataset size: {trainSize}")
    testSize = totalSize - trainSize
    
    # random_split creates two Subset objects
    trainSet, testSet = random_split(fullDataset, [trainSize, testSize])
    if EXECUTION_MODE == 0:
        import matplotlib.pyplot as plt
        sizes, losses, bestLosses = learningCurveExtrapolation(trainSet, testSet)
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, losses, marker='x', linestyle='-', color='b', label='Final Validation Loss')
        plt.plot(sizes, bestLosses, marker='x', linestyle='-', color='r', label='Best Validation Loss')
        plt.title('Dataset Samples against Model Performance')
        plt.xlabel('Training Sample Count')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('learningCurve.png')
        plt.show()
    elif EXECUTION_MODE == 1:
        # Train model regularly
        currentModel = DeepEmotionModel().to(device)
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        # Train
        finalLoss, bestLoss = trainModel(currentModel, trainLoader, valLoader=testLoader)
        print(f"Final loss: {finalLoss}, Best validation loss: {bestLoss}")
    else:
        # Shuffle training data to prevent order bias, but not test data
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"Data loaded ({trainSize} training, {testSize} testing)")

        model = DeepEmotionModel().to(device=device)

        # Train Model
        print("Start Training")
        trainModel(model, trainLoader, valLoader=testLoader)
        print("Training complete")

        # Save model
        saveModel(model, "baseModel")
        print("Model saved")

        # Verify on manual input for inference
        testInput = [0.5, 0.1, 0.9, 0.2, 0.0, 0.5, 1.0] 
        
        rawResult = testModel("baseModel", testInput, baseModel=DeepEmotionModel(), device=device)
        mappedResult = mapRawOutput(rawResult)
        print(f"Inference: {mappedResult}")
