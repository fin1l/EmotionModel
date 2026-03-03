import os
import math
from matplotlib import colors
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from modelUtils import *
from sklearn.model_selection import KFold
from torch.utils.data import Subset

DATASETS_DIRECTORY = "./datasets/"
MODELS_DIRECTORY = "./models/"
EPOCH_COUNT = 25
BATCH_SIZE = 4
LEARNING_RATE = 0.001
# Constants specify output
VERBOSE_TRAINING = False
# 0 - perform learning curve extrapolation
# 1 - learning rate search
# 2 - train model normally
# 3 - k-fold validation across multiple architectures
# 4 - k-fold validation across dropout configurations
# 5 - train final model and save
EXECUTION_MODE = 4

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

class DeepEmotionModel2(nn.Module):
    def __init__(self):
        super(DeepEmotionModel2, self).__init__()
        self.modelLayers = nn.Sequential(
            nn.Linear(7, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
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

def trainModel(emotionConfigurationModel, trainLoader, valLoader=None, epochCount = EPOCH_COUNT, learningRate=LEARNING_RATE):
    lossCriterion = nn.MSELoss()
    optimiser = optim.AdamW(
        emotionConfigurationModel.parameters(), lr=learningRate,
        weight_decay=0.01 # Critical for regularisation in small datasets
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
            # Unpack emotion and configuration vectors from the dataset
            inputParameters = batchData['emotion']
            targetValues = batchData['configuration']
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
            currentModel = ImprovedDeepEmotionModel().to(device)
            
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

def learningRateEstimation(trainSet, testSet):
    print("Starting Learning Rate Estimation\n")
    # Increments to test in
    lrIncrements = [1e-1,1e-2,5e-3,1e-3,1e-4,1e-5]
    resultLRs = []
    resultsLoss = []
    resultsBestLoss = []
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for learningRate in lrIncrements:
        currentModel = ImprovedDeepEmotionModel().to(device)
        # Train
        finalLoss, bestLoss = trainModel(currentModel, trainLoader, valLoader=testLoader, learningRate=learningRate)
        resultLRs.append(learningRate)
        resultsLoss.append(finalLoss)
        resultsBestLoss.append(bestLoss)
    return resultLRs, resultsLoss, resultsBestLoss

def evaluateArchitecturesWithKFold(fullDataset, modelClasses, numFolds=5, batchSize=BATCH_SIZE, epochs=EPOCH_COUNT, device=torch.device("cpu")):
    print(f"Starting {numFolds}-Fold Cross Validation\n")
    kFold = KFold(n_splits=numFolds, shuffle=True, random_state=42)
    architectureResults = {}

    for modelName, modelClass in modelClasses.items():
        print(f"Evaluating Architecture: {modelName}")
        foldLosses = []

        # Iterate through the generated folds
        for fold, (trainIndices, valIndices) in enumerate(kFold.split(fullDataset)):
            
            # Create subsets for fold + data loaders
            trainSubset = Subset(fullDataset, trainIndices)
            valSubset = Subset(fullDataset, valIndices)
            trainLoader = DataLoader(trainSubset, batch_size=batchSize, shuffle=True, num_workers=0)
            valLoader = DataLoader(valSubset, batch_size=batchSize, shuffle=False, num_workers=0)
            currentModel = modelClass().to(device)
            
            # Take best loss (ignore final to ignore overfitting)
            finalLoss, bestLoss = trainModel(currentModel, trainLoader, valLoader=valLoader, epochCount=epochs)
            foldLosses.append(bestLoss)
            print(f"  Fold {fold + 1} Best Validation Loss: {bestLoss:.5f}")
        # Note that this is *mean* not *median* (shown in boxplot)
        architectureResults[modelName] = foldLosses
        avgLoss = sum(foldLosses) / numFolds
        print(f"Average Validation Loss for {modelName}: {avgLoss:.5f}\n")

    return architectureResults

def plotKFoldResults(architectureResults):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    modelNames = list(architectureResults.keys())
    # losses for each fold
    modelLosses = list(architectureResults.values())
    # boxplot to show variance across folds for each architecture
    bplot = plt.boxplot(modelLosses, tick_labels=modelNames, patch_artist=True)
    for patch in bplot['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
    plt.title('Architecture Performance (K-Fold Cross Validation)')
    plt.ylabel('MSE Loss')
    plt.xlabel('Model Architecture')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('kFoldArchitectureComparison.png')
    plt.show()

# Lay out results in console similarly to the architecture comparison
def evaluateDropoutConfigurations(fullDataset, dropoutConfigs, numFolds=5, batchSize=BATCH_SIZE, epochs=EPOCH_COUNT, device=torch.device("cpu")):
    print(f"Starting {numFolds}-Fold Cross Validation for Dropout Configurations\n")
    kFold = KFold(n_splits=numFolds, shuffle=True, random_state=42)
    configurationResults = {}

    for configName, rates in dropoutConfigs.items():
        print(f"Evaluating Dropout Strategy: {configName} (Rates: {rates})")
        foldLosses = []

        for fold, (trainIndices, valIndices) in enumerate(kFold.split(fullDataset)):
            trainSubset = Subset(fullDataset, trainIndices)
            valSubset = Subset(fullDataset, valIndices)
            
            trainLoader = DataLoader(trainSubset, batch_size=batchSize, shuffle=True, num_workers=0)
            valLoader = DataLoader(valSubset, batch_size=batchSize, shuffle=False, num_workers=0)
            # Uses a modifiable model to change individual dropout rates
            currentModel = ModifiableRateModel(dropoutRates=rates).to(device)
            
            finalLoss, bestLoss = trainModel(currentModel, trainLoader, valLoader=valLoader, epochCount=epochs)
            foldLosses.append(bestLoss)
            print(f"  Fold {fold + 1} Best Validation Loss: {bestLoss:.5f}")
        # Only storing mean
        avgLoss = sum(foldLosses) / numFolds
        configurationResults[configName] = avgLoss
        print(f"Average Validation Loss for {configName}: {avgLoss:.5f}\n")
        
    return configurationResults

def plotDropoutConfigurations(configurationResults):
    import matplotlib.pyplot as plt
    
    configNames = list(configurationResults.keys())
    avgLosses = list(configurationResults.values())
    
    plt.figure(figsize=(10, 6))
    # Use bar chart instead
    plt.bar(configNames, avgLosses, color='skyblue', edgecolor='black')
    plt.title('Dropout Strategy Performance (K-Fold Cross Validation)')
    plt.ylabel('Average MSE Loss')
    plt.xlabel('Dropout Configuration')
    
    # Slight rotation to stop label overlap
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('dropoutStrategyComparison.png')
    plt.tight_layout()
    plt.show()

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
    fullDataset = EmotionConfigurationDataset(DATASETS_DIRECTORY + "trainingData.csv", device=device)

    # Create initial 80-20 Train-Test split
    totalSize = len(fullDataset)
    print(f"Total dataset size: {totalSize}")
    trainSize = int(0.8 * totalSize)
    print(f"Training dataset size: {trainSize}")
    testSize = totalSize - trainSize
    
    # random_split creates two Subset objects
    trainSet, testSet = random_split(fullDataset, [trainSize, testSize])
    if EXECUTION_MODE == 0: # Plot accuracy against data set size
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
        baseModel = EmotionConfigurationModel().to(device)
        deepModel = DeepEmotionModel().to(device)
        improvedModel = ImprovedDeepEmotionModel().to(device)
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        # Train
        finalLoss, bestLoss = trainModel(baseModel, trainLoader, valLoader=testLoader)
        print(f"Base Model\nFinal loss: {finalLoss}, Best validation loss: {bestLoss}\n")
    elif EXECUTION_MODE == 2:
        import matplotlib.pyplot as plt
        lrs, losses, bestLosses = learningRateEstimation(trainSet, testSet)
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, marker='x', linestyle='-', color='b', label='Final Validation Loss')
        plt.plot(lrs, bestLosses, marker='x', linestyle='-', color='r', label='Best Validation Loss')
        plt.title('Learning Rate against Model Performance')
        plt.xlabel('Learning Rate')
        plt.xscale('log')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('learningRateCurve.png')
        plt.show()
    elif EXECUTION_MODE == 3: # k-fold validation across multiple architectures
        modelsToTest = {
            "Base Model": EmotionConfigurationModel,
            "Deep Model": DeepEmotionModel,
            "Deep Model 2": DeepEmotionModel2,
            "Refined Model": ImprovedDeepEmotionModel
        }
        
        kFoldResults = evaluateArchitecturesWithKFold(
            fullDataset=fullDataset, 
            modelClasses=modelsToTest, 
            numFolds=8, 
            batchSize=BATCH_SIZE, 
            epochs=EPOCH_COUNT, 
            device=device
        )
        
        plotKFoldResults(kFoldResults)
    elif EXECUTION_MODE == 4: # Check refined model dropouts with k-fold validation
        dropoutConfigs = {
            "Aggressive Constant": [0.4, 0.4, 0.4],
            "Moderate Constant": [0.2, 0.2, 0.2],
            "Aggressive Taper": [0.4, 0.2, 0.0],
            "Gentle Taper": [0.3, 0.2, 0.1],
            "Baseline (Current)": [0.3, 0.3, 0.2]
        }
        configResults = evaluateDropoutConfigurations(
            fullDataset=fullDataset, 
            dropoutConfigs=dropoutConfigs, 
            numFolds=5, 
            batchSize=BATCH_SIZE, 
            epochs=EPOCH_COUNT, 
            device=device
        )
        plotDropoutConfigurations(configResults)
    else: # Train and save a finalised model
        # Shuffle training data to prevent order bias, but not test data
        trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"Data loaded ({trainSize} training, {testSize} testing)")

        model = ImprovedDeepEmotionModel().to(device=device)

        # Train Model
        print("Start Training")
        trainModel(model, trainLoader, valLoader=testLoader)
        print("Training complete")

        # Save model
        saveModel(model, "emotionInferenceModel")
        print("Model saved")

        # Verify on manual input for inference
        testInput = [0.5, 0.1, 0.9, 0.2, 0.0, 0.5, 1.0] 
        loadedModel = loadModel(MODELS_DIRECTORY + "emotionInferenceModel", device=device)
        rawResult = performInference(loadedModel, testInput, device=device)
        mappedResult = mapRawOutput(rawResult)
        print(f"Inference: {mappedResult}")
