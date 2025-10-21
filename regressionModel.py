import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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