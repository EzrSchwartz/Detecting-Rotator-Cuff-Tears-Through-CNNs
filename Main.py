import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

import tqdm
from tqdm import tqdm
from Synthetic3dDataset import Random3DDataset
from Clymer import randintModel, transferModel

# currently made for 2d images since that was what i was testing 
# def preProcessing(image_path):
#     image = Image.open(image_path)
#     image = image.convert('L')
#     image = image.resize((50, 50))
#     image_array = np.array(image)
#     image_array = image_array / 255.0
#     image_tensor = torch.tensor(image_array, dtype=torch.float32)
#     image_tensor = image_tensor.unsqueeze(0)  # Shape: (1, 50, 50)
#     return image_tensor




if __name__ == "__main__":
    # Create dataloader
    dataset = Random3DDataset()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Example of accessing the data
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"Data shape: {data.shape}")  # Should be (batch_size, 1, depth, height, width)
        print(f"Labels: {labels}")
        
        if batch_idx == 0:  # Just print first batch
            break



class Bran2d2CNN(nn.Module):
    def __init__(self, TransferModel):
        super(Bran2d2CNN, self).__init__()
        self.TransferModel = TransferModel
        
        self.fc1 = nn.Linear(in_features=TransferModel.fc_input_size, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=4)
    def forward(self, x):
        x = self.TransferModel(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    dataset = Random3DDataset()

    TransferModel= Convolutional_autoencoder()
    model = ShoulderClassificationmodel(TransferModel)
    model2 = RandomInitModelReplica()
    print("model", model)
    print("model2", model2)

    # Shoulderdataset = R"Path to shoulder data directory"
    # TransferDataset = R"Path to transfer data directory"

    # Shoulders = CustomDataset(Shoulderdataset)
    # Transfer = CustomDataset(TransferDataset)

    
    Shoulders = Random3DDataset(num_samples=1000)
    Transfer = Random3DDataset(num_samples=1000)

    print("Datasets Made")

    ShoulderDataLoader = DataLoader(Shoulders, batch_size=1, shuffle=True)
    TransferDataLoader = DataLoader(Transfer, batch_size=1, shuffle=True)

    for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
        continue

    for batch_idx, (data_input) in enumerate(TransferDataLoader):
        continue
    print("Transfer Data Dataloader length:", len(TransferDataLoader))

    print("Shoulder Data Dataloader Length:", len(ShoulderDataLoader))

    numEpoch = 1

#currently set up for randominit model, can change to transfer model if you replace "model2" with "model" in the for loops below
    randintModel(numEpoch, TransferDataLoader, ShoulderDataLoader, model2)
    transferModel(numEpoch, TransferDataLoader, ShoulderDataLoader, model)