import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

import tqdm
from tqdm import tqdm
from ImageAug import extractImages
from Clymer import randintModel, transferModel
from Datasets import shoulders, transfer, random, RealData
from UNetEncoder import UNet


if __name__ == "__main__":
    # dataset = Random3DDataset()

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # for batch_idx, (data, labels) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}")
    #     print(f"Data shape: {data.shape}")  # Should be (batch_size, 1, depth, height, width)
    #     print(f"Labels: {labels}")
        
    #     if batch_idx == 0:  # print first batch
    #         break
    extractImages(R"/home/ec2-user/ShoulderTears/All")



# Directory containing .pt files
    directory = "/home/ec2-user/Shoulders"
    output_file = "/home/ec2-user/TrainingData/TransferLearingData.pt"

# List all .pt files
    pt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]

# Load and concatenate all data
    datasets = [torch.load(f) for f in pt_files]

# Check if data is in tensor format or dictionary
    if isinstance(datasets[0], dict):  # If .pt files store dictionaries
        merged_data = {key: torch.cat([d[key] for d in datasets], dim=0) for key in datasets[0].keys()}
    elif isinstance(datasets[0], torch.Tensor):  # If .pt files store tensors directly
        merged_data = torch.cat(datasets, dim=0)
    else:
        raise ValueError("Unsupported data format in .pt files.")

# Save the merged dataset
    torch.save(merged_data, output_file)
    print(f"Merged dataset saved to {output_file}")
    
    # transferModel(1, random(1000), random(1000))
    UNet(1,RealData('/home/ec2-user/TrainingData/TransferLearingData.pt'))
