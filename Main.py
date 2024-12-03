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
from Clymer import randintModel, transferModelfrom Clymer import randintModel, transferModel
from Datasets import shoulders, transfer, random



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




if __name__ == "__main__":
    transferModel(1, random(1000), random(1000))
                