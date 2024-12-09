import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

import tqdm
from tqdm import tqdm

from Clymer import randintModel, transferModel
from Datasets import shoulders, transfer, random
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
        
    # transferModel(1, random(1000), random(1000))
    UNet(1,random(1000))
             