import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
from tqdm import tqdm
from Clymer import randintModel, transferModel
from Datasets import shoulders, transfer, random
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
    # TransferModel= Convolutional_autoencoder()
    # model = ShoulderClassificationmodel(TransferModel)
    # model2 = RandomInitModelReplica()
    # print("model", model)
    # print("model2", model2)

    # Shoulderdataset = R"Path to shoulder data directory"
    # TransferDataset = R"Path to transfer data directory"

    # Shoulders = CustomDataset(Shoulderdataset)
    # Transfer = CustomDataset(TransferDataset)

    
    # Shoulders = Random3DDataset(num_samples=1000)
    # Transfer = Random3DDataset(num_samples=1000)

    # print("Datasets Made")

    # ShoulderDataLoader = DataLoader(Shoulders, batch_size=1, shuffle=True)
    # TransferDataLoader = DataLoader(Transfer, batch_size=1, shuffle=True)

    # for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
    #     continue

    # for batch_idx, (data_input) in enumerate(TransferDataLoader):
    #     continue
    # print("Transfer Data Dataloader length:", len(TransferDataLoader))

    # print("Shoulder Data Dataloader Length:", len(ShoulderDataLoader))

    # for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
    #     print(f"Batch {batch_idx}, Data shape: {data_input.shape}, Labels: {labels}")

    transferModel(1, random(1000), random(1000))
                  

#currently set up for randominit model, can change to transfer model if you replace "model2" with "model" in the for loops below
    # for epoch in range(numEpoch):
    #     for batch_idx, (data_input) in enumerate(TransferDataLoader):
    #         optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
    #         optimizer.step()

    #         for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
    #             outputs = model2(data_input)
    #             loss = F.cross_entropy(outputs, labels)
    #             print("Epoch:", epoch, "Loss:", loss.item())
    #             torch.save(model2.state_dict(), f'Path to where we save the models')
    #         loss = F.cross_entropy(outputs, labels)


    #     torch.save(model.state_dict(), f'Path to where we save the different versions of the transfer learning models')

