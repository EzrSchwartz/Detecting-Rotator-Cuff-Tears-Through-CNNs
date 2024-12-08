import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
from tqdm import tqdm

#Copy of the top randomly initialized model architecture developed in the previous paper
class RandomInitModelReplica(nn.Module):
    def __init__(self):
        super(RandomInitModelReplica, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3,3), stride=(1, 1,1), padding=(1, 1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2,2), stride=(2, 2,2))

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3,3), stride=(1,1,1), padding=(1, 1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2,2), stride=2,padding=(1,0,0), ceil_mode=False)

        self.fc1 = nn.Linear(in_features=self._get_fc_input_size(), out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=2)
      
    
        


    def _get_fc_input_size(self):
        with torch.no_grad():
            x = torch.randn(1, 32, 4, 25, 25)
            
            x=self.pool2(x)
            return x.numel()
    def forward(self, x):
        print(type(x))
        print(x.shape)
        x = self.conv1(x)
        print(f'conv1{x.shape}')
        x = F.relu(x)
        x = self.pool1(x)
        print(f'pool1{x.shape}')
        x = self.conv2(x)
        print(f'conv2{x.shape}')

        x = F.relu(x)
        x = self.pool2(x)
        print(f'pool2{x.shape}')
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x=self.fc1(x)
        print(f'fc1{x.shape}')
        x = F.relu(x)
        x = self.fc2(x)
        print(f'fc2{x.shape}')

        x = x.view(x.size(0), -1)

        return x






#Architecture of convolutional autoencoder used to transfer learned weights to glenoid labrum

class Convolutional_autoencoder(nn.Module):
    def __init__(self):
        super(Convolutional_autoencoder, self).__init__()

        self.conv1 = nn.Conv3d(in_channels= 1, out_channels= 64, kernel_size= (3,3,3), stride=1,padding=1 )
        self.pool1 = nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2))
        self.conv2 = nn.Conv3d(in_channels= 64, out_channels= 64,kernel_size= (3,3,3),stride=1,padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2),padding=0, ceil_mode=True)


        self.conv3 = nn.Conv3d(in_channels= 64, out_channels= 64, kernel_size= (3, 3, 3),stride= (1,1,1),padding=1)
        self.upsample1 = nn.Upsample(size= (4,26,26))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=1)
        self. upsample2 = nn.Upsample(size= (8,52,52))
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=1)
        self.output = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 3, 3), stride=1)
        
#Output size: torch.Size([1, 1, 6, 50, 52])
    def _get_fc_input_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 8, 50, 50 )
            return x.numel()



    def forward(self, x):
        print(type(x))
        print(f"Input size: {x.size()}")
        x = self.conv1(x)
        print(f"Conv1 output size: {x.size()}")
        x= F.relu(x)
        x = self.pool1(x)
        print(f"Pool1 output size: {x.size()}")
        x = F.relu(x)
        x= self.conv2(x)
        print(f"Conv2 output size: {x.size()}")
        x = F.relu(x)
        x=self.pool2(x)
        print(f"Pool2 output size: {x.size()}")
        
        x = F.relu(x)
        x = self.conv3(x)
        print(f"Conv3 output size: {x.size()}")
        x = F.relu(x)
        x = self.upsample1(x)
        print(f"Upsample1 output size: {x.size()}")
        x = F.relu(x)
        x = self.conv4(x)
        print(f"Conv4 output size: {x.size()}")
        x = F.relu(x)
        x = self.upsample2(x)
        print(f"Upsample2 output size: {x.size()}")
        x = F.relu(x)
        x = self.conv5(x)
        print(f"Conv5 output size: {x.size()}")
        x = F.relu(x)
        x = self.output(x)
        print(f"Output size: {x.size()}")
        return x

#Architecture of best performing shoulder labral tear classification model using transferred weights
class ShoulderClassificationmodel(nn.Module,):
    def __init__(self,TransferModel):
        super(ShoulderClassificationmodel,self).__init__()
        self.TransferModel = TransferModel

        self.conv1 = nn.Conv3d(in_channels= 1, out_channels=64, kernel_size=(3,3,3),stride=1,padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size= (2,2,2), stride= (2, 2, 2),ceil_mode=True)

        self.conv2 = nn.Conv3d(in_channels= 64, out_channels= 64, kernel_size= (3, 3, 3),stride=1,padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size= (2, 2, 2), stride= (2, 2, 2), ceil_mode=True)

        self.conv3 = nn.Conv3d(in_channels= 64, out_channels= 64, kernel_size= (3, 3, 3), stride=1, padding=1)
        
        self.fc1 = nn.Linear(self._get_fc_input_size(), out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)


    def _get_fc_input_size(self):
        # Define a dummy tensor with the shape of the expected input to the model
        x = torch.randn(1, 64, 2, 13, 13)  # (batch_size, channels, depth, height, width)
        
        # Pass it through the convolutional layer to get the output size
        x = self.conv3(x)
        print(x.numel())
        # Flatten the output and get the number of features for the fully connected layer
        return x.numel() 
    def forward(self, x):

        x = self.TransferModel(x)
        print(f"Input size: {x.size()}")
        x = self.conv1(x)
        print(f'conv1 output: {x.size()}')
        x = F.relu(x)
        x = self.pool1(x)
        print(f'pool1 output: {x.size()}')

        x = self.conv2(x)
        print(f'conv2 output: {x.size()}')
        x = F.relu(x)
        x = self.pool2(x)
        print(f'pool2 output: {x.size()}')

        x = self.conv3(x)
        print(f'conv3 output: {x.size()}')
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print(f'flatten output: {x.size()}')
        x=self.fc1(x)
        print(f'fc1 output: {x.size()}')
        x = F.relu(x)
        x = self.fc2(x)
        print(f'fc2 output: {x.size()}')


        x = x.view(x.size(0), -1)

        return x




def randintModel(_numEpoch, _shoulderData):
    numEpoch = _numEpoch
    shoulderData = _shoulderData
    model = RandomInitModelReplica()
    for epoch in range(numEpoch):
        for (data_input) in enumerate(shoulderData):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.step()
            torch.save(model.state_dict(), f'Path to where we save the models{epoch}')

def transferModel(_numEpoch,_TransferDataLoader,_ShoulderDataLoader):
    numEpoch = _numEpoch
    TransferDataLoader = _TransferDataLoader
    ShoulderDataLoader = _ShoulderDataLoader
    model = Convolutional_autoencoder()
    for epoch in range(numEpoch):
        for batch_idx, (transfer_data, _) in enumerate(TransferDataLoader):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
        
        # Forward pass through the autoencoder
            _ = model(transfer_data)

            optimizer.step()

    # Now, use the trained autoencoder with the classification model
            model2 = ShoulderClassificationmodel(model)
            model2_optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
            for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
                model2_optimizer.zero_grad()

                outputs = model2(data_input)
                loss = F.cross_entropy(outputs, labels)

                loss.backward()
                model2_optimizer.step()

                print("Epoch:", epoch, "Loss:", loss.item())
                torch.save(model2.state_dict(), f'Path to where we save the models')

