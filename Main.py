import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import tqdm
from tqdm import tqdm

def preProcessing(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.resize((50, 50))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)  # Shape: (1, 50, 50)
    return image_tensor

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_tensor = preProcessing(image_path)
        if 'gl' in image_path:
            label = 1
        elif 'me' in image_path:
            label = 2
        elif 'pi' in image_path:
            label = 3
        else:
            label = 0

        return image_tensor, label


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


#Copy of the top randomly initialized model architecture developed in the previous paper
class RandomInitModelReplica(nn.Module):
    def __init__(self):
        super(RandomInitModelReplica, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3,3), stride=(1, 1,1), padding=(1, 1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2,2), stride=(2, 2))

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3,3), stride=(1, 1,1), padding=(1, 1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2,2), stride=(2,2,2))
        self.fc_input_size = self._get_fc_input_size()
        self.fc1 = nn.Linear(in_features=self.fc_input_size, out_features=16,stride=(1,1,1))
        self.fc2 = nn.Linear(in_features=16, out_features=4, stride = (1,1,1))
      
    
        


    def _get_fc_input_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 50, 50,8)
            

            return x.numel()
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x=self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)


        x = x.view(x.size(0), -1)

        return x






#Architecture of convolutional autoencoder used to transfer learned weights to glenoid labrum\

class Convolutional_autoencoder(nn.Module):
    def __init__(self):
        super(Convolutional_autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels= 1, out_channels= 64, kernel_size= (3,3,3), stride= (1,1,1) ),
            nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2)),
            nn.Conv3d(in_channels= 64, out_channels= 64,kernel_size= (3,3,3)),
            nn.MaxPool3d(kernel_size= (2,2,2), stride= (2,2,2)),
            nn.Conv3d(in_channels= 64, out_channels= 64, kernel_size= (3, 3, 3),stride= (1,1,1)),
            nn.Upsample(size=(64,26, 26, 4)),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.Upsample(size=(64, 52, 52, 8)),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.Conv3d(in_channels=64,out_channels=1,kernel_size=(3,3,1), stride=(1, 1, 1), padding=(0,0, 0))
            )
            
        
    def forward(self, x):
        x = self.encoder(x)
        return x

#Architecture of best performing shoulder labral tear classification model using transferred weights
class ShoulderClassificationmodel(nn.Module,):
    def __init__(self,transferModel):

        self.transferModel = transferModel()
        self.conv1 = nn.Conv3d(in_channels= self.transferModel.fc_input_size, out_channels=64)
        self.pool1 = nn.MaxPool3d(kernel_size= (2,2,2), stride= (2, 2, 2))

        self.conv2 = nn.Conv3d(in_channels= 64, out_channels= 64, kernel_size= (3, 3, 3),stride=(1,1,1),padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size= (2, 2, 2), stride= (2, 2, 2))

        self.conv3 = nn.Conv3d(in_channels= 64, out_channels= 64, kernel_size= (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.fc_input_size = self._get_fc_input_size()

        self.fc1 = nn.Linear(in_features=self.fc_input_size, kernel_size= (1,1), out_features=64,stride=(1, 1, 1))
        self.fc2 = nn.Linear(in_features=64, out_features=2, stride = (1, 1, 1))


    def _get_fc_input_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 50, 50, 8)


            return x.numel()
    def forward(self, x):
        x = self.transferModel(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x=self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)


        x = x.view(x.size(0), -1)

        return x










if __name__ == "__main__":

    # imageDirectory = R"/home/ezra/SciResearch/CNN-Labral-Tears/Data/Training"
    # imageDirectory2 = R"/home/ezra/SciResearch/CNN-Labral-Tears/Data/Testing"

    # dataset = CustomDataset(imageDirectory)
    # dataset2 = CustomDataset(imageDirectory)
    # Data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Data_loader2 = DataLoader(dataset2, batch_size=1, shuffle=True)
    # for batch_idx, (data_input, labels) in enumerate(Data_loader):
    #     continue

    # for batch_idx, (data_input, labels) in enumerate(Data_loader2):
    #     continue

    # print("Data_loader2", len(Data_loader2))
    # print("dataloader1", len(Data_loader))
    transmodel= Convolutional_autoencoder()
    model = ShoulderClassificationmodel(transmodel)
    model2 = RandomInitModelReplica()
    print("model", model)
    print("model2", model2)
    # numEpoch = 11

    # for epoch in tqdm(range(numEpoch)):
    #     for batch_idx, (data_input, labels) in tqdm(enumerate(Data_loader)):
    #         outputs = model(data_input)

    #         loss = F.cross_entropy(outputs, labels)
    #         loss.backward()
    #         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #         optimizer.step()

    #     if loss <= 0:
    #         for batch_idx, (data_input, labels) in tqdm(enumerate(Data_loader2)):
    #             outputs = model(data_input)
    #             loss = F.cross_entropy(outputs, labels)
    #             print("Epoch:", epoch, "Loss:", loss.item())
    #             torch.save(model.state_dict(), f'/home/ezra/SciResearch/CNN-Labral-Tears/CNN-LabralTears/Models/model-epoch({epoch}).pth')
    #         loss = F.cross_entropy(outputs, labels)
    #     print("Epoch:", epoch, "Loss:", loss.item())
    #     torch.save(model.state_dict(), f'/home/ezra/SciResearch/CNN-Labral-Tears/CNN-LabralTears/Models/model-epoch({epoch}).pthl.lll
