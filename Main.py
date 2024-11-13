import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
from tqdm import tqdm


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




class Random3DDataset(Dataset):
    def __init__(self, num_samples=1000, depth=8, height=50, width=50, num_classes=2):
        self.num_samples = num_samples
        self.depth = depth
        self.height = height
        self.width = width
        self.num_classes = num_classes
        
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            image = np.random.rand(depth, height, width)
            
            for d in range(depth):
                center_y = np.random.randint(0, height)
                center_x = np.random.randint(0, width)
                radius = np.random.randint(5, 15)
                
                y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
                mask = x*x + y*y <= radius*radius
                image[d][mask] = np.random.rand()
                
                noise = np.random.normal(0, 0.1, (height, width))
                image[d] += noise
            
            image = (image - image.min()) / (image.max() - image.min())
            
            image_tensor = torch.FloatTensor(image).unsqueeze(0) 
            
            label = np.random.randint(0, num_classes)
            
            self.data.append(image_tensor)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_tensor = preProcessing(image_path)
        if 'Tear' in image_path:
            label = 1
        else:
            label = 0

        return image_tensor, label
class CustomDatasetUnsupervised(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_tensor = preProcessing(image_path)
        
        return image_tensor


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







def randintModel():
    for epoch in range(numEpoch):
        for batch_idx, (data_input) in enumerate(TransferDataLoader):
            optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
            optimizer.step()

            for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
                outputs = model2(data_input)
                loss = F.cross_entropy(outputs, labels)
                print("Epoch:", epoch, "Loss:", loss.item())
                torch.save(model2.state_dict(), f'Path to where we save the models')
            loss = F.cross_entropy(outputs, labels)
def transferModel():
    for epoch in range(numEpoch):
        for batch_idx, (data_input) in enumerate(TransferDataLoader):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.step()

            for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
                outputs = model(data_input)
                loss = F.cross_entropy(outputs, labels)
                print("Epoch:", epoch, "Loss:", loss.item())
                torch.save(model.state_dict(), f'Path to where we save the models')
            loss = F.cross_entropy(outputs, labels)


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
    for epoch in range(numEpoch):
        for batch_idx, (data_input) in enumerate(TransferDataLoader):
            optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
            optimizer.step()

            for batch_idx, (data_input, labels) in enumerate(ShoulderDataLoader):
                outputs = model2(data_input)
                loss = F.cross_entropy(outputs, labels)
                print("Epoch:", epoch, "Loss:", loss.item())
                torch.save(model2.state_dict(), f'Path to where we save the models')
            loss = F.cross_entropy(outputs, labels)


        torch.save(model.state_dict(), f'Path to where we save the different versions of the transfer learning models')

