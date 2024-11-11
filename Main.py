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

class TransferDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endsiwth('.jpg')]
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,i):
        image_path = self.image_paths[i]
        image_tensor = preProcessing(image_path)
        label = None
        return image_tensor, label
    
    
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        image_tensor = preProcessing(image_path)
        if 'gl' in image_path:
            label = 1
        elif 'no' in image_path:
            label = 0
        else:
            pass
        


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


class Brain2dCNN(nn.Module):
    def __init__(self):
        super(Brain2dCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc_input_size = self._get_fc_input_size()

    def _get_fc_input_size(self):
        with torch.no_grad():
            x = torch.randn(1, 1, 50, 50)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)

        # x = self.fc1(x)
        # x = F.relu(x)

        # x = self.fc2(x)
        return x


if __name__ == "__main__":

    imageDirectory = R"\\NASDA4788\Public\Data\Training\glioma"
    imageDirectory2 = R"\\NASDA4788\Public\Data\Training\glioma"

    dataset = CustomDataset(imageDirectory)
    dataset2 = CustomDataset(imageDirectory)
    print("Datasets Made")
    Data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    Data_loader2 = DataLoader(dataset2, batch_size=1, shuffle=True)
    for batch_idx, (data_input, labels) in enumerate(Data_loader):
        continue

    for batch_idx, (data_input, labels) in enumerate(Data_loader2):
        continue

    print("Data_loader2", len(Data_loader2))
    print("dataloader1", len(Data_loader))
    model = Brain2dCNN()
    model2 = Bran2d2CNN(model)
    print(model)
    numEpoch = 11

    for epoch in tqdm(range(numEpoch)):
        for batch_idx, (data_input, labels) in tqdm(enumerate(Data_loader)):
            outputs = model(data_input)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.step()

        if loss <= 0:
            for batch_idx, (data_input, labels) in enumerate(Data_loader2):
                outputs = model(data_input)
                loss = F.cross_entropy(outputs, labels)
                print("Epoch:", epoch, "Loss:", loss.item())
                torch.save(model.state_dict(),
                           f'\\Users\ezran\OneDrive\Desktop\Models\Modelepoch({epoch}).pth')
            loss = F.cross_entropy(outputs, labels)
        print("Epoch:", epoch, "Loss:", loss.item())
        torch.save(model.state_dict(),
                   f'\\Users\ezran\OneDrive\Desktop\Models\Modelepoch({epoch}).pth')




