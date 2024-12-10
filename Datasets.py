import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image


class Random3DDataset(Dataset):
    def __init__(self, num_samples=1000, depth=8, height=214, width=214, num_classes=2):
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

if __name__ == "__Datasets__":
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




def shoulders(Count):
    dataset = CustomDatasetUnsupervised(R"Path to shoulder data directory")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        if batch_idx == Count:
            return dataloader
def transfer(Count):
    dataset = CustomDatasetUnsupervised(R"Path to transfer data directory")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        if batch_idx == Count:
            return dataloader
def random(Count):
    dataset = Random3DDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        continue
    return dataloader