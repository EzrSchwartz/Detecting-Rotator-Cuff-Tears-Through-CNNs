import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

class Real3DDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = torch.load(dataset_path)  # Load .pt file

        # Ensure the dataset is structured correctly
        if isinstance(self.dataset, dict):
            self.data = self.dataset["data"]  # Assuming key is 'data'
            self.labels = self.dataset["labels"]  # Assuming key is 'labels'
        elif isinstance(self.dataset, (list, torch.Tensor)):
            self.data = self.dataset
            self.labels = None  # If no labels exist
        else:
            raise ValueError("Unsupported dataset format in .pt file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]  # Unsupervised case


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

def RealData(file):
    dataset = Real3DDataset(file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch_idx, (data_input) in enumerate(dataloader):
        continue
    return dataloader





def TransferDataLoader():
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
    #make a data loader out of the merged data
    transfer_data_loader = DataLoader(merged_data, batch_size=4, shuffle=True)
    return transfer_data_loader

def ShoulderDataLoader():
    directory = "/home/ec2-user/Shoulders"
    output_file = "/home/ec2-user/TrainingData/ShoulderData.pt"
    #make it go through the direcotry which has tons of .pt files and make a dataloader out of it
    pt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
    datasets = [torch.load(f) for f in pt_files]
    if isinstance(datasets[0], dict):  # If .pt files store dictionaries
        merged_data = {key: torch.cat([d[key] for d in datasets], dim=0) for key in datasets[0].keys()}
    elif isinstance(datasets[0], torch.Tensor):  # If .pt files store tensors directly
        merged_data = torch.cat(datasets, dim=0)
    else:
        raise ValueError("Unsupported data format in .pt files.")
    torch.save(merged_data, output_file)
    print(f"Merged dataset saved to {output_file}")
    #make a data loader out of the merged data
    shoulder_data_loader = DataLoader(merged_data, batch_size=4, shuffle=True)
    return shoulder_data_loader