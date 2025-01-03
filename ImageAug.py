import imgaug.augmenters as iaa
import tqdm
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import torch

seq = iaa.Sequential([
    iaa.Affine(rotate=(-45, 45)),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])

import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class
class TensorDataset(Dataset):
    def __init__(self, tensor_list):
        """
        :param tensor_list: List of tensors to be used by the Dataset.
        """
        self.tensors = tensor_list

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]


def extractImages(rootDirectory):
    allTensors = []
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    for seed in tqdm(range(1,51)):
        seed1 = seed * np.random.randint(1, 1000000000)
        np.random.seed(seed1)

        for root, dirs, files in os.walk(rootDirectory):
            for dir in dirs:
                tensorstack = []
                images_batch = []
                image_paths = os.listdir(os.path.join(rootDirectory, dir))
                
                for image in image_paths:
                    img_path = os.path.join(rootDirectory, dir, image)
                    if image.endswith(".jpg") or image.endswith(".png"):
                        try:
                            img = Image.open(img_path)
                            img_array = np.array(img)
                            images_batch.append(img_array)
                        except Exception as e:
                            print(f"Error processing image {img_path}: {e}")
                
                if images_batch:
                    augmenter = seq.to_deterministic() 
                    augmented_images = augmenter(images=images_batch)
                    
                    for aug_img in augmented_images:
                        aug_pil = Image.fromarray(aug_img)
                        tensor = transform(aug_pil)
                        tensorstack.append(tensor)
                    
                    if tensorstack:
                        tensor_3d = torch.stack(tensorstack, dim=0)
                        tensor_3d = tensor_3d.permute(1, 0, 2, 3).unsqueeze(0)
                        allTensors.append(tensor_3d)
        torch.save(allTensors, f'D:\Shoulders\ShouldersAugmented({seed}).pt')
        allTensors = []
    return allTensors
extractImages(R"D:\ShoulderTears\All")

