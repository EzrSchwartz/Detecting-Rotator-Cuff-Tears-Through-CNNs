import imgaug.augmenters as iaa
import tqdm
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import torch
import torch
import os


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
        transforms.Resize((214, 214)),
        transforms.ToTensor()
    ])
    for seed in tqdm(range(1,11)):
        seed1 = seed * np.random.randint(1, 1000)
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
                        tensor_3d = torch.stack(tensorstack, dim=0)  # Shape: [num_images, C, H, W]
                        tensor_3d = tensor_3d.permute(1, 0, 2, 3)  # Shape: [C, num_images, H, W]
                        tensor_3d = tensor_3d.unsqueeze(0)  # Add batch dimension: [1, C, num_images, H, W]
                        tensor_3d = tensor_3d.unsqueeze(1)  # Add channels dimension: [1, 1, 16, 214, 214] ✅

                        allTensors.append(tensor_3d)
        torch.save(allTensors, f'D:\Shoulderaugmented\Shoulder({seed}).pt')
        allTensors = []
    return allTensors


# import imgaug.augmenters as iaa
# import tqdm
# from tqdm import tqdm
# from torchvision import transforms
# import numpy as np
# import os
# from PIL import Image
# import torch
# import Datasets
# from torch.utils.data import Dataset, DataLoader
# seq = iaa.Sequential([
#     iaa.Affine(rotate=(-45, 45)),
#     iaa.Fliplr(0.5),
#     iaa.GaussianBlur(sigma=(0, 3.0))
# ])

# class TensorDataset(Dataset):
#     def __init__(self, tensor_list, labels):
#         """
#         :param tensor_list: List of tensors to be used by the Dataset
#         :param labels: List of corresponding labels
#         """
#         self.tensors = tensor_list
#         self.labels = labels

#     def __len__(self):
#         return len(self.tensors)

#     def __getitem__(self, idx):
#         return self.tensors[idx], self.labels[idx]

def extractImagesR(rootDirectory, outputDirectory):
    allTensors = []
    allLabels = []
    
    transform = transforms.Compose([
        transforms.Resize((214, 214)),
        transforms.ToTensor()
    ])

    # Create a dictionary to map directory names to numerical labels
    dirs = [d for d in os.listdir(rootDirectory) if os.path.isdir(os.path.join(rootDirectory, d))]
    label_dict = {dir_name: idx for idx, dir_name in enumerate(sorted(dirs))}
    
    for seed in tqdm(range(1,11)):
        seed1 = seed * np.random.randint(1, 100000)
        np.random.seed(seed1)

        tensors_for_seed = []
        labels_for_seed = []

        for root, dirs, files in os.walk(rootDirectory):
            for dir in dirs:
                tensorstack = []
                images_batch = []
                current_label = 0  # Get numerical label for current directory
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
                        tensor_3d = tensor_3d.permute(1, 0, 2, 3)

                        tensors_for_seed.append(tensor_3d)
                        labels_for_seed.append(current_label)

        # Save both tensors and labels for this seed
        torch.save({
            'tensors': tensors_for_seed,
            'labels': labels_for_seed,
            'label_dict': label_dict  # Save the label dictionary for reference
        }, f'{outputDirectory}\ShoulderNormal({seed}).pt')
        
        allTensors.extend(tensors_for_seed)
        allLabels.extend(labels_for_seed)
    
    return allTensors, allLabels, label_dict



def extractImagesT(rootDirectory,outputDirectory):
    allTensors = []
    allLabels = []
    
    transform = transforms.Compose([
        transforms.Resize((214, 214)),
        transforms.ToTensor()
    ])

    # Create a dictionary to map directory names to numerical labels
    dirs = [d for d in os.listdir(rootDirectory) if os.path.isdir(os.path.join(rootDirectory, d))]
    label_dict = {dir_name: idx for idx, dir_name in enumerate(sorted(dirs))}
    
    for seed in tqdm(range(1,11)):
        seed1 = seed * np.random.randint(1, 100000)
        np.random.seed(seed1)

        tensors_for_seed = []
        labels_for_seed = []

        for root, dirs, files in os.walk(rootDirectory):
            for dir in dirs:
                tensorstack = []
                images_batch = []
                current_label = 1  # Get numerical label for current directory
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
                        tensor_3d = tensor_3d.permute(1, 0, 2, 3)
                        # tensor_3d = tensor_3d.unsqueeze(0)

                        tensors_for_seed.append(tensor_3d)
                        labels_for_seed.append(current_label)
                        # print(f"Size of tensorstack: {tensor_3d.size()}")
                        # print(f"Size of labels_for_seed: {len(labels_for_seed)}")

        # Save both tensors and labels for this seed
        torch.save({
            'tensors': tensors_for_seed,
            'labels': labels_for_seed,
            'label_dict': label_dict  # Save the label dictionary for reference
        }, f'{outputDirectory}\ShoulderTorn({seed}).pt')
        
        allTensors.extend(tensors_for_seed)
        allLabels.extend(labels_for_seed)
    
    return allTensors, allLabels, label_dict


# # Example usage:
# if __name__ == "__main__":
#     root_dir = "path_to_your_image_directory"
#     tensors, labels, label_mapping = extractImages(root_dir)
#     print("Label mapping:", label_mapping)  # Shows which class corresponds to which number
