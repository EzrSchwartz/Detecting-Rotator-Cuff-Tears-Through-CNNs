import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def extractImages(rootDirectory, saveDirectory):
    # Define the augmentation pipeline
    augmenter = A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.Rotate(limit=45, p=1.0),
        A.Resize(512, 512),
        ToTensorV2()
    ])

    allTensors = []
    os.makedirs(saveDirectory, exist_ok=True)  # Create save directory if it doesn't exist

    # Process images with different seeds
    for seed in tqdm(range(1, 51)):
        seed1 = seed * np.random.randint(1, 1000000000)
        np.random.seed(seed1)

        for root, dirs, files in os.walk(rootDirectory):
            print(f"Root: {root}, Dirs: {dirs}, Files: {files}")
            for dir in tqdm(dirs):
                tensorstack = []
                images_batch = []
                dir_path = os.path.join(root, dir)
                print(f"Processing directory: {dir_path}")

                # List only images in the directory
                image_paths = [f for f in os.listdir(dir_path) if f.endswith('.png')]
                print(f"Found images: {len(image_paths)} in directory: {dir_path}")

                for image in image_paths:
                    img_path = os.path.join(dir_path, image)
                    try:
                        img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
                        img_array = np.array(img)
                        images_batch.append(img_array)
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")

                if images_batch:
                    # Apply augmentations
                    for img_array in images_batch:
                        augmented = augmenter(image=img_array)
                        tensor = augmented["image"]
                        tensorstack.append(tensor)

                    if tensorstack:
                        # Stack tensors into a 4D batch
                        tensor_3d = torch.stack(tensorstack, dim=0)
                        allTensors.append(tensor_3d)
                        print(f"Appended tensor stack of shape: {tensor_3d.shape}")

        # Save the augmented tensors for this seed
        save_path = os.path.join(saveDirectory, f'ShouldersAugmented({seed}).pt')
        print(f"Number of tensors to save: {len(allTensors)}")
        torch.save(allTensors, save_path)
        allTensors = []  # Clear tensors for the next seed

    return allTensors

# Example usage
rootDirectory = '/All'
saveDirectory = './AugmentedTensors'
extractImages(rootDirectory, saveDirectory)
