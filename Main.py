import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from evolveNoProp import evolve_population,evaluate_fitness
import tqdm
from tqdm import tqdm
from ImageAug import extractImages, extractImagesT, extractImagesR
# from Clymer import randintModel, transferModel
from Datasets import create_data_loader, shoulders, transfer, random, RealData, ShoulderDataLoader, TransferDataLoader
from UNetEncoder import UNet, complete_training_pipeline , evaluate_models
from resnet import UnsupervisedResNet503D, UnsupervisedLoss, train_step, train_unsupervised

# 0 is the label for Normal, 1 is the label for Tear

# if __name__ == "__main__":
#     # extractImagesT(R'\\NASDA4788\SciResearchRTCStorage\ShoulderTears\TESTING\FullTear')
#     # extractImagesR(R'\\NASDA4788\SciResearchRTCStorage\ShoulderTears\TESTING\Normal')
#     # Set random seed for reproducibility    
#     # Initialize model
#     model = UnsupervisedResNet503D(input_channels=1)
    
#     # Test the model with a dummy input
#     dummy_input = torch.randn(32, 1, 16, 214, 214)
#     output = model(dummy_input)
#     print(f"Input shape: {dummy_input.shape}")
#     print(f"Output shape: {output.shape}")
    
#     # Train the model
#     train_unsupervised(
#         model=model,
#         num_epochs=5,
#         batch_size=32,
#         learning_rate=0.001
#     )


# Directory containing .pt files
    # ShoulderDataLoader(R'\\NASDA4788\SciResearchRTCStorage\Shoulderaugmented', R'D:\ShoulderDataLoaders\shoulderdataloader.pt')
    # model_paths = ['model_joint.pth', 'model_slow.pth', 'model_frozen.pth']
    # test_loader = ShoulderDataLoader(R'D:\ShoulderAugmented', batch_size=9)
    # results = evaluate_models(model_paths, test_loader) 


# if __name__ == "__main__":
#     # complete_training_pipeline(TransferDataLoader(R"D:\ShoulderDataLoaders\transferdataloader.pt"), ShoulderDataLoader(R'D:\ShoulderAugmented'), transfer_epochs=100, classification_epochs=15, device='cuda', save_checkpoints=True)

#     test_loader = ShoulderDataLoader(R'D:\Testing', batch_size=1)
#     model_paths = ['model_joint.pth', 'model_slow.pth', 'model_frozen.pth']
#     # model_paths = ['model_checkpoint_epoch_5.pth', 'model_checkpoint_epoch_10.pth', 'model_checkpoint_epoch_15.pth']

#     # Add debug information about the data loader
#     print(f"Test loader length: {len(test_loader)}")
#     results = evaluate_models(model_paths, test_loader)
#     print(results )

# if __name__ == "__main__":
    # extractImagesR(R"D:\Shoulders\ShoulderTears\ShoulderTears\Val\0")
    # extractImagesT(R"D:\Shoulders\ShoulderTears\ShoulderTears\Val\1")

#     train_loader = ShoulderDataLoader(R'D:\Training', batch_size=1)
 
#     val_loader = ShoulderDataLoader(R'D:\Validation', batch_size=1)
#     test_loader = ShoulderDataLoader(R'D:\Testing', batch_size=1)
#     best_model = evolve_population(
# train_loader,val_loader
#     )
#     test_accuracy = evaluate_fitness(best_model, test_loader)
#     print(f"Test Accuracy of the best evolved model: {test_accuracy:.4f}")

#     # 5.4  Save the best model (optional)
#     torch.save(best_model.state_dict(), "best_evolved_model.pth")
#     print("Best model saved to best_evolved_model.pth")


if __name__ == "__main__":
    # Example usage:
    training_dir0 = R"D:\Training0" 
    training_dir1 = R"D:\Training1"  
    val_dir0 = R"D:\Validation0"
    val_dir1 = R"D:\Validation1" 

    test_dir0 = R"D:\Testing0"
    test_dir1 = R"D:\Testing1"

    # extractImagesT(R"D:\Shoulders\ShoulderTears\ShoulderTears\Tears",training_dir1)

    # extractImagesR(R"D:\Shoulders\ShoulderTears\ShoulderTears\Normal",training_dir0)
    # extractImagesR(R"D:\Shoulders\ShoulderTears\ShoulderTears\Val\0",val_dir0)
    # extractImagesT(R"D:\Shoulders\ShoulderTears\ShoulderTears\Val\1",val_dir1)
    # extractImagesR(R"D:\Shoulders\ShoulderTears\ShoulderTears\TESTING\0",test_dir0)
    # extractImagesT(R"D:\Shoulders\ShoulderTears\ShoulderTears\TESTING\1",test_dir1)
    
    
    train_loader = create_data_loader(training_dir0, training_dir1)
    val_loader = create_data_loader(val_dir0, val_dir1)
    test_loader = create_data_loader(test_dir0, test_dir1)

    data_iter = iter(train_loader)
    first_batch = next(data_iter)
    images, labels = first_batch #unpack

    # # Print out the labels:
    # print(labels)
    # print(f"Shape of the images: {images.shape}")
    # print(f"Shape of the labels: {labels.shape}")

    # # Example of iterating through the DataLoader
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"Data shape: {data.shape}")
    #     print(f"Target shape: {target.shape}")
    #     print(f"Target: {target}")
    #     #  Your training code here...
    #     if batch_idx > 2:
    #         break # only do a few batches for the example
    
    best_model = evolve_population(
train_loader,val_loader
    )
    test_accuracy = evaluate_fitness(best_model, test_loader)
    print(f"Test Accuracy of the best evolved model: {test_accuracy:.4f}")

    # 5.4  Save the best model (optional)
    torch.save(best_model.state_dict(), "best_evolved_model.pth")
    print("Best model saved to best_evolved_model.pth")
