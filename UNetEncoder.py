import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import collections
from typing import Tuple, Dict, List
import os

# If you need to import your model classes

# Optional imports for visualization/logging if needed
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.enc1 = self._conv_block(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        return enc1, enc2, enc3

# Bottleneck Module
class Bottleneck3D(nn.Module):
    def __init__(self):
        super(Bottleneck3D, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.bottleneck(x)

# Decoder Module
class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block2(128, 64)

        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self._conv_block2(64, 32)
        self.final = nn.Conv3d(32, 1, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def _conv_block1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(2,2,2), padding=(1,4,4)),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(2,1,1), padding=(1,3,3)),
            nn.ReLU(inplace=True),
        )
    
    def _conv_block2(self, in_channels, out_channels):
        return nn.Sequential(
        # Upsample spatial dimensions (doubling size)
        nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        nn.ReLU(inplace=True),
        
        # Reduce the number of channels
        nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        nn.ReLU(inplace=True)
    )

    
    def forward(self, bottleneck_output, encoder_outputs):
        enc3, enc2, enc1 = encoder_outputs


        dec1 = self.dec1(bottleneck_output)
        
        dec2 = self.dec2(dec1)
        dec2 = F.pad(dec2, (0, 1, 0, 1))  # Pad right and bottom by 1

        # dec2 = nn.tran((1,64,4,25,25))

        dec3 = self.dec3(dec2)
        return dec1,dec2,dec3,self.final(dec3)

# Complete Model
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.encoder = Encoder3D()
        self.bottleneck = Bottleneck3D()
        self.decoder = Decoder3D()
    
    def forward(self, x):
        # Get encoder outputs
        enc_outputs = self.encoder(x)
        enc1 = enc_outputs[0]  # Get encoder features
        enc2 = enc_outputs[1]
        enc3 = enc_outputs[2]
        bottleneck_output = self.bottleneck(enc_outputs[-1])

        # print(bottleneck_output.shape)
        # # Get decoder outputs at each level
        # dec1 = self.decoder.dec1(torch.cat([self.decoder.up1(bottleneck_output)], dim=1))
        # dec2 = self.decoder.dec2(torch.cat([self.decoder.up2(dec1), enc2], dim=1))
        # dec3 = self.decoder.dec3(torch.cat([self.decoder.up3(dec2), enc1], dim=1))
        # final_output = self.decoder.final(dec3)
        
        outputs= self.decoder(bottleneck_output, (enc3, enc2, enc1))
    
    # If you still want to calculate MSE losses:
        dec1 = outputs[0]  # Adjust these indices based on what your decoder returns
        dec2 = outputs[1]
        dec3 = outputs[2]
        final_output = outputs[3]



        mse_loss1 = torch.nn.functional.mse_loss(dec1, enc3)  # Deepest level

        mse_loss2 = torch.nn.functional.mse_loss(dec2, enc2)  # Middle level
        
        mse_loss3 = torch.nn.functional.mse_loss(dec3, enc1)  # Shallow level

        mse_loss4 = torch.nn.functional.mse_loss(final_output, x)  # Final output vs input

        return final_output, mse_loss1,mse_loss2,mse_loss3,mse_loss4


def UNet(_numEpoch,_TrainingData):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def UNet(_numEpoch, _TrainingData):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = nn.DataParallel(UNet3D()).to(device)
        TrainingDataLoader = _TrainingData
        numEpoch = _numEpoch
        
        for epoch in tqdm(range(numEpoch)):
            for batch_idx, (TrainingData,_) in tqdm(enumerate(TrainingDataLoader)):
                optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)
                optimizer.zero_grad()
    
                # Ensure TrainingData has correct shape before passing to the model
                TrainingData = TrainingData.squeeze()  # Remove extra dimensions

                # Ensure batch and channel dimensions exist
                if TrainingData.dim() == 4:  # If shape is [batch, depth, height, width]
                    TrainingData = TrainingData.unsqueeze(1)  # Add channel dimension -> [batch, 1, depth, height, width]
                elif TrainingData.dim() == 3:  # If shape is [depth, height, width]
                    TrainingData = TrainingData.unsqueeze(0).unsqueeze(0)  # Add batch & channel -> [1, 1, depth, height, width]

                print("Fixed TrainingData Shape:", TrainingData.shape)  # Should be [batch, 1, 16, 214, 214]

                # Pass corrected tensor into the model
                _ = model(TrainingData)

                optimizer.step()
            print(F'epoch:{epoch}')
            print(f"MSE1: {_[1]}")
            print(f"MSE2: {_[2]}")
            print(f"MSE3: {_[3]}")
            print(f"MSE4: {_[4]}")
    
        # Extract the encoder from the trained model
        trained_encoder = model.module.encoder
        
        # Save only the encoder state dict
        encoder_state_dict = trained_encoder.state_dict()
        torch.save(encoder_state_dict, 'encoder_model.pth')
        print("Encoder model saved to encoder_model.pth")
        
        return trained_encoder
    
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import copy








class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# class ClassificationHead(nn.Module):
#     def __init__(self, input_channels=128, num_classes=2):
#         super(ClassificationHead, self).__init__()
        
#         # ResNet blocks
#         self.res_layers = nn.Sequential(
#             ResBlock(input_channels, 256, stride=2),
#             ResBlock(256, 512, stride=2),
#             ResBlock(512, 512)
#         )
        
#         # Global Average Pooling
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
#         # Fully Connected layers
#         self.fc_layers = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )
        
#         # Initialize weights
#         self._initialize_weights()
        
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         # ResNet blocks
#         x = self.res_layers(x)
        
#         # Global Average Pooling
#         x = self.avg_pool(x)
        
#         # Flatten
#         x = x.view(x.size(0), -1)
        
#         # FC layers
#         x = self.fc_layers(x)
        
#         return x

# class EncoderWithClassifier(nn.Module):
#     def __init__(self, encoder, num_classes=2):
#         super(EncoderWithClassifier, self).__init__()
#         self.encoder = encoder
#         self.classifier = ClassificationHead(input_channels=128, num_classes=num_classes)
        
#     def forward(self, x):
#         # Get encoder output
#         enc_outputs = self.encoder(x)
#         final_encoder_output = enc_outputs[2]  # assuming this is enc3
        
#         # Pass through classification head
#         classification_output = self.classifier(final_encoder_output)
#         return classification_output


class ClassificationHead(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(ClassificationHead, self).__init__()
        
        # Add Batch Normalization and Dropout for better regularization
        self.res_layers = nn.Sequential(
            nn.BatchNorm3d(input_channels),
            ResBlock(input_channels, 1438208, stride=2),
            nn.Dropout3d(0.3),
            ResBlock(2, 1438208, stride=2),
            nn.Dropout3d(0.3),
            ResBlock(512, 512)
        )
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Modified FC layers with better regularization
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # Final layer with adjusted initialization
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Use a smaller standard deviation for initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Initialize the final layer with smaller weights
                if m.out_features == 2:  # Final classification layer
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.res_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# class EncoderWithClassifier(nn.Module):
#     def __init__(self, encoder, num_classes=2):
#         super(EncoderWithClassifier, self).__init__()
#         self.encoder = encoder
#         # Adjust num_classes to match your labels (1 and 2)
#         self.classifier = ClassificationHead(input_channels=128, num_classes=2)
        
#     def forward(self, x):
#         enc_outputs = self.encoder(x)
#         final_encoder_output = enc_outputs[2]  # using enc3
#         classification_output = self.classifier(final_encoder_output)
#         return classification_output

#     def train_step(self, batch, device):
#         self.train()
#         x = batch
#         x = x.to(device)
#         # # Adjust labels to be 0-based (subtract 1 from labels)
#         # y = y.to(device) - 1  # Convert labels from [1,2] to [0,1]
        
#         outputs = self(x)
#         # Use CrossEntropyLoss with class weights to handle imbalance
#         weights = torch.tensor([0.64, 0.36]).to(device)  # Adjusted for your class distribution
#         criterion = nn.CrossEntropyLoss(weight=weights)
#         # loss = criterion(outputs, y)
        
#         return 0


# class EncoderWithClassifier(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#         print("made encoder")
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 512),  # Assuming encoder output is 128-dimensional
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         # Handle tuple output from encoder
#         encoder_output = self.encoder(x)
#         if isinstance(encoder_output, tuple):
#             # Take the last element of the tuple (usually the features)
#             features = encoder_output[-1]
#         else:
#             features = encoder_output
            
#         # Ensure features are properly shaped
#         if features.dim() > 2:
#             features = features.view(features.size(0), -1)
            
#         return self.classifier(features)
    
#     def train_step(self, batch, device):
#         print("TrwinStep")
#         batch = batch.to(device)
#         try:
#             outputs = self(batch)
            
#             # Create target of ones (assuming all samples are positive)
#             targets = torch.ones(outputs.size()).to(device)
            
#             # BCE Loss
#             loss = F.binary_cross_entropy(outputs, targets)
            
#             return loss
#         except Exception as e:
#             print(f"Error in train_step: {str(e)}")
#             print(f"Batch shape: {batch.shape}")
#             return None


# def evaluate_full_model(model_path, test_loader, device='cuda'):
#     """
#     Evaluate the full model and show class distribution for labels 1 and 2
#     """
#     print("\nStarting evaluation...")
    
#     # Initialize model with correct dimensions
#     encoder = Encoder3D()  # Your encoder class
#     model = EncoderWithClassifier(encoder)
    
#     # Load the saved state dict
#     checkpoint = torch.load(model_path, map_location=device)
    
#     # Get the saved dimensions from the checkpoint
#     saved_input_dim = checkpoint['state_dict']['classifier.0.weight'].size(1)
#     current_input_dim = model.classifier[0].weight.size(1)
    
#     print(f"Saved model input dimension: {saved_input_dim}")
#     print(f"Current model input dimension: {current_input_dim}")
    
#     # Recreate the model with correct dimensions
#     model = EncoderWithClassifier(encoder)
#     model.classifier[0] = nn.Linear(saved_input_dim, 512)  # Use saved dimensions
    
#     # Now load the state dict
#     try:
#         model.load_state_dict(checkpoint['state_dict'])
#     except:
#         # If the above fails, try loading without 'state_dict' key
#         model.load_state_dict(checkpoint)
    
#     model = model.to(device)
#     model.eval()
    
#     # Initialize counters
#     true_labels_count = {1: 0, 2: 0}
#     predicted_count = {1: 0, 2: 0}
    
#     with torch.no_grad():
#         for batch in tqdm(test_loader):
#             batch = batch.to(device)
            
#             try:
#                 outputs = model(batch)
#                 # Convert outputs to binary predictions (threshold at 0.5)
#                 predictions = (outputs >= 0.5).float()
                
#                 # Update prediction counts
#                 predicted_count[1] += (predictions == 0).sum().item()
#                 predicted_count[2] += (predictions == 1).sum().item()
                
#             except Exception as e:
#                 print(f"Error processing batch: {str(e)}")
#                 continue
    
#     print("\nResults:")
#     print("--------")
#     print("Model Predictions Distribution:")
#     print(f"Predicted Class 1: {predicted_count[1]} samples")
#     print(f"Predicted Class 2: {predicted_count[2]} samples")

#     return predicted_count




def evaluate_full_model(model_path, test_loader, device='cuda'):
    """
    Evaluate the full model and show class distribution for labels 1 and 2
    """
    print("\nStarting evaluation...")
    
    # Initialize model with correct dimensions
    encoder = Encoder3D()  # Your encoder class
    model = EncoderWithClassifier(encoder)
    
    # Load the saved state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get the saved dimensions from the checkpoint
    saved_input_dim = checkpoint['classifier.0.weight'].size(1)  # Removed 'state_dict'
    current_input_dim = model.classifier[0].weight.size(1)
    
    print(f"Saved model input dimension: {saved_input_dim}")
    print(f"Current model input dimension: {current_input_dim}")
    
    # Recreate the model with correct dimensions
    model = EncoderWithClassifier(encoder)
    model.classifier[0] = nn.Linear(saved_input_dim, 512)  # Use saved dimensions
    
    # Now load the state dict
    model.load_state_dict(checkpoint)  # Load directly
    
    model = model.to(device)
    model.eval()
    
    # Initialize counters
    predicted_count = {1: 0, 2: 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            
            try:
                outputs = model(batch)
                # Convert outputs to binary predictions (threshold at 0.5)
                predictions = (outputs >= 0.5).float()
                
                # Update prediction counts
                predicted_count[1] += (predictions == 0).sum().item()
                predicted_count[2] += (predictions == 1).sum().item()
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue
    
    print("\nResults:")
    print("--------")
    print("Model Predictions Distribution:")
    print(f"Predicted Class 1: {predicted_count[1]} samples")
    print(f"Predicted Class 2: {predicted_count[2]} samples")

    return predicted_count

# class EncoderWithClassifier(nn.Module):
#     def __init__(self, encoder, input_dim=128):  # Set to the correct flattened dimension
#         super().__init__()
#         self.encoder = encoder
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         # Print input shape for debugging
#         batch_size = x.size(0)
#         print(f"Input batch shape: {x.shape}")
        
#         # Get encoder output
#         encoder_output = self.encoder(x)
        
#         # Handle tuple output from encoder
#         if isinstance(encoder_output, tuple):
#             features = encoder_output[-1]
#         else:
#             features = encoder_output
            
#         print(f"Features shape before flatten: {features.shape}")
        
#         # Flatten the features while preserving batch dimension
#         features = features.view(batch_size, -1)
#         print(f"Features shape after flatten: {features.shape}")
        
#         # Pass through classifier
#         return self.classifier(features)
class EncoderWithClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Calculate the flattened size
        # For input shape [1, 1, 16, 214, 214] that produces [1, 128, 4, 53, 53]
        flattened_size = 128 * 4 * 53 * 53  # = 1438208

        # Add a reduction layer before the classifier
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),  # Reduce from 1438208 to 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Print input shape for debugging
        batch_size = x.size(0)
        print(f"Input batch shape: {x.shape}")
        
        # Get encoder output
        encoder_output = self.encoder(x)
        
        # Handle tuple output from encoder
        if isinstance(encoder_output, tuple):
            features = encoder_output[-1]
        else:
            features = encoder_output
            
        print(f"Features shape before flatten: {features.shape}")
        
        # Flatten the features while preserving batch dimension
        features = features.view(batch_size, -1)
        print(f"Features shape after flatten: {features.shape}")
        
        # Pass through classifier
        return self.classifier(features)

from resnet import UnsupervisedResNet503D

def evaluate_full_model(model_path, test_loader, device='cuda'):
    print("\nStarting evaluation...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print("Checkpoint keys:", checkpoint.keys())
        
        # Initialize model with correct dimensions

        encoder = Encoder3D()
        model = EncoderWithClassifier(encoder)  # Set fixed dimension
        # model = UnsupervisedResNet503D(input_channels=1)
        # Print model structure
        print("\nModel structure:")
        print(model)
        
        # Load state dict
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        
        # Initialize counters
        predicted_count = {1: 0, 2: 0}
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                if batch is None or len(batch.shape) != 5:
                    print(f"Skipping invalid batch shape: {batch.shape if batch is not None else None}")
                    continue
                    
                try:
                    batch = batch.to(device)
                    outputs = model(batch)
                    
                    # Convert outputs to binary predictions (threshold at 0.5)
                    predictions = (outputs >= 0.5).float()
                    
                    # Update prediction counts
                    predicted_count[1] += (predictions == 0).sum().item()
                    predicted_count[2] += (predictions == 1).sum().item()
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    print(f"Batch shape: {batch.shape}")
                    continue
        
        print("\nResults:")
        print("--------")
        print("Model Predictions Distribution:")
        print(f"Predicted Class 1: {predicted_count[1]} samples")
        print(f"Predicted Class 2: {predicted_count[2]} samples")
        
        return predicted_count
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_models(model_paths, test_loader, device='cuda'):
    """
    Evaluate multiple models
    """
    results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        try:
            results[model_path] = evaluate_full_model(model_path, test_loader, device)
        except Exception as e:
            print(f"Error evaluating {model_path}: {str(e)}")
            continue
    
    return results

# Usage




def train_encoder_classifier(encoder, train_loader, num_epochs, mode='slow', 
                           learning_rate_encoder=1e-5, learning_rate_classifier=1e-3, 
                           device='cuda'):
    """
    Train the encoder with classifier using BCE loss for unsupervised learning
    """
    model = EncoderWithClassifier(encoder).to(device)
    
    # Separate parameters for encoder and classifier
    encoder_params = model.encoder.parameters()
    classifier_params = model.classifier.parameters()
    
    # Create optimizer with different parameter groups
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': learning_rate_encoder},
        {'params': classifier_params, 'lr': learning_rate_classifier}
    ], weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=3)
    
    print(f"\nTraining in {mode} mode:")
    print(f"Encoder learning rate: {learning_rate_encoder}")
    print(f"Classifier learning rate: {learning_rate_classifier}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        model.train()
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            if batch is None or len(batch.shape) != 5:
                print(f"Skipping batch with incorrect shape: {batch.shape if batch is not None else None}")
                continue
                
            try:
                # Move batch to device
                batch = batch.to(device)
                
                # Training step
                optimizer.zero_grad()
                loss = model.train_step(batch, device)
                
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Print batch loss occasionally
                    if batch_count % 10 == 0:
                        print(f"\nBatch {batch_count} Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue
        
        # Calculate average loss for the epoch
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            scheduler.step(avg_loss)
            print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f'model_{mode}_checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'mode': mode
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
    
    return model

  
# def train_encoder_classifier(encoder, train_loader, num_epochs, mode='joint', learning_rate_encoder=1e-4, 
#                            learning_rate_classifier=1e-3, device='cuda'):
#     """
#     Train the encoder and classifier in different modes
    
#     Args:
#         encoder: Pre-trained encoder
#         train_loader: DataLoader containing the training data
#         num_epochs: Number of epochs to train
#         mode: One of ['joint', 'slow', 'frozen']
#         learning_rate_encoder: Learning rate for encoder
#         learning_rate_classifier: Learning rate for classifier
#         device: Device to train on
#     """
    
#     # Move model to device
#     model = EncoderWithClassifier(encoder).to(device)
#     criterion = nn.CrossEntropyLoss()
    
#     # Configure optimizers based on mode
#     if mode == 'frozen':
#         # Freeze encoder
#         for param in model.encoder.parameters():
#             param.requires_grad = False
#         optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate_classifier)
    
#     elif mode == 'slow':
#         # Separate optimizers with different learning rates
#         encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate_encoder)
#         classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate_classifier)
        
#     else:  # joint
#         # Single optimizer for both networks
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_encoder)
    
#     # Training loop
#     for epoch in tqdm(range(num_epochs)):
#         model.train()
       
        
#         for batch_idx, inputs in enumerate(train_loader):
#             inputs= inputs.to(device)
            
#             # Zero the parameter gradients
#             if mode == 'slow':
#                 encoder_optimizer.zero_grad()
#                 classifier_optimizer.zero_grad()
#             else:
#                 optimizer.zero_grad()
            
#             # Forward pass
            
#             # Backward pass and optimize
            
#             if mode == 'slow':
#                 encoder_optimizer.step()
#                 classifier_optimizer.step()
#             else:
#                 optimizer.step()
            
#             # Statistics
           

# #     return model
# def train_encoder_classifier(encoder, train_loader, num_epochs, device='cuda'):
#     model = EncoderWithClassifier(encoder).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-10, weight_decay=1e-5)
#     #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
#     for epoch in range(num_epochs):
#         total_loss = 0
#         model.train()
#         for batch in tqdm(train_loader):
#             print("HERE")
#             if len(batch.shape) != 5:
#                     print(f"Skipping batch with incorrect shape: {batch.shape}")
#                     continue
#             optimizer.zero_grad()
#             loss = model.train_step(batch, device)
#             # loss.backward()
#             optimizer.step()
#             # total_loss += loss.item()
            
#         # # avg_loss = total_loss / len(train_loader)
#         # scheduler.step(avg_loss)
#         print(f"Epoch {epoch+1},")
        
#         # Save checkpoint
#         if (epoch + 1) % 5 == 0:
#             torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')
    
#     return model

# Function to run all three training scenarios
def train_all_scenarios(encoder, train_loader, num_epochs_per_mode=10, device='cuda'):
    """
    Run all three training scenarios sequentially   
    """
    # 1. Joint training
    print("Starting joint training...")
    model_joint = train_encoder_classifier(
        copy.deepcopy(encoder),
        train_loader,
        num_epochs_per_mode,
        mode='joint',
        learning_rate_encoder=1e-4,
        learning_rate_classifier=1e-3,
        device=device
    )
    torch.save(model_joint.state_dict(), 'model_joint.pth')
    
    # 2. Slow encoder training
    print("\nStarting slow encoder training...")
    model_slow = train_encoder_classifier(
        copy.deepcopy(encoder),
        train_loader,
        num_epochs_per_mode,
        mode='slow',
        learning_rate_encoder=1e-5,  # Slower learning rate for encoder
        learning_rate_classifier=1e-3,
        device=device
    )
    torch.save(model_slow.state_dict(), 'model_slow.pth')
    
    # 3. Frozen encoder training
    print("\nStarting frozen encoder training...")
    model_frozen = train_encoder_classifier(
        copy.deepcopy(encoder),
        train_loader,
        num_epochs_per_mode,
        mode='frozen',
        learning_rate_classifier=1e-3,
        device=device
    )
    torch.save(model_frozen.state_dict(), 'model_frozen.pth')
    
    return model_joint, model_slow, model_frozen

# Usage example:
"""
# First load your pretrained encoder
encoder = load_encoder()

# Prepare your data loader
train_loader = ... # Your DataLoader here

# Run all three training scenarios
model_joint, model_slow, model_frozen = train_all_scenarios(
    encoder,
    train_loader,
    num_epochs_per_mode=10
)
"""

def complete_training_pipeline(transfer_data_loader, shoulder_data_loader, 
                             transfer_epochs=10, classification_epochs=10,
                             device='cuda', save_checkpoints=True):
    """
    Complete training pipeline that:
    1. Trains the encoder using transfer learning on general medical data
    2. Uses the trained encoder for three different classification scenarios on shoulder data
    
    Args:
        transfer_data_loader: DataLoader for transfer learning phase
        shoulder_data_loader: DataLoader for shoulder classification phase
        transfer_epochs: Number of epochs for transfer learning phase
        classification_epochs: Number of epochs for each classification scenario
        device: Device to train on ('cuda' or 'cpu')
        save_checkpoints: Whether to save model checkpoints
    
    Returns:
        Dictionary containing all trained models and training history
    """
    print("Starting complete training pipeline...")
    
    # Initialize training history
    # history = {
    #     'transfer_learning': {'loss': []},
    #     'joint_training': {'loss': [], 'accuracy': []},
    #     'slow_training': {'loss': [], 'accuracy': []},
    #     'frozen_training': {'loss': [], 'accuracy': []}
    # }
    
    try:
    #     # 1. Initial UNet Training Phase
    #     print("\nPhase 1: Training encoder with transfer learning...")
    #     model = nn.DataParallel(UNet3D()).to(device)
        
    #     # Training loop for transfer learning
    #     for epoch in tqdm(range(transfer_epochs), desc="Transfer Learning"):
    #         running_loss = 0.0
            
    #         for batch_idx, data in enumerate(transfer_data_loader):
    #             data = data.to(device)
                
    #             optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    #             optimizer.zero_grad()
                
    #             outputs = model(data)
    #             # Sum all MSE losses
    #             loss = sum(outputs[1:])  # outputs[1:] contains all MSE losses
                
    #             loss.backward()
    #             optimizer.step()
                
    #             running_loss += loss.item()
                
    #             if batch_idx % 500 == 499:
    #                 avg_loss = running_loss / 50
    #                 print("Average Loss:", avg_loss)
    #                 history['transfer_learning']['loss'].append(avg_loss)
                    
    #                 running_loss = 0.0
        
    #     # Extract and save the trained encoder
    #     trained_encoder = model.module.encoder
    #     if save_checkpoints:
    #         torch.save(trained_encoder.state_dict(), 'trained_encoder.pth')
    #         print("Saved trained encoder checkpoint")
        
        # 2. Classification Training Phase
        # 
        # Run all three classification scenarios
        # First create a new encoder instance
        encoder = Encoder3D()

# Load the saved state dictionary
        encoder_state_dict = torch.load(R"D:\ShoulderDataLoaders\trained_encoder.pth")
        encoder.load_state_dict(encoder_state_dict)
        print("Loaded trained encoder")
# Now pass the loaded encoder to train_all_scenarios
        models = train_all_scenarios(
            encoder=encoder,
            train_loader=shoulder_data_loader,
            num_epochs_per_mode=classification_epochs,
            device=device
        )

        # models = train_all_scenarios(
        #     encoder=R"D:\ShoulderDataLoaders\trained_encoder.pth",
        #     train_loader=shoulder_data_loader,
        #     num_epochs_per_mode=classification_epochs,
        #     device=device
        # )
        
        # Package results
        results = {
            'trained_encoder': encoder,
            'joint_model': models[0],
            'slow_model': models[1],
            'frozen_model': models[2],
        }
        
        print("\nTraining pipeline completed successfully!")
        
        return results
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

# def evaluate_models(results, test_loader, device='cuda'):
#     """
#     Evaluate all trained models on test data with additional metrics
#     """
#     models = {
#         'joint': results['joint_model'],
#         # 'slow': results['slow_model'],
#         # 'frozen': results['frozen_model']
#     }
    
#     evaluation_results = {}
    
#     for name, model in models.items():
#         model = EncoderWithClassifier()
#         model.load_state_dict(torch.load(f'{name}_model.pth'))
#         model = model.to(device)
#         model.eval()
#         correct = 0
#         total = 0
#         all_predictions = []
#         all_labels = []
        
#         with torch.no_grad():
#             for data, labels in test_loader:
#                 data, labels = data.to(device), labels.to(device)
#                 outputs = model(data)
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
                
#                 # Store predictions and labels for additional metrics
#                 all_predictions.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         # Calculate metrics
#         accuracy = 100. * correct / total
#         precision = precision_score(all_labels, all_predictions, average='weighted')
#         recall = recall_score(all_labels, all_predictions, average='weighted')
#         f1 = f1_score(all_labels, all_predictions, average='weighted')
        
#         evaluation_results[name] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1
#         }
        
#         print(f'{name.capitalize()} Model Results:')
#         print(f'  Accuracy: {accuracy:.2f}%')
#         print(f'  Precision: {precision:.2f}')
#         print(f'  Recall: {recall:.2f}')
#         print(f'  F1 Score: {f1:.2f}\n')
    
#     return evaluation_results


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from pathlib import Path

def validate_tensor_dimensions(tensor, expected_shape=(1,1,1, 16, 214, 214)):
    """
    Validates if a single tensor (before batching) has the correct dimensions.
    
    Args:
        tensor: The input tensor to validate
        expected_shape: The expected shape (channels, depth, height, width)
    
    Returns:
        bool: True if tensor dimensions are valid, False otherwise
    """
    try:
        # Remove batch dimension if present
        if tensor.dim() == 5:
            tensor = tensor.squeeze(0)
            
        # Check if dimensions match expected shape
        if tensor.shape != expected_shape:
            print(f"Invalid tensor shape: {tensor.shape}, expected: {expected_shape}")
            return False
            
        # Check if tensor contains valid values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print("Tensor contains NaN or Inf values")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating tensor: {str(e)}")
        return False

class IterableTensorDataset(Dataset):
    """
    Dataset that loads and filters tensor files iteratively
    """
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.file_list = []
        self.sample_map = []
        self.valid_samples = []
        
        # Scan directory and validate samples
        print("Validating dataset...")
        for file_path in tqdm(Path(tensor_dir).glob('*.pt')):
            try:
                data = torch.load(file_path, map_location='cpu')
                num_samples = len(data['tensors'])
                
                # Validate each sample in the file
                for sample_idx in range(num_samples):
                    tensor = data['tensors'][sample_idx]
                    if validate_tensor_dimensions(tensor):
                        self.valid_samples.append((str(file_path), sample_idx))
                
                self.file_list.append(str(file_path))
                
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
                continue
        
        print(f"Found {len(self.valid_samples)} valid samples out of {len(self.sample_map)} total samples")
        
        self.current_file = None
        self.current_file_data = None

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        file_path, sample_idx = self.valid_samples[idx]
        
        # Load new file only if necessary
        if self.current_file != file_path:
            self.current_file = file_path
            self.current_file_data = torch.load(file_path, map_location='cpu')
        
        # Get the specific sample from the current file
        tensor = self.current_file_data['tensors'][sample_idx]
        label = self.current_file_data['labels'][sample_idx]
        
        # Ensure correct shape
        tensor = tensor.squeeze()
        if tensor.dim() == 3:  # If shape is [16, 214, 214]
            tensor = tensor.unsqueeze(0)  # Add channel dimension -> [1, 16, 214, 214]
        
        return tensor, torch.tensor(label)

def create_filtered_dataloader(tensor_dir, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader that filters out invalid tensors
    """
    dataset = IterableTensorDataset(tensor_dir)
    
    if len(dataset) == 0:
        raise ValueError("No valid samples found in the dataset!")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True)
    
    return dataloader



def create_augmented_dataloader(tensor_dir, batch_size=1, shuffle=True, num_workers=0):
    """
    Create a DataLoader that loads data iteratively
    """
    dataset = IterableTensorDataset(tensor_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader

def evaluate_models(model_paths, test_loader, device='cuda'):
    """
    Evaluate multiple models
    """
    results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        try:
            results[model_path] = evaluate_full_model(model_path, test_loader, device)
        except Exception as e:
            print(f"Error evaluating {model_path}: {str(e)}")
            continue
    
    return results

# Usage


    # import psutil
    # import gc
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # tensor_dir = "D:\Testing"  # Adjust path as needed

    # # Create test dataloader
    # test_loader = create_filtered_dataloader(
    #     tensor_dir=tensor_dir,
    #     batch_size=1,
    #     shuffle=False
    # )
    
    # # Evaluate model
    # model_paths = ["model_joint.pth","model_slow.pth","model_frozen.pth"]
    # try:
    #     for model_path in model_paths:
    #         print(f"\nEvaluating model: {model_path}")
    #         results = evaluate_full_model(
    #             model_path=model_path,
    #             test_loader=test_loader,
    #             device=device
    #         )
    #         print("\nEvaluation completed successfully!")
    #         print("Results:", results)
     
    # except Exception as e:
    #     print(f"Error during evaluation: {str(e)}")
    
    # def print_memory_usage():
    #     process = psutil.Process()
    #     print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    

# Usage example:
"""
# Prepare your dataloaders
transfer_loader = ... # DataLoader for transfer learning phase
shoulder_loader = ... # DataLoader for shoulder classification
test_loader = ... # DataLoader for testing

# Run complete pipeline
results = complete_training_pipeline(
    transfer_data_loader=transfer_loader,
    shoulder_data_loader=shoulder_loader,
    transfer_epochs=10,
    classification_epochs=10
)

# Evaluate all models
evaluation_results = evaluate_models(results, test_loader)

# Access individual models if needed
trained_encoder = results['trained_encoder']
joint_model = results['joint_model']
slow_model = results['slow_model']
frozen_model = results['frozen_model']

# Access training history
history = results['history']
"""
