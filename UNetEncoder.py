import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm 
from tqdm import tqdm
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

class ClassificationHead(nn.Module):
    def __init__(self, input_channels=128, num_classes=2):
        super(ClassificationHead, self).__init__()
        
        # ResNet blocks
        self.res_layers = nn.Sequential(
            ResBlock(input_channels, 256, stride=2),
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512)
        )
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully Connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ResNet blocks
        x = self.res_layers(x)
        
        # Global Average Pooling
        x = self.avg_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc_layers(x)
        
        return x

class EncoderWithClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(EncoderWithClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = ClassificationHead(input_channels=128, num_classes=num_classes)
        
    def forward(self, x):
        # Get encoder output
        enc_outputs = self.encoder(x)
        final_encoder_output = enc_outputs[2]  # assuming this is enc3
        
        # Pass through classification head
        classification_output = self.classifier(final_encoder_output)
        return classification_output
    
def train_encoder_classifier(encoder, train_loader, num_epochs, mode='joint', learning_rate_encoder=1e-4, 
                           learning_rate_classifier=1e-3, device='cuda'):
    """
    Train the encoder and classifier in different modes
    
    Args:
        encoder: Pre-trained encoder
        train_loader: DataLoader containing the training data
        num_epochs: Number of epochs to train
        mode: One of ['joint', 'slow', 'frozen']
        learning_rate_encoder: Learning rate for encoder
        learning_rate_classifier: Learning rate for classifier
        device: Device to train on
    """
    
    # Move model to device
    model = EncoderWithClassifier(encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Configure optimizers based on mode
    if mode == 'frozen':
        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate_classifier)
    
    elif mode == 'slow':
        # Separate optimizers with different learning rates
        encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate_encoder)
        classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate_classifier)
        
    else:  # joint
        # Single optimizer for both networks
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_encoder)
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            if mode == 'slow':
                encoder_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            if mode == 'slow':
                encoder_optimizer.step()
                classifier_optimizer.step()
            else:
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, '
                      f'Loss: {running_loss / 100:.3f}, '
                      f'Accuracy: {100. * correct / total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0
    
    return model

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
    history = {
        'transfer_learning': {'loss': []},
        'joint_training': {'loss': [], 'accuracy': []},
        'slow_training': {'loss': [], 'accuracy': []},
        'frozen_training': {'loss': [], 'accuracy': []}
    }
    
    try:
        # 1. Initial UNet Training Phase
        print("\nPhase 1: Training encoder with transfer learning...")
        model = nn.DataParallel(UNet3D()).to(device)
        
        # Training loop for transfer learning
        for epoch in tqdm(range(transfer_epochs), desc="Transfer Learning"):
            running_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(transfer_data_loader):
                data = data.to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
                optimizer.zero_grad()
                
                outputs = model(data)
                # Sum all MSE losses
                loss = sum(outputs[1:])  # outputs[1:] contains all MSE losses
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 50 == 49:
                    avg_loss = running_loss / 50
                    history['transfer_learning']['loss'].append(avg_loss)
                    print(f'Transfer Learning - Epoch: {epoch + 1}, '
                          f'Batch: {batch_idx + 1}, Loss: {avg_loss:.3f}')
                    running_loss = 0.0
        
        # Extract and save the trained encoder
        trained_encoder = model.module.encoder
        if save_checkpoints:
            torch.save(trained_encoder.state_dict(), 'trained_encoder.pth')
            print("Saved trained encoder checkpoint")
        
        # 2. Classification Training Phase
        print("\nPhase 2: Starting classification scenarios...")
        
        # Run all three classification scenarios
        models = train_all_scenarios(
            encoder=trained_encoder,
            train_loader=shoulder_data_loader,
            num_epochs_per_mode=classification_epochs,
            device=device
        )
        
        # Package results
        results = {
            'trained_encoder': trained_encoder,
            'joint_model': models[0],
            'slow_model': models[1],
            'frozen_model': models[2],
            'history': history
        }
        
        print("\nTraining pipeline completed successfully!")
        
        return results
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
def evaluate_models(results, test_loader, device='cuda'):
    """
    Evaluate all trained models on test data
    """
    models = {
        'joint': results['joint_model'],
        'slow': results['slow_model'],
        'frozen': results['frozen_model']
    }
    
    evaluation_results = {}
    
    for name, model in models.items():
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        evaluation_results[name] = accuracy
        print(f'{name.capitalize()} Model - Test Accuracy: {accuracy:.2f}%')
    
    return evaluation_results

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
