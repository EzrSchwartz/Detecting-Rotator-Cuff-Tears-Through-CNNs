import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class UnsupervisedResNet503D(nn.Module):
    def __init__(self, input_channels=1):
        super(UnsupervisedResNet503D, self).__init__()
        
        self.in_channels = 64
        
        # Modified first convolution to handle the input shape
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                              stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), 
                                  padding=(0, 1, 1))

        # ResNet layers
        self.layer1 = self._make_layer(Bottleneck3D, 64, 3)
        self.layer2 = self._make_layer(Bottleneck3D, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck3D, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck3D, 512, 3, stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512 * Bottleneck3D.expansion, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self._initialize_weights()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        output = self.projection(features)
        return output

class UnsupervisedLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(UnsupervisedLoss, self).__init__()
        self.temperature = temperature

    def forward(self, outputs):
        # Implement your unsupervised loss here
        # This is a placeholder - modify according to your specific needs
        return F.mse_loss(outputs, torch.zeros_like(outputs))

def train_step(model, optimizer, data, loss_fn, device):
    model.train()
    optimizer.zero_grad()
    
    data = data.to(device)
    outputs = model(data)
    loss = loss_fn(outputs)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

from Datasets import ShoulderDataLoader


def train_unsupervised(model, num_epochs=100, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_loader = ShoulderDataLoader(
        directory=R'D:\Shoulderaugmented',
        batch_size=batch_size,
        num_workers=4
    )
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Ensure correct shape: [batch_size, channels, depth, height, width]
            if len(batch.shape) != 5:
                continue  # Skip malformed batches
                
            batch = batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Define target
            target = torch.zeros_like(outputs).to(device)
            target[:, 0] = 1.0
            
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_count}], '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / batch_count
        print(f'Epoch [{epoch+1}/{num_epochs}] complete, '
              f'Average Loss: {epoch_loss:.4f}')
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'unsupervised_checkpoint_epoch_{epoch+1}.pth')


