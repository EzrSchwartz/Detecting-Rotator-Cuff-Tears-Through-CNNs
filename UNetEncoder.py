import torch
import torch.nn as nn

# Encoder Module
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
        self.dec2 = self._conv_block(128, 64)
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(64, 32)
        self.final = nn.Conv3d(32, 1, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, bottleneck_output, encoder_outputs):
        enc3, enc2, enc1 = encoder_outputs
        dec1 = self.dec1(torch.cat([self.up1(bottleneck_output), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec1), enc2], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec2), enc1], dim=1))
        return self.final(dec3)

# Complete Model
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.encoder = Encoder3D()
        self.bottleneck = Bottleneck3D()
        self.decoder = Decoder3D()
    
    def forward(self, x):
        enc_outputs = self.encoder(x)
        bottleneck_output = self.bottleneck(enc_outputs[-1])
        return self.decoder(bottleneck_output, enc_outputs[:-1])

# Initialize and Test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D().to(device)
input_data = torch.rand((1, 1, 128, 128, 128), device=device)
print(f"Output shape: {model.shape}")