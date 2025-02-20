import os
import numpy as np
import torch

# Define input and output directories
input_dir = "/path/to/original/MRNet"  # Change to the actual `.npy` directory
output_dir = "/path/to/processed/data"  # Change to where you want to save tensors

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_and_save_npy():
    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    for file in files:
        file_path = os.path.join(input_dir, file)
        data = np.load(file_path)  # Load .npy file
        
        # Ensure correct shape (C, D, H, W) for 3D ConvNet
        if len(data.shape) == 3:  # (Depth, Height, Width)
            data = np.expand_dims(data, axis=0)  # Add channel dimension (C=1)

        # Convert to PyTorch tensor
        tensor_data = torch.tensor(data, dtype=torch.float32)

        # Save as a new `.pt` file in output directory
        save_path = os.path.join(output_dir, file.replace('.npy', '.pt'))
        torch.save(tensor_data, save_path)

        print(f"✅ Processed & saved: {save_path}")

# Run the conversion
process_and_save_npy()
print("🎉 All files processed and saved!")
