import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create the model and move it to GPU
model = SimpleNN().to(device)

# Create dummy input and move to GPU
dummy_input = torch.randn(5, 10).to(device)  # Batch size 5, input size 10

# Perform a forward pass
output = model(dummy_input)

# Print output tensor
print("Output:", output)
