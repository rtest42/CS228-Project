import torch

# 1. Check for Windows / Linux NVIDIA GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
# 2. Check for Mac Apple Silicon M1/M2/M3/M4 GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")
# 3. Fallback if no specialized hardware/drivers are found
else:
    device = torch.device("cpu")

print(f" PyTorch is running on: {device}")

# Example: Push your model and data to the active device
# model = MyModel().to(device)
