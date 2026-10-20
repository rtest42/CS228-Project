import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()

        # 1. The CNN Backbone: Extracts spatial ridges and minutiae features
        self.backbone = nn.Sequential(
            # Input: 1 channel (Grayscale fingerprint), Output: 16 feature maps
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduces image size by half

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 2. Fully Connected Head: Outputs a clean 128-dimensional embedding
        # (Assuming your input images are resized to 128x128. After two max pools, it becomes 32x32)
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128) # Final embedding size
        )

    def forward_once(self, x):
        # Pass one image through the shared network
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # Flatten the CNN output for the linear layer
        x = self.fc(x)
        return x

    def forward(self, img1, img2):
        # Pass both images through the exact same weights
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2
