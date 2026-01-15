import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessPieceClassifier(nn.Module):
    def __init__(self, img_channels=3, num_classes=13):
        """
        num_classes: 13 usually (6 white + 6 black + 1 empty)
        """
        super(ChessPieceClassifier, self).__init__()

        # --- FEATURE EXTRACTOR (Mirrors your VAE Encoder) ---
        # Input: (Batch, 3, 224, 224)
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 112, 112)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 56, 56)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # -> (128, 28, 28)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # -> (256, 14, 14)
        self.bn4 = nn.BatchNorm2d(256)

        # Flatten Size: 256 channels * 14 * 14
        self.flatten_size = 256 * 14 * 14

        # --- CLASSIFICATION HEAD ---
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.dropout = nn.Dropout(0.5)  # Prevents overfitting
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Convolutional Blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
