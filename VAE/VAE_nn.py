import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from VAE.dataset import FilenameDataset


import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
LATENT_DIM = 20  # Size of the compressed "bottleneck"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super(VAE, self).__init__()

        # --- ENCODER ---
        # Input: (Batch, 3, 224, 224)
        self.enc1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 112, 112)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 56, 56)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # -> (128, 28, 28)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # -> (256, 14, 14)

        # Exact Flatten Size for 224x224: 256 channels * 14 * 14 = 50,176
        self.flatten_size = 256 * 14 * 14

        # Latent Vectors
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        # No output_padding needed because 224 divides perfectly!
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> (128, 28, 28)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 56, 56)
        self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 112, 112)
        self.dec4 = nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1)  # -> (3, 224, 224)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 14, 14)  # Unflatten to (Batch, 256, 14, 14)

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = torch.sigmoid(self.dec4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # 1. Reconstruction Loss (MSE)
    # Sum of squared differences between pixels
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def train(train_loader):
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training VAE on {DEVICE}...")

    model.train()

    for epoch in range(EPOCHS):
        # Wrap the loader with tqdm for the progress bar
        loop = tqdm(train_loader, leave=True)

        total_loss = 0

        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Calculate loss
            loss = loss_function(recon_batch, data, mu, logvar)

            # Backward pass
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            # Update the progress bar text
            loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

    print("Training Complete!")
    return model




