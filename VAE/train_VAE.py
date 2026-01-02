import os


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from VAE.VAE_nn import VAE, loss_function
from pathlib import Path


import torch

from preprocessing.load_dataset import ChessTilesCSV, get_train_dataloader

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 1
LATENT_DIM = 20  # Size of the compressed "bottleneck"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_loader):
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training VAE on {DEVICE}...")

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            data = batch["image"].to(DEVICE)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Calculate loss
            loss = loss_function(recon_batch, data, mu, logvar)

            # Backward pass
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

    print("Training Complete!")
    return model

if __name__ == "__main__":
    train_loader = get_train_dataloader(batch_size=BATCH_SIZE,num_workers=4)
    trained_model = train(train_loader)

    # Save only the model parameters (weights/biases)
    model_filename = f"model_bs{BATCH_SIZE}_lr{LEARNING_RATE}_ep{EPOCHS}_lat{LATENT_DIM}.pth"
    torch.save(trained_model.state_dict(), model_filename)


