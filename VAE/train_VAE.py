import os


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from VAE.VAE_nn import VAE, loss_function
from VAE.dataset import FilenameDataset


import torch


# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
LATENT_DIM = 20  # Size of the compressed "bottleneck"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(parent_dir, 'preprocessed_data')

    train_dataset = FilenameDataset(img_dir=data_dir)

    # Load Data
    transform = transforms.ToTensor()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    trained_model = train(train_loader)

    # Save only the model parameters (weights/biases)
    model_filename = f"model_bs{BATCH_SIZE}_lr{LEARNING_RATE}_ep{EPOCHS}_lat{LATENT_DIM}.pth"
    torch.save(trained_model.state_dict(), model_filename)


