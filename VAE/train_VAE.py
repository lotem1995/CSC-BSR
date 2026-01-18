import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Assuming these are your custom modules
from VAE.VAE_nn import VAE, loss_function
from preprocessing.load_dataset import get_train_dataloader, get_val_dataloader

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 300
LATENT_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Validation Config ---
PATIENCE = 5
MIN_DELTA = 0.1

# --- Checkpoint Config ---
CHECKPOINT_INTERVAL = 5  # Save weights every 5 epochs
CHECKPOINT_DIR = "checkpoints"  # Folder to keep things organized


def train(train_loader, val_loader):
    # Create checkpoint directory if it doesn't exist
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    model = VAE()
    model.to(DEVICE)
    # Load weights if restarting, otherwise comment out
    # weights_path = 'model_bs64_...'
    # model.load_state_dict(torch.load(weights_path))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training VAE on {DEVICE}...")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        # 1. Training Phase
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for batch in pbar:
            data = batch["image"].to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader.dataset)

        # 2. Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch["image"].to(DEVICE)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 3. Save Interval Checkpoint (Every 5 Epochs)
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            filename = os.path.join(CHECKPOINT_DIR, f"vae_epoch_{epoch + 1}_loss{avg_val_loss:.2f}.pth")
            torch.save(model.state_dict(), filename)

        # 4. Early Stopping & Best Model Logic
        if avg_val_loss < (best_val_loss - MIN_DELTA):
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save BEST model
            torch.save(model.state_dict(), "best_vae_model.pth")
            print("Validation loss improved. Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training Complete!")
    return model


if __name__ == "__main__":
    train_loader = get_train_dataloader(batch_size=BATCH_SIZE, num_workers=4)
    val_loader = get_val_dataloader(batch_size=BATCH_SIZE, num_workers=4)
    trained_model = train(train_loader, val_loader)