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

from VAE.VAE_nn import train, DEVICE, BATCH_SIZE
from VAE.dataset import FilenameDataset


import torch
import torch.nn as nn
import torch.nn.functional as F




def get_anomaly_score(model, image):
    model.eval()
    with torch.no_grad():
        # --- FIX 1: Add Batch Dimension ---
        # Current shape: (3, 224, 224) -> Model thinks batch size is 3!
        # New shape: (1, 3, 224, 224) -> Model knows batch size is 1.
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

            # Forward pass
        reconstructed_image, mu, logvar = model(image)

        # --- FIX 2: Match dimensions correctly ---
        # Don't hardcode 784 (28x28) if your model is built for 224x224
        # We flatten both images completely to compare them
        loss = F.mse_loss(
            reconstructed_image.view(image.size(0), -1),
            image.view(image.size(0), -1),
            reduction='sum'
        )

        return loss.item(), reconstructed_image


def show_comparison(original, reconstructed, title):
    # 1. Handle Batch Dimension
    # If shape is (1, 3, 224, 224), squeeze to (3, 224, 224)
    if original.dim() == 4:
        original = original.squeeze(0)
    if reconstructed.dim() == 4:
        reconstructed = reconstructed.squeeze(0)

    # 2. Convert to Numpy and Fix Color Channels
    # PyTorch uses (Channels, Height, Width) -> (3, 224, 224)
    # Matplotlib needs (Height, Width, Channels) -> (224, 224, 3)
    original_np = original.cpu().detach().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.cpu().detach().permute(1, 2, 0).numpy()

    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # We remove cmap='gray' because these are likely color images
    axes[0].imshow(original_np)
    axes[0].set_title("Input")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_np)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    plt.suptitle(title)
    plt.show()

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

    # --- Run the Anomaly Test ---
    normal_image, _ = train_dataset[0]
    normal_image = normal_image.to(DEVICE)

    # 2. Create an "Anomaly" (Random Noise)
    anomaly_image = torch.rand(3, 224, 224).to(DEVICE)

    # 3. Calculate Scores
    score_normal, recon_normal = get_anomaly_score(trained_model, normal_image)
    score_anomaly, recon_anomaly = get_anomaly_score(trained_model, anomaly_image)

    # 4. Print Results
    print(f"Anomaly Score (Normal Digit): {score_normal:.4f}")
    print(f"Anomaly Score (Random Noise): {score_anomaly:.4f}")

    if score_anomaly > score_normal:
        print("SUCCESS: The anomaly has a higher error score.")

    # Run visualization if you are in a notebook/GUI environment
    show_comparison(normal_image, recon_normal, "Normal Image Reconstruction")
    show_comparison(anomaly_image, recon_anomaly, "Anomaly Reconstruction")
