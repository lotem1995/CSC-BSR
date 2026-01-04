import os
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# --- Your Imports ---
from VAE.VAE_nn import VAE
from preprocessing.load_dataset import ChessTilesCSV


def get_anomaly_score(model, image):
    """
    Calculates reconstruction error for a single image (or batch of 1).
    """
    model.eval()
    with torch.no_grad():
        # Handle Batch Dimension: Ensure shape is (1, C, H, W)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Forward pass
        reconstructed_image, mu, logvar = model(image)

        # Calculate MSE loss (sum of squared errors)
        loss = F.mse_loss(
            reconstructed_image.view(image.size(0), -1),
            image.view(image.size(0), -1),
            reduction='sum'
        )
        return loss.item(), reconstructed_image


def show_comparison(original, reconstructed, title):
    """Plots the Original vs Reconstructed image side-by-side."""
    if original.dim() == 4:
        original = original.squeeze(0)
    if reconstructed.dim() == 4:
        reconstructed = reconstructed.squeeze(0)

    original_np = original.cpu().detach().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.cpu().detach().permute(1, 2, 0).numpy()

    original_np = np.clip(original_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Input")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_np)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def calibrate_threshold(model, dataloader, device, num_stds=3):
    """
    Runs the model on NORMAL data to find the baseline error statistics.
    """
    print(f"\n--- Calibrating Threshold on Normal Data ---")
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calibrating"):
            images = batch["image"].to(device)

            # Forward pass
            recon, _, _ = model(images)

            # Calculate MSE per image in the batch
            # Flatten to (Batch_Size, Pixels) to sum errors per image
            loss = F.mse_loss(
                recon.view(images.size(0), -1),
                images.view(images.size(0), -1),
                reduction='none'  # Returns a vector of losses, one per image
            )

            # Sum pixels for each image to get total error per image
            batch_scores = torch.sum(loss, dim=1).cpu().numpy()
            scores.extend(batch_scores)

    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Set threshold at Mean + (N * Std)
    threshold = mean_score + (num_stds * std_score)

    print(f"Normal Data stats | Mean: {mean_score:.2f} | Std: {std_score:.2f}")
    print(f"Calculated Threshold ({num_stds} stds): {threshold:.2f}")

    return threshold


def scan_dataloader_and_plot_ood(model, dataloader, threshold, device):
    """
    Iterates through a DataLoader, checks every image, and PLOTS it if it is OOD.
    """
    print(f"\n--- Scanning Test Data for Anomalies (Threshold: {threshold:.2f}) ---")

    ood_count = 0
    total_checked = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Scanning Batches")):
        images = batch["image"].to(device)

        for i in range(images.size(0)):
            img_tensor = images[i]
            total_checked += 1

            score, recon_tensor = get_anomaly_score(model, img_tensor)

            if score > threshold:
                ood_count += 1
                info_str = f"Batch {batch_idx} | Img {i}"
                print(f"🚨 OOD DETECTED: {info_str} | Score: {score:.2f}")

                show_comparison(
                    img_tensor,
                    recon_tensor,
                    title=f"OOD DETECTED\n{info_str}\nScore: {score:.0f} > {threshold:.0f}"
                )

    print(f"\nScan complete. Found {ood_count} OOD images out of {total_checked}.")


# --- Usage Example ---
if __name__ == "__main__":
    # 1. Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {DEVICE}")

    # 2. Load Model
    model = VAE().to(DEVICE)
    model_path = r'VAE\models_weights\model_bs64_lr0.001_ep10_lat20.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Warning: Model path not found at {model_path}")

    # 3. Data Config
    splits_dir = Path("data/splits")
    path_root = Path("data")

    # --- THRESHOLD CALCULATION OPTION ---
    CALIBRATE = False  # Set to False to use the hardcoded value

    threshold = 0.0

    if CALIBRATE:
        # Load NORMAL data (Train or Validation split) to calculate threshold
        # Assuming you have 'train.csv' or 'val.csv' in the splits folder
        calib_dataset = ChessTilesCSV(splits_dir / "train.csv", root=path_root)
        calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=False)

        # Calculate!
        threshold = calibrate_threshold(model, calib_loader, DEVICE, num_stds=3)
    else:
        # Hardcoded backup
        threshold = 416.28 + (3 * 173.22)
        print(f"Using Hardcoded Threshold: {threshold:.2f}")

    # 4. Run Scan on Test Data
    if (splits_dir / "test.csv").exists():
        test_dataset = ChessTilesCSV(splits_dir / "test.csv", root=path_root)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        scan_dataloader_and_plot_ood(model, test_loader, threshold, DEVICE)
    else:
        print("Test data CSV not found.")