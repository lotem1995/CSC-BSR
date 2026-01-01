import os

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from VAE.VAE_nn import VAE
from VAE.dataset import FilenameDataset
# Your existing import
from VAE.train_VAE import BATCH_SIZE


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


def calibrate_threshold(model, val_loader, device):
    """
    Runs the model on normal validation data to find the 'safe' error range.
    """
    model.eval()
    scores = []

    print("Calibrating threshold on normal data...")
    with torch.no_grad():
        for data, _ in tqdm(val_loader):
            data = data.to(device)
            # Reconstruct
            recon, _, _ = model(data)

            # Calculate MSE per image in the batch
            # (Batch, C, H, W) -> (Batch, -1) to sum errors per image
            loss = F.mse_loss(recon.view(data.size(0), -1),
                              data.view(data.size(0), -1),
                              reduction='none')  # 'none' gives vector of losses

            # Sum pixels for each image to get total error per image
            image_scores = torch.sum(loss, dim=1).cpu().numpy()
            scores.extend(image_scores)

    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Set threshold at 3 standard deviations above the mean
    threshold = mean_score + (3 * std_score)

    print(f"Normal Data - Mean Error: {mean_score:.2f}, Std: {std_score:.2f}")
    print(f"Calculated Threshold: {threshold:.2f}")
    return threshold


def detect_ood(model, image, threshold):
    """
    Returns True if the image is OOD, False if Normal.
    Input `image` should be a tensor on the correct device.
    """
    # Use your existing scoring function
    score, _ = get_anomaly_score(model, image)

    is_ood = score > threshold

    print(f"Image Score: {score:.2f} | Threshold: {threshold:.2f}")
    if is_ood:
        print("🚨 ALERT: Out of Distribution Data Detected!")
    else:
        print("✅ Status: Normal Data")

    return is_ood


def analyze_image_from_path(model, image_path, threshold, device):
    """
    Helper: Loads an image file, transforms it, and runs OOD detection.
    """
    # 1. Load Image
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        return

    # 2. Transform (Must match training size exactly!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure this matches your VAE input
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).to(device)

    # 3. Handle Batch Dimension (Add 1 at the start if missing)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 4. Run Detection
    print(f"\n--- Analyzing: {image_path} ---")
    detect_ood(model, img_tensor, threshold)


# --- Usage Example ---
if __name__ == "__main__":
    # 1. Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ... Assume 'model' and 'train_loader' are loaded here ...
    model = VAE()
    model.load_state_dict(torch.load('model_weights.pth'))
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(parent_dir, 'preprocessed_data')
    train_dataset = FilenameDataset(img_dir=data_dir)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Calibrate (Calculate threshold ONCE)
    threshold = calibrate_threshold(model, train_loader, DEVICE)

    anomaly_path = "C:\\Users\\gilad\\PycharmProjects\\CSC-BSR\\preprocessed_data\\game2_frame_000856_tile_row7_column3_class5.png"
    # 3. Test on a real file
    analyze_image_from_path(model,anomaly_path, threshold, DEVICE)

    anomaly_image = transforms.ToTensor()(Image.open(anomaly_path).convert('RGB'))
    score_anomaly, recon_anomaly = get_anomaly_score(model, anomaly_image)
    show_comparison(anomaly_image, recon_anomaly, "Anomaly Reconstruction")
    print(f"Anomaly Score: {score_anomaly:.4f}")

    anomaly_image = torch.rand(3, 224, 224).to(DEVICE)
    score_anomaly, recon_anomaly = get_anomaly_score(model, anomaly_image)
    show_comparison(anomaly_image, recon_anomaly, "Anomaly Reconstruction")
    print(f"Anomaly Score (Random Noise): {score_anomaly:.4f}")
