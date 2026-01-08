from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- IMPORTS FROM YOUR PROJECT ---
# Ensure these paths are correct in your directory structure
from VAE.VAE_nn import VAE
from VAE.calculate_treshold import find_optimal_amount_of_cycles, analyze_ood_threshold
from VAE.model_evaluation import get_multicycle_scores
from preprocessing.load_dataset import ChessTilesCSV, path_root, splits_dir


def predict_single_image_ood(image_tensor, model, threshold, cycles, device="cuda"):
    """
    Returns (is_ood: bool, score: float)
    """
    model.eval()

    # 1. Prepare Image: Add batch dimension (C,H,W) -> (1,C,H,W) if needed
    if image_tensor.dim() == 3:
        img = image_tensor.unsqueeze(0).to(device)
    else:
        img = image_tensor.to(device)

    # 2. Calculate Drift (Same logic as get_multicycle_scores, but for one item)
    cumulative_drift = 0.0

    with torch.no_grad():
        # Initial encoding
        mu_current, _ = model.encode(img)

        for _ in range(cycles):
            # Decode -> Re-encode
            recon = model.decode(mu_current)
            mu_next, _ = model.encode(recon)

            # Calculate distance (L2 norm)
            dist = torch.norm(mu_current - mu_next, p=2, dim=1)
            cumulative_drift += dist.item()

            # Update for next cycle
            mu_current = mu_next

    # 3. Decision
    is_ood = cumulative_drift > threshold
    return is_ood, cumulative_drift


if __name__ == "__main__":
    # 1. SETUP
    print("--- SETUP ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = VAE()
    weights_path = 'VAE/models_weights/cluster_trained_data_leakage/model_bs64_lr0.001_ep150_lat20_total_lossf13112878.92.pth'  # Update if needed
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    print("Model loaded.")

    # print("calculating threshold and the amount of cycles...")
    # ood_scores = get_multicycle_scores(model, ood_dataset, max_cycles=10, device=device)
    # id_scores = get_multicycle_scores(model, id_dataset, max_cycles=10, device=device)
    #
    # amount_of_cycles = find_optimal_amount_of_cycles(model, id_scores, ood_scores, max_cycles=10, device=device)
    # threshold = analyze_ood_threshold(id_scores, ood_scores, plot=True)
    amount_of_cycles=1
    threshold=3.3

    test_dataset = ChessTilesCSV(splits_dir / "test.csv", root=path_root)

    print("Scanning dataset using Single-Image Detection (Stopping after 100 OODs)...")

    found_ood_images = []  # To store the image tensors
    found_ood_scores = []  # To store the scores

    # Iterate over the dataset one by one
    for idx in tqdm(range(len(test_dataset)), desc="Scanning"):
        # Stop if we already found 100 to save time
        if len(found_ood_images) >= 100:
            break

        img_tensor = test_dataset[idx]['image']

        # --- USE THE SINGLE FUNCTION HERE ---
        is_ood, score = predict_single_image_ood(
            img_tensor,
            model,
            threshold=threshold,
            cycles=amount_of_cycles,
            device=device
        )

        if is_ood:
            found_ood_images.append(img_tensor)
            found_ood_scores.append(score)

    print(f"Scan complete. Found {len(found_ood_images)} OOD samples.")

    # --- PLOTTING LOGIC ---
    if len(found_ood_images) > 0:
        num_to_plot = len(found_ood_images)

        print("Plotting detected OOD samples...")

        # Create the plot (10x10 grid)
        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        fig.suptitle(f"Top {num_to_plot} Detected OOD Samples (Drift > {threshold:.2f})")
        axes = axes.flatten()

        for i in range(num_to_plot):
            img_tensor = found_ood_images[i]
            score = found_ood_scores[i]

            # Convert Tensor to Numpy for plotting
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)

            axes[i].imshow(img_np)
            axes[i].axis('off')
            # Optional: Show score
            # axes[i].set_title(f"{score:.2f}", fontsize=8)

        # Turn off unused subplots
        for j in range(num_to_plot, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    else:
        print(f"No samples exceeded the threshold of {threshold:.2f}")





