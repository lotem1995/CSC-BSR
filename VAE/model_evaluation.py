import sys
import os
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

# ==========================================
# 0. PROJECT SETUP & STYLE LOADING
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VAE.VAE_nn import VAE
from preprocessing.load_dataset import ChessTilesCSV


def load_custom_style(style_path):
    """
    Manually parses a .mplstyle file to handle comments and quotes that
    might confuse the standard matplotlib.style.use() loader.
    """
    style_dict = {}

    with open(style_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Split key: value
            if ':' not in line:
                continue

            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()

            # Remove inline comments (anything after #)
            if '#' in val:
                val = val.split('#', 1)[0].strip()

            # Remove quotes if present
            val = val.strip('"').strip("'")

            # Special handling for cycler
            if key == 'axes.prop_cycle':
                # If it's the complex cycler string, we let matplotlib parse it raw later
                # OR we just skip it if it's causing the specific error seen earlier.
                # However, usually cleaning the quotes fixes the "unterminated string" error.
                pass

            style_dict[key] = val

    return style_dict


def apply_project_style(style_name="boardstate-dark.mplstyle"):
    """
    Locates the project's .mplstyle file, cleans it, and applies it.
    """
    style_path = PROJECT_ROOT / "utils" / "styles" / style_name

    if style_path.exists():
        print(f"Applying custom style: {style_name}")
        try:
            # First try standard loading
            plt.style.use(str(style_path))
        except Exception as e:
            print(f"Standard loading failed ({e}). Trying manual parsing...")
            try:
                # Manual parsing fallback
                clean_style = load_custom_style(style_path)
                plt.rcParams.update(clean_style)
                print("Manual style parsing applied successfully.")
            except Exception as e2:
                print(f"Manual parsing also failed: {e2}")
                print("Falling back to Seaborn dark theme.")
                sns.set_theme(style="darkgrid", context="notebook")
    else:
        print(f"Warning: Style '{style_name}' not found at {style_path}")
        print("Falling back to Seaborn dark theme.")
        sns.set_theme(style="darkgrid", context="notebook")


# ==========================================
# 1. LATENT VECTOR EXTRACTION & t-SNE
# ==========================================
def extract_latent_vectors(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)
    all_latents = []
    all_labels = []

    print("Extracting latent vectors...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting for t-SNE"):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            mu, _ = model.encode(imgs)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_latents, axis=0), np.concatenate(all_labels, axis=0)


def plot_tsne(features, labels, n_components=2, perplexity=30):
    print(f"Computing t-SNE on {features.shape[0]} vectors...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(features)

    print("Plotting t-SNE results...")
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=labels,
        palette="tab10",
        legend="full",
        alpha=0.8,
        s=50,
        edgecolor=None
    )
    plt.title("t-SNE Visualization of VAE Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    sns.despine()
    plt.show()


# ==========================================
# 2. RECONSTRUCTION ERROR ANOMALIES
# ==========================================
def show_top_anomalies(model, dataloader, top_k=10, device="cuda"):
    model.eval()
    model.to(device)
    results = []

    print("Scanning dataset for Reconstruction Error anomalies...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Recon Loss"):
            imgs = batch["image"].to(device)
            recon, _, _ = model(imgs)
            # MSE Loss per image
            loss = torch.mean((imgs.view(len(imgs), -1) - recon.view(len(imgs), -1)) ** 2, dim=1)

            imgs_cpu = imgs.cpu()
            loss_cpu = loss.cpu().numpy()

            for i in range(len(imgs)):
                results.append((loss_cpu[i], imgs_cpu[i]))

    results.sort(key=lambda x: x[0], reverse=True)

    print(f"Plotting top {top_k} Reconstruction Anomalies...")
    cols = min(5, top_k)
    rows = int(np.ceil(top_k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    if top_k == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i in range(top_k):
        loss_val, img_tensor = results[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        axes[i].imshow(img_np)
        axes[i].set_title(f"MSE: {loss_val:.4f}", fontsize=9)
        axes[i].axis("off")

    for j in range(top_k, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Highest Reconstruction Error Images (Standard MSE)")
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. LATENT OUTLIERS (ELLIPTIC ENVELOPE)
# ==========================================
def visualize_latent_outliers_2(model, dataset, contamination=0.05, top_k=10, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    model.to(device)

    all_mus = []
    print("Extracting features for Elliptic Envelope...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Latents"):
            imgs = batch["image"].to(device)
            mu, _ = model.encode(imgs)
            all_mus.append(mu.cpu().numpy())

    features = np.concatenate(all_mus, axis=0)

    print(f"Fitting Elliptic Envelope (contamination={contamination})...")
    clf = EllipticEnvelope(contamination=contamination, random_state=42)
    clf.fit(features)
    scores = clf.decision_function(features)

    anomaly_indices = np.argsort(scores)[:top_k]

    print(f"Plotting top {top_k} Latent Outliers...")
    cols = min(5, top_k)
    rows = int(np.ceil(top_k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(anomaly_indices):
        img_tensor = dataset[idx]["image"]
        score = scores[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        axes[i].imshow(img_np)
        axes[i].set_title(f"Score: {score:.2f}", fontsize=9)
        axes[i].axis("off")

    for j in range(top_k, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Top {top_k} Outliers via Elliptic Envelope")
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. CYCLE CONSISTENCY & DRIFT
# ==========================================
def get_consistency_scores(model, dataset, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    model.to(device)
    consistency_scores = []

    print("Calculating Single-Cycle Consistency Scores...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Cycle Consistency"):
            imgs = batch["image"].to(device)
            mu_1, _ = model.encode(imgs)
            recon_imgs = model.decode(mu_1)
            mu_2, _ = model.encode(recon_imgs)
            dist = torch.norm(mu_1 - mu_2, p=2, dim=1)
            consistency_scores.append(dist.cpu().numpy())

    return np.concatenate(consistency_scores)


def get_multicycle_scores(model, dataset, max_cycles=6, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    model.to(device)

    drift_results = {i: [] for i in range(1, max_cycles + 1)}

    print(f"Calculating Multi-Cycle Drift (up to {max_cycles} cycles)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Multi-Cycle Drift"):
            imgs = batch["image"].to(device)
            mu_current, _ = model.encode(imgs)
            batch_cumulative_drift = torch.zeros(imgs.size(0)).to(device)

            for cycle_i in range(1, max_cycles + 1):
                recon = model.decode(mu_current)
                mu_next, _ = model.encode(recon)
                step_dist = torch.norm(mu_current - mu_next, p=2, dim=1)

                batch_cumulative_drift += step_dist
                drift_results[cycle_i].append(batch_cumulative_drift.cpu().numpy().copy())
                mu_current = mu_next

    return {k: np.concatenate(v) for k, v in drift_results.items()}


def plot_worst_consistency(dataset, scores, title_text, top_k=30):
    anomaly_indices = np.argsort(scores)[::-1][:top_k]

    print(f"Plotting top {top_k} images for: {title_text}")
    cols = int(np.ceil(np.sqrt(top_k)))
    rows = int(np.ceil(top_k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    if top_k == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(anomaly_indices):
        img_tensor = dataset[idx]["image"]
        score = scores[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        axes[i].imshow(img_np)
        axes[i].set_title(f"{score:.2f}", fontsize=8)
        axes[i].axis("off")

    for j in range(top_k, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title_text)
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. LIKELIHOOD REGRET
# ==========================================
def get_likelihood_regret_scores(model, dataset, steps=20, lr=1e-2, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    model.to(device)
    regret_scores = []

    print(f"Calculating Likelihood Regret ({steps} opt steps)...")
    for batch in tqdm(dataloader, desc="Likelihood Regret"):
        imgs = batch["image"].to(device)
        batch_size = imgs.size(0)

        with torch.no_grad():
            mu, _ = model.encode(imgs)
            recon_init = model.decode(mu)
            loss_init = torch.sum((imgs.view(batch_size, -1) - recon_init.view(batch_size, -1)) ** 2, dim=1)

        z_optimized = mu.clone().detach().requires_grad_(True)
        optimizer_z = optim.Adam([z_optimized], lr=lr)

        for _ in range(steps):
            recon = model.decode(z_optimized)
            loss = F.mse_loss(recon, imgs, reduction='sum')
            optimizer_z.zero_grad()
            loss.backward()
            optimizer_z.step()

        with torch.no_grad():
            recon_final = model.decode(z_optimized)
            loss_final = torch.sum((imgs.view(batch_size, -1) - recon_final.view(batch_size, -1)) ** 2, dim=1)

        improvement = (loss_init - loss_final).cpu().numpy()
        regret_scores.append(np.maximum(improvement, 0))

    return np.concatenate(regret_scores)


def plot_regret_anomalies(dataset, scores, top_k=10):
    anomaly_indices = np.argsort(scores)[::-1][:top_k]
    print(f"Plotting top {top_k} Likelihood Regret anomalies...")

    cols = 5
    rows = int(np.ceil(top_k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(anomaly_indices):
        img_tensor = dataset[idx]["image"]
        score = scores[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        axes[i].imshow(img_np)
        axes[i].set_title(f"Rank {i + 1}\nRegret: {score:.0f}", fontsize=9)
        axes[i].axis("off")

    for j in range(top_k, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Likelihood Regret Anomalies (High Regret = OOD)")
    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- APPLY DARK THEME ---
    apply_project_style("boardstate-dark.mplstyle")

    # 1. SETUP
    print("--- SETUP ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = VAE()
    weights_path = 'VAE/models_weights/checkpoints/best_vae_model.pth'
    if not os.path.exists(weights_path):
        weights_path = PROJECT_ROOT / 'VAE/models_weights/checkpoints/best_vae_model.pth'

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    print("Model loaded.")

    # Load Dataset
    splits_dir = PROJECT_ROOT / "data/splits"

    # [FIXED] Point directly to PROJECT_ROOT to avoid duplication in load_dataset.py
    # This prevents the "data/data/..." path duplication error
    path_root = PROJECT_ROOT

    test_dataset = ChessTilesCSV(splits_dir / "test.csv", root=path_root)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Dataset loaded. {len(test_dataset)} samples.")
    print("-" * 30)

    # ----------------------------------------
    # RUN EVAL 2: STANDARD RECONSTRUCTION ERROR
    # ----------------------------------------
    print("\n--- RUNNING MSE RECONSTRUCTION ERROR ---")
    show_top_anomalies(model, test_loader, top_k=30, device=device)

    # ----------------------------------------
    # RUN EVAL 4: LATENT CYCLE CONSISTENCY (1 CYCLE)
    # ----------------------------------------
    print("\n--- RUNNING CYCLE CONSISTENCY (1 Cycle) ---")
    scores_cycle = get_consistency_scores(model, test_dataset, device=device)

    plt.figure(figsize=(10, 6))
    sns.histplot(scores_cycle, kde=True, bins=50)
    plt.title("Distribution of Single-Cycle Consistency Scores")
    plt.xlabel("Consistency Error (L2 Norm)")
    plt.ylabel("Count")
    sns.despine()
    plt.show()

    plot_worst_consistency(test_dataset, scores_cycle, "1-Cycle Consistency Anomalies", top_k=30)

    # ----------------------------------------
    # RUN EVAL 5: MULTI-CYCLE DRIFT
    # ----------------------------------------
    print("\n--- RUNNING MULTI-CYCLE DRIFTS  ---")
    max_cycles = 5
    all_drift_scores = get_multicycle_scores(model, test_dataset, max_cycles=max_cycles, device=device)

    for cycle_num in range(1, max_cycles + 1):
        scores = all_drift_scores[cycle_num]

        plt.figure(figsize=(10, 6))
        sns.histplot(scores, kde=True, bins=50)
        plt.title(f"Distribution of {cycle_num}-Cycle Drift Scores")
        plt.xlabel("Cumulative Drift Error")
        plt.ylabel("Frequency")
        sns.despine()
        plt.show()

        plot_worst_consistency(
            test_dataset,
            scores,
            f"{cycle_num}-Cycle Drift Anomalies",
            top_k=30
        )

    # ----------------------------------------
    # RUN EVAL 6: LIKELIHOOD REGRET
    # ----------------------------------------
    print("\n--- RUNNING LIKELIHOOD REGRET ---")
    scores_regret = get_likelihood_regret_scores(model, test_dataset, steps=30, lr=0.05, device=device)

    plt.figure(figsize=(10, 6))
    sns.histplot(scores_regret, kde=True, bins=50)
    plt.title("Distribution of Likelihood Regret Scores")
    plt.xlabel("Improvement Score (Regret)")
    plt.ylabel("Count")
    sns.despine()
    plt.show()

    plot_regret_anomalies(test_dataset, scores_regret, top_k=30)

    print("\n--- ALL EVALUATIONS COMPLETE ---")