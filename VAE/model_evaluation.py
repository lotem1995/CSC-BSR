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
from preprocessing.load_dataset import ChessTilesCSV


# ==========================================
# 1. LATENT VECTOR EXTRACTION & t-SNE
# ==========================================
def extract_latent_vectors(model, dataloader, device="cuda"):
    """
    Passes data through the trained model to extract 'mu' (latent vectors)
    and corresponding labels.
    """
    model.eval()
    model.to(device)

    all_latents = []
    all_labels = []

    print("Extracting latent vectors...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting for t-SNE")):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            mu, _ = model.encode(imgs)

            all_latents.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


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
        alpha=0.7
    )
    plt.title("t-SNE Visualization of VAE Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
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
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calculating Recon Loss")):
            imgs = batch["image"].to(device)
            recon, _, _ = model(imgs)

            # MSE Loss per image
            loss = torch.mean((imgs.view(len(imgs), -1) - recon.view(len(imgs), -1)) ** 2, dim=1)

            imgs_cpu = imgs.cpu()
            loss_cpu = loss.cpu().numpy()

            for i in range(len(imgs)):
                results.append((loss_cpu[i], imgs_cpu[i]))

    # Sort Descending (Highest Error first)
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"Plotting top {top_k} Reconstruction Anomalies...")
    fig, axes = plt.subplots(1, top_k, figsize=(20, 4))

    # Handle case where top_k might be 1 (subplots behaves differently)
    if top_k == 1: axes = [axes]

    for i in range(top_k):
        loss_val, img_tensor = results[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        axes[i].imshow(img_np)
        axes[i].set_title(f"MSE Loss: {loss_val:.4f}")
        axes[i].axis("off")

    plt.suptitle("Highest Reconstruction Error Images (Standard MSE)")
    plt.show()


# ==========================================
# 3. LATENT OUTLIERS (ELLIPTIC ENVELOPE)
# ==========================================
def visualize_latent_outliers_2(model, dataset, contamination=0.05, top_k=10, device="cuda"):
    # Create internal loader to ensure shuffle=False
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    model.to(device)

    all_mus = []

    print("Extracting features for Elliptic Envelope...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting Latents")):
            imgs = batch["image"].to(device)
            mu, _ = model.encode(imgs)
            all_mus.append(mu.cpu().numpy())

    features = np.concatenate(all_mus, axis=0)

    print(f"Fitting Elliptic Envelope (contamination={contamination})...")
    clf = EllipticEnvelope(contamination=contamination, random_state=42)
    clf.fit(features)
    scores = clf.decision_function(features)  # lower = more anomalous

    # argsort sorts low to high; we want lowest scores
    anomaly_indices = np.argsort(scores)[:top_k]

    print(f"Plotting top {top_k} Latent Outliers...")
    fig, axes = plt.subplots(1, top_k, figsize=(20, 5))
    if top_k == 1: axes = [axes]

    for i, idx in enumerate(anomaly_indices):
        img_tensor = dataset[idx]["image"]
        score = scores[idx]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize for display

        axes[i].imshow(img_np)
        axes[i].set_title(f"Score: {score:.2f}\nIdx: {idx}")
        axes[i].axis("off")

    plt.suptitle(f"Top {top_k} Outliers via Elliptic Envelope (Latent Space Density)")
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
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Cycle Consistency")):
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

    # Dictionary to store cumulative scores for each cycle: {1: [], 2: [], ...}
    drift_results = {i: [] for i in range(1, max_cycles + 1)}

    print(f"Calculating Multi-Cycle Drift (up to {max_cycles} cycles) in a single pass...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Multi-Cycle Drift")):
            imgs = batch["image"].to(device)

            # Initial encoding
            mu_current, _ = model.encode(imgs)

            # Initialize cumulative drift for this batch
            batch_cumulative_drift = torch.zeros(imgs.size(0)).to(device)

            for cycle_i in range(1, max_cycles + 1):
                # 1. Decode -> Re-encode
                recon = model.decode(mu_current)
                mu_next, _ = model.encode(recon)

                # 2. Calculate distance for this specific step
                step_dist = torch.norm(mu_current - mu_next, p=2, dim=1)

                # 3. Add to total drift
                batch_cumulative_drift += step_dist

                # 4. Save the CURRENT cumulative total to the dictionary
                drift_results[cycle_i].append(batch_cumulative_drift.cpu().numpy())

                # 5. Update latent for next cycle
                mu_current = mu_next

    # Concatenate lists into numpy arrays for each cycle key
    final_scores = {k: np.concatenate(v) for k, v in drift_results.items()}

    return final_scores


def plot_worst_consistency(dataset, scores, title_text, top_k=100):
    # Sort indices by Highest Score (Worst Consistency) -> Descending order
    anomaly_indices = np.argsort(scores)[::-1][:top_k]

    print(f"Plotting top {top_k} images for: {title_text}")

    # dynamic grid calculation
    cols = int(np.ceil(np.sqrt(top_k)))
    rows = int(np.ceil(top_k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))

    # Flatten axes array for easy iteration
    if top_k == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, idx in enumerate(anomaly_indices):
        img_tensor = dataset[idx]["image"]
        score = scores[idx]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        axes[i].imshow(img_np)
        # Simplified title to save space in grid
        axes[i].set_title(f"{score:.2f}", fontsize=8)
        axes[i].axis("off")

    # Turn off unused axes if top_k doesn't perfectly fill the grid
    for i in range(top_k, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title_text)
    plt.tight_layout()
    plt.show()


# ==========================================
# 5. LIKELIHOOD REGRET
# ==========================================
def get_likelihood_regret_scores(model, dataset, steps=20, lr=1e-2, device="cuda"):
    # Small batch size for optimization
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    model.to(device)
    regret_scores = []

    print(f"Calculating Likelihood Regret ({steps} opt steps per batch)...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Likelihood Regret")):
        imgs = batch["image"].to(device)
        batch_size = imgs.size(0)

        # Phase 1: Initial Guess
        with torch.no_grad():
            mu, _ = model.encode(imgs)
            recon_init = model.decode(mu)
            loss_init = torch.sum((imgs.view(batch_size, -1) - recon_init.view(batch_size, -1)) ** 2, dim=1)

        # Phase 2: Optimization
        z_optimized = mu.clone().detach().requires_grad_(True)
        optimizer_z = optim.Adam([z_optimized], lr=lr)

        for _ in range(steps):
            recon = model.decode(z_optimized)
            loss = F.mse_loss(recon, imgs, reduction='sum')
            optimizer_z.zero_grad()
            loss.backward()
            optimizer_z.step()

        # Phase 3: Final Loss
        with torch.no_grad():
            recon_final = model.decode(z_optimized)
            loss_final = torch.sum((imgs.view(batch_size, -1) - recon_final.view(batch_size, -1)) ** 2, dim=1)

        # Regret = Improvement
        improvement = (loss_init - loss_final).cpu().numpy()
        improvement = np.maximum(improvement, 0)
        regret_scores.append(improvement)

    return np.concatenate(regret_scores)


def plot_regret_anomalies(dataset, scores, top_k=10):
    anomaly_indices = np.argsort(scores)[::-1][:top_k]

    print(f"Plotting top {top_k} Likelihood Regret anomalies...")

    # Calculate grid size
    cols = 5
    rows = int(np.ceil(top_k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(anomaly_indices):
        img_tensor = dataset[idx]["image"]
        score = scores[idx]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        axes[i].imshow(img_np)
        axes[i].set_title(f"Rank: {i + 1}\nRegret: {score:.0f}\nIdx: {idx}")
        axes[i].axis("off")

    for i in range(top_k, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Likelihood Regret Anomalies (High Regret = OOD)")
    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
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

    # Load Dataset
    splits_dir = Path("data/splits")
    path_root = Path("data")
    test_dataset = ChessTilesCSV(splits_dir / "test.csv", root=path_root)

    # Create a general loader for functions that need it
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Dataset loaded. {len(test_dataset)} samples.")
    print("-" * 30)

    # # ----------------------------------------
    # # RUN EVAL 1: t-SNE
    # # ----------------------------------------
    # print("\n--- RUNNING T-SNE ---")
    # features, labels = extract_latent_vectors(model, test_loader, device=device)
    # plot_tsne(features, labels)
    #
    # # ----------------------------------------
    # # RUN EVAL 2: STANDARD RECONSTRUCTION ERROR
    # # ----------------------------------------
    # print("\n--- RUNNING MSE RECONSTRUCTION ERROR ---")
    # show_top_anomalies(model, test_loader, top_k=30, device=device)
    #
    # # ----------------------------------------
    # # RUN EVAL 3: ELLIPTIC ENVELOPE (LATENT DENSITY)
    # # ----------------------------------------
    # print("\n--- RUNNING ELLIPTIC ENVELOPE OUTLIER DETECTION ---")
    # visualize_latent_outliers_2(model, test_dataset, contamination=0.1, top_k=30, device=device)
    #
    # ----------------------------------------
    # RUN EVAL 4: LATENT CYCLE CONSISTENCY (1 CYCLE)
    # ----------------------------------------
    print("\n--- RUNNING CYCLE CONSISTENCY (1 Cycle) ---")
    scores_cycle = get_consistency_scores(model, test_dataset, device=device)

    # Histogram
    plt.figure()
    plt.hist(scores_cycle, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Single-Cycle Consistency Scores")
    plt.xlabel("Error")
    plt.show()

    plot_worst_consistency(test_dataset, scores_cycle, "1-Cycle Consistency Anomalies", top_k=100)

    # ----------------------------------------
    # RUN EVAL 5: MULTI-CYCLE DRIFT
    # ----------------------------------------
    print("\n--- RUNNING MULTI-CYCLE DRIFTS  ---")

    # Run ONLY ONCE for the maximum number of cycles you care about
    max_cycles = 10
    all_drift_scores = get_multicycle_scores(model, test_dataset, max_cycles=max_cycles, device=device)

    # Loop through the results to plot
    for cycle_num in range(1, max_cycles + 1):
        scores = all_drift_scores[cycle_num]

        # -- Plot Histogram for this cycle ---
        plt.figure(figsize=(8, 4))
        plt.hist(scores, bins=50, color='orange', alpha=0.7, edgecolor='black')
        plt.title(f"Distribution of {cycle_num}-Cycle Drift Scores")
        plt.xlabel("Cumulative Drift Error")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.5)
        plt.show()
        # ------------------------------------------

        plot_worst_consistency(
            test_dataset,
            scores,
            f"{cycle_num}-Cycle Drift Anomalies",
            top_k=100
        )

    # ----------------------------------------
    # RUN EVAL 6: LIKELIHOOD REGRET (Most expensive)
    # ----------------------------------------
    print("\n--- RUNNING LIKELIHOOD REGRET ---")
    # Reduced steps/lr slightly for speed in testing, adjust as needed
    scores_regret = get_likelihood_regret_scores(model, test_dataset, steps=30, lr=0.05, device=device)

    # Histogram
    plt.figure()
    plt.hist(scores_regret, bins=50, color='green', alpha=0.7)
    plt.title("Distribution of Likelihood Regret Scores")
    plt.xlabel("Improvement Score")
    plt.show()

    plot_regret_anomalies(test_dataset, scores_regret, top_k=30)

    print("\n--- ALL EVALUATIONS COMPLETE ---")