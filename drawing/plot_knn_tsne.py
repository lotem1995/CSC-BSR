import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os
from pathlib import Path

# --- CONFIGURATION: PATHS ---
CURRENT_DIR = Path(__file__).resolve().parent  # .../CSC-BSR/drawing
PROJECT_ROOT = CURRENT_DIR.parent  # .../CSC-BSR

# Add Project Root to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTS ---
try:
    from embedding.classifier import FENClassifier
except ImportError:
    try:
        sys.path.append(str(PROJECT_ROOT / "embedding"))
        from classifier import FENClassifier
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        sys.exit(1)


# --- MOCK EMBEDDING ---
class MockEmbedding:
    def __init__(self, dim=384):
        self.dim = dim
        self.device = torch.device('cpu')

    def get_embedding_dim(self):
        return self.dim


# --- LABEL MAPPING ---
LABEL_MAP = {
    0: "Empty", 1: "W Pawn", 2: "W Knight", 3: "W Bishop", 4: "W Rook", 5: "W Queen", 6: "W King",
    11: "B Pawn", 12: "B Knight", 13: "B Bishop", 14: "B Rook", 15: "B Queen", 16: "B King",
    17: "OOD"
}


def find_classifier_file(filename="classifier_dino_small.pt"):
    """Searches for the classifier file in likely locations."""
    candidate_paths = [
        PROJECT_ROOT / "final" / filename,
        PROJECT_ROOT / filename,
        CURRENT_DIR / filename,
        PROJECT_ROOT / "embedding" / filename,
        PROJECT_ROOT / "data" / filename,
    ]

    print(f"Searching for '{filename}'...")
    for path in candidate_paths:
        if path.exists():
            print(f"✓ Found: {path}")
            return path

    print("\n❌ Error: Could not find the file in any common folder.")
    return None


def plot_knn_space():
    # 1. Find the file
    classifier_path = find_classifier_file()
    if not classifier_path:
        return

    # 2. Initialize Output Path
    output_image = PROJECT_ROOT / "results" / "tsne_visualization.png"
    if not output_image.parent.exists():
        os.makedirs(output_image.parent)

    print(f"--- t-SNE Visualization ---")

    # 3. Initialize Classifier
    mock_backbone = MockEmbedding(dim=384)
    classifier = FENClassifier(embedding_extractor=mock_backbone)

    # 4. Load Data
    print("Loading classifier data...")
    try:
        state = torch.load(str(classifier_path), map_location='cpu')
        if isinstance(state, dict) and 'global_embeddings' in state:
            classifier.global_embeddings = state['global_embeddings']
            classifier.global_labels = state['global_labels']
        elif hasattr(state, 'global_embeddings'):
            classifier.global_embeddings = state.global_embeddings
            classifier.global_labels = state.global_labels
        else:
            print("❌ Error: Checkpoint format not recognized.")
            return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    embeddings_list = classifier.global_embeddings
    labels_list = classifier.global_labels

    if not embeddings_list:
        print("❌ Error: Classifier database is empty.")
        return

    print(f"✓ Loaded {len(embeddings_list)} data points.")

    # 5. Convert to Numpy
    X = torch.stack(embeddings_list).cpu().numpy()
    y = np.array(labels_list)

    # 6. Sampling
    MAX_SAMPLES = 3000
    if len(X) > MAX_SAMPLES:
        print(f"⚠️ Sampling {MAX_SAMPLES} random points (Dataset size: {len(X)})...")
        indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X = X[indices]
        y = y[indices]

    # 7. Run t-SNE (Safe Version)
    print("Running t-SNE (this may take 10-30 seconds)...")

    # FIX: Removed 'n_iter' and 'init' specific arguments to ensure compatibility
    # with different scikit-learn versions. Defaults are usually sufficient.
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        learning_rate='auto'
    )

    X_embedded = tsne.fit_transform(X)

    # 8. Plotting
    print("Generating plot...")
    plt.figure(figsize=(16, 10))
    human_labels = [LABEL_MAP.get(label, f"Class {label}") for label in y]

    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=human_labels, palette="tab20", s=60, alpha=0.8, edgecolor="w"
    )

    plt.title(f"t-SNE of Chess Tile Embeddings (N={len(X)})", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"✓ Plot saved to: {output_image}")
    plt.show()


if __name__ == "__main__":
    plot_knn_space()