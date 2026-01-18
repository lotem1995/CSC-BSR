import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os
from pathlib import Path

# --- CONFIGURATION: PATHS ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

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


# --- STYLE CONFIGURATION ---
# Derived from matplotlib_sniplets.py and boardstate-dark.mplstyle
def apply_boardstate_style(style="dark"):
    """Attempts to load the custom mplstyle and defines helper functions."""

    # 1. Load the .mplstyle file
    style_filename = f"boardstate-{style}.mplstyle"
    style_path = CURRENT_DIR / style_filename

    if style_path.exists():
        plt.style.use(str(style_path))
        print(f"✓ Style loaded: {style_filename}")
    else:
        print(f"⚠️ Style file '{style_filename}' not found. Using default style.")
        # Fallback settings if file is missing to match the aesthetic roughly
        if style == "dark":
            plt.rcParams.update({
                "figure.facecolor": "#0F1115",
                "axes.facecolor": "#151A22",
                "text.color": "#E8EDF7",
                "axes.labelcolor": "#E8EDF7",
                "grid.color": "#2A3446",
            })

    # 2. Set Figure constraints (from snippets)
    plt.rcParams["figure.constrained_layout.use"] = True


# Helper function from matplotlib_sniplets.py
def boardstate_axes(ax):
    """Removes top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# Color palette from snippets (available for annotations)
CALLOUT = {
    "decision": "#7C5CFF",
    "result": "#2ECC71",
    "warning": "#F2C94C",
    "danger": "#EB5757",
}


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
    # 1. Setup Style
    # Change to "light" if you prefer the light theme
    apply_boardstate_style("dark")

    # 2. Find File
    classifier_path = find_classifier_file()
    if not classifier_path:
        return

    # 3. Output Path
    output_image = PROJECT_ROOT / "results" / "tsne_visualization.png"
    if not output_image.parent.exists():
        os.makedirs(output_image.parent)

    print(f"--- t-SNE Visualization ---")

    # 4. Initialize Classifier & Load Data
    mock_backbone = MockEmbedding(dim=384)
    classifier = FENClassifier(embedding_extractor=mock_backbone)

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

    # 5. Process Data
    X = torch.stack(embeddings_list).cpu().numpy()
    y = np.array(labels_list)

    MAX_SAMPLES = 3000
    if len(X) > MAX_SAMPLES:
        print(f"⚠️ Sampling {MAX_SAMPLES} random points (Dataset size: {len(X)})...")
        indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
        X = X[indices]
        y = y[indices]

    # 6. Run t-SNE
    print("Running t-SNE (this may take 10-30 seconds)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # 7. Plotting with BoardState Style
    print("Generating plot...")

    # Using 16:9 ratio but scaled up from snippets (7.2, 4.0) -> (14.4, 8.0)
    fig, ax = plt.subplots(figsize=(14.4, 8.0))

    human_labels = [LABEL_MAP.get(label, f"Class {label}") for label in y]

    # Note: We use "tab20" because we have ~14 classes.
    # The boardstate cycler only has 6 colors.
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=human_labels,
        palette="tab20",
        s=60,
        alpha=0.8,
        edgecolor=plt.rcParams.get("axes.facecolor", "#151A22"),  # Match bg
        linewidth=0.5,
        ax=ax
    )

    # Apply specific axes styling
    ax = boardstate_axes(ax)

    # Titles and Labels
    ax.set_title(f"t-SNE of Chess Tile Embeddings (N={len(X)})", pad=20)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Adjust Legend to fit style
    # Move legend outside to keep plot clean
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Classes")

    # Save
    plt.savefig(output_image)  # dpi is handled by mplstyle (default 200)
    print(f"✓ Plot saved to: {output_image}")
    plt.show()


if __name__ == "__main__":
    plot_knn_space()