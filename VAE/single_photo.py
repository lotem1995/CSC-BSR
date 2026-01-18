import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import seaborn as sns

# --- 1. SETUP PROJECT PATHS ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the VAE model definition
from VAE.VAE_nn import VAE


# --- 2. STYLE HELPERS ---
def load_custom_style(style_path):
    style_dict = {}
    with open(style_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if ':' not in line: continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            if '#' in val: val = val.split('#', 1)[0].strip()
            val = val.strip('"').strip("'")
            if key == 'axes.prop_cycle':
                pass
            else:
                style_dict[key] = val
    return style_dict


def apply_project_style(style_name="boardstate-dark.mplstyle"):
    style_path = PROJECT_ROOT / "utils" / "styles" / style_name
    if style_path.exists():
        print(f"Applying custom style: {style_name}")
        try:
            plt.style.use(str(style_path))
        except Exception:
            try:
                clean_style = load_custom_style(style_path)
                plt.rcParams.update(clean_style)
            except Exception as e:
                print(f"Style loading failed: {e}. Using Seaborn dark theme.")
                sns.set_theme(style="darkgrid", context="notebook")
    else:
        print(f"Warning: Style '{style_name}' not found. Using Seaborn dark theme.")
        sns.set_theme(style="darkgrid", context="notebook")


# --- 3. CONFIGURATION ---
MODEL_WEIGHTS_PATH = PROJECT_ROOT / "VAE" / "models_weights" / "checkpoints" / "best_vae_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vae_model(weights_path, device):
    print(f"Loading model from {weights_path}...")
    if not weights_path.exists():
        alt_path = PROJECT_ROOT / "VAE" / "models_weights" / "cluster_trained_data_leakage" / "vae_epoch_150_loss691.16.pth"
        if alt_path.exists():
            print(f"⚠️ 'best_vae_model.pth' not found. Using fallback: {alt_path.name}")
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"❌ Model weights not found at: {weights_path}")

    model = VAE().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def process_single_image(image_path, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_tensor = transform(img)

    return img_tensor.unsqueeze(0).to(device)


def visualize_reconstruction(model, image_path, device):
    input_tensor = process_single_image(image_path, device)

    with torch.no_grad():
        recon_tensor, _, _ = model(input_tensor)

    original_np = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    recon_np = recon_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

    original_np = np.clip(original_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Input")
    plt.imshow(original_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("VAE Reconstruction")
    plt.imshow(recon_np)
    plt.axis("off")

    plt.tight_layout()

    # [FIX] Force extra space at the top so titles aren't cropped
    plt.subplots_adjust(top=0.85)

    plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    apply_project_style("boardstate-dark.mplstyle")

    target_image_path = PROJECT_ROOT / "data" / "preprocessed_data" / "game4_frame_038816_tile_row3_column5_class1.png"

    try:
        vae = load_vae_model(MODEL_WEIGHTS_PATH, DEVICE)
        print(f"Visualizing reconstruction for: {target_image_path.name}")
        visualize_reconstruction(vae, target_image_path, DEVICE)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")