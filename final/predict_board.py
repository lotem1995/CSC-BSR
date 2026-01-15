import sys
import os
import torch
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

# --- CONFIGURATION: PATHS ---
BASE_DIR = Path(__file__).resolve().parent

BACKBONE_PATH = BASE_DIR / "chess_encoder_finetuned_dino-small_backbone.pt"
BINARY_GUARD_PATH = BASE_DIR / "binary_ood_dino_small_epoch3.pt"
CLASSIFIER_DB_PATH = BASE_DIR / "classifier_dino_small.pt"

# --- CONFIGURATION: CLASS MAPPING ---
# Maps Internal Model IDs -> Your Required Output Encoding [0-14]
INTERNAL_TO_OUTPUT = {
    3: 0,  # P (White Pawn) -> 0
    5: 1,  # R (White Rook) -> 1
    2: 2,  # N (White Knight) -> 2
    0: 3,  # B (White Bishop) -> 3
    4: 4,  # Q (White Queen) -> 4
    1: 5,  # K (White King) -> 5
    10: 6,  # p (Black Pawn) -> 6
    12: 7,  # r (Black Rook) -> 7
    9: 8,  # n (Black Knight) -> 8
    6: 9,  # b (Black Bishop) -> 9
    11: 10,  # q (Black Queen) -> 10
    8: 11,  # k (Black King) -> 11
    7: 12,  # empty -> 12
    17: 13  # Internal OOD -> 13
}

# --- SETUP IMPORTS ---
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from embedding.dinov2 import DINOv2Embedding
    from embedding.classifier import FENClassifier
    from preprocessing.splitting_images import slice_image_with_coordinates
    from drawing.draw_board import generate_ood_board
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- INITIALIZATION ---
print("Initializing Chess Predictor...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize Backbone
if not BACKBONE_PATH.exists():
    raise FileNotFoundError(f"Backbone model not found at {BACKBONE_PATH}")
dino_model = DINOv2Embedding(model_size="small")
checkpoint = torch.load(BACKBONE_PATH, map_location=DEVICE)
if 'model' in checkpoint:
    dino_model.model.load_state_dict(checkpoint['model'])
else:
    dino_model.model.load_state_dict(checkpoint)
dino_model.model.eval()

# 2. Initialize Classifier
if not CLASSIFIER_DB_PATH.exists():
    raise FileNotFoundError(f"Classifier DB not found at {CLASSIFIER_DB_PATH}")
classifier = FENClassifier(embedding_extractor=dino_model)
classifier.load(str(CLASSIFIER_DB_PATH))

# 3. Initialize Safety Guard
if not BINARY_GUARD_PATH.exists():
    raise FileNotFoundError(f"Binary Guard not found at {BINARY_GUARD_PATH}")
classifier.set_binary_model(str(BINARY_GUARD_PATH), dino_size="small")

print("✓ Models Loaded Successfully.\n")


def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.

    Args:
        image (np.ndarray): Input image array (H, W, 3) in RGB format.
                            Dtype uint8, Range [0, 255].

    Returns:
        torch.Tensor: Shape (8, 8), Dtype int64, on CPU.
                      Values 0-14 per strict encoding.
    """
    # 1. Validation
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

    try:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image_pil = Image.fromarray(image).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not convert input numpy array to Image: {e}")

    # 2. Slice
    tile_images = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_board_path = os.path.join(tmp_dir, "temp_board.jpg")
        image_pil.save(tmp_board_path)

        dummy_board = np.zeros((8, 8), dtype=int)

        slice_image_with_coordinates(
            "pred",  # game_name
            tmp_board_path,  # image_path
            tmp_dir,  # output_folder
            dummy_board,  # board
            overlap_percent=0.7,
            final_size=(224, 224)
        )

        base_name = os.path.splitext(os.path.basename(tmp_board_path))[0]
        for r in range(8):
            for c in range(8):
                fname = f"pred_{base_name}_tile_row{r}_column{c}_class0.png"
                tile_path = os.path.join(tmp_dir, fname)

                if not os.path.exists(tile_path):
                    raise FileNotFoundError(f"Slicing failed, missing tile: {fname}")

                with Image.open(tile_path) as img:
                    tile_images.append(img.convert("RGB").copy())

    # 3. Embed
    tile_embeddings = classifier.embedding_extractor.extract_batch_embeddings(tile_images)

    # 4. Predict
    preds, confs, is_ood = classifier.predict_with_ood(
        tile_embeddings,
        prediction_method="knn",
        ood_method="binary_ood_model",
        tile_images=tile_images
    )

    # 5. Mask OOD (Internal ID 17)
    internal_preds = preds.copy()
    internal_preds[is_ood] = 17

    # 6. Apply Class Mapping (Internal -> Target 0-14)
    final_preds = np.array([INTERNAL_TO_OUTPUT.get(p, 13) for p in internal_preds])

    # 7. Return Tensor (8x8)
    return torch.from_numpy(final_preds).long().cpu().view(8, 8)


# --- UPDATED TEST BLOCK ---
if __name__ == "__main__":
    target_image = PROJECT_ROOT / "data" / "game4_per_frame" / "tagged_images" / "frame_000616.jpg"
    print(f"Looking for image at: {target_image}")

    if target_image.exists():
        print("✓ File found. Running prediction...")
        try:
            with Image.open(target_image) as img:
                real_img_np = np.array(img.convert("RGB"))

            result_tensor = predict_board(real_img_np)
            print(result_tensor)

            print("\n✓ Prediction Successful!")
            print(f"Output Shape: {result_tensor.shape}")

            # --- SAVE VISUALIZATION TO ./results/ ---
            print("\nGenerating visual board representation...")

            results_dir = PROJECT_ROOT / "results"
            if not results_dir.exists():
                os.makedirs(results_dir)

            output_vis_path = str(results_dir / "prediction_visual.png")

            # [FIX IS HERE] Pass the TENSOR directly, do NOT convert to list
            generate_ood_board(
                result_tensor,  # <--- Correct: Passing Tensor
                output_vis_path
            )

            print(f"✓ Visualization saved to: {output_vis_path}")

        except Exception as e:
            print(f"\n❌ Prediction Failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"\n❌ ERROR: File not found.")
        print(f"Checked path: {target_image}")