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
INTERNAL_TO_OUTPUT = {
    0: 12,  # Model: empty        -> Target: 12
    1: 0,  # Model: white_pawn   -> Target: 0
    2: 2,  # Model: white_knight -> Target: 2
    3: 3,  # Model: white_bishop -> Target: 3
    4: 1,  # Model: white_rook   -> Target: 1
    5: 4,  # Model: white_queen  -> Target: 4
    6: 5,  # Model: white_king   -> Target: 5
    11: 6,  # Model: black_pawn   -> Target: 6
    12: 8,  # Model: black_knight -> Target: 8
    13: 9,  # Model: black_bishop -> Target: 9
    14: 7,  # Model: black_rook   -> Target: 7
    15: 10,  # Model: black_queen  -> Target: 10
    16: 11,  # Model: black_king   -> Target: 11
    17: 13  # Model: OOD          -> Target: 13
}

# --- SETUP IMPORTS ---
PROJECT_ROOT = BASE_DIR
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


def load_models():
    """
    Initializes the DINO backbone, FEN Classifier, and OOD Guard.

    Returns:
        classifier (FENClassifier): The fully loaded classifier object ready for prediction.
    """
    print("Initializing Chess Predictor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Backbone
    if not BACKBONE_PATH.exists():
        raise FileNotFoundError(f"Backbone model not found at {BACKBONE_PATH}")

    dino_model = DINOv2Embedding(model_size="small")
    checkpoint = torch.load(BACKBONE_PATH, map_location=device)

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
    return classifier


def predict_board(image: np.ndarray, classifier: FENClassifier) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.

    Args:
        image (np.ndarray): Input image array (H, W, 3) in RGB format.
        classifier (FENClassifier): The loaded classifier object.

    Returns:
        torch.Tensor: Shape (8, 8), Dtype int64, on CPU.
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

    # 3. Embed (Using passed classifier)
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

    # 6. Apply Class Mapping
    final_preds = np.array([INTERNAL_TO_OUTPUT.get(p, 13) for p in internal_preds])

    # 7. Return Tensor
    return torch.from_numpy(final_preds).long().cpu().view(8, 8)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Models (Once)
    loaded_classifier = load_models()

    # 2. Define Image Path
    target_image = PROJECT_ROOT / "data" / "game4_per_frame" / "tagged_images" / "frame_039084.jpg"
    print(f"Looking for image at: {target_image}")

    if target_image.exists():
        print("✓ File found. Running prediction...")
        try:
            # 3. Load Image
            with Image.open(target_image) as img:
                real_img_np = np.array(img.convert("RGB"))

            # 4. Predict (Pass image AND classifier)
            result_tensor = predict_board(real_img_np, loaded_classifier)

            print("\n✓ Prediction Successful!")
            print(f"Output Shape: {result_tensor.shape}")

            # 5. Visualization
            print("\nGenerating visual board representation...")
            results_dir = PROJECT_ROOT / "results"
            if not results_dir.exists():
                os.makedirs(results_dir)

            output_vis_path = str(results_dir / "prediction_visual.png")

            generate_ood_board(
                result_tensor,
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