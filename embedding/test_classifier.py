"""
Example: Using Fine-Tuned Embeddings with FENClassifier

This script demonstrates how to:
1. Load a fine-tuned model from fine_tune.py
2. Extract embeddings using the fine-tuned backbone
3. Use those embeddings in the per-tile classifier
4. Evaluate on test.csv

IMPORTANT: This script must be run from the project root directory:
    cd /home/lotems/Documents/DL_Oren/CSC-BSR
    python embedding/test_classifier.py

The CSV files contain paths relative to project root (e.g., 'preprocessed_data/...').
"""
import os
import shutil
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from typing import List
import pandas as pd
from tqdm import tqdm
import re
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.embedding_base import EmbeddingModel
from embedding.classifier import FENClassifier
from embedding.dinov2 import DINOv2Embedding
from embedding.qwen3 import QwenVisionEmbedding


class FineTunedEmbeddingModel(EmbeddingModel):
    """
    Wrapper for fine-tuned models that extracts embeddings BEFORE the classifier head.
    
    This class loads a checkpoint from fine_tune.py and provides the EmbeddingModel interface
    for use with FENClassifier.
    """
    
    def __init__(self, checkpoint_path: str, base_model: EmbeddingModel):
        """
        Args:
            checkpoint_path: Path to saved checkpoint from fine_tune.py
            base_model: The base embedding model (QwenVisionEmbedding or DINOv2Embedding)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = base_model
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"Loaded fine-tuned checkpoint from: {checkpoint_path}")
        print(f"Original model: {checkpoint.get('embedding_model_name', 'Unknown')}")
        
        # The base_model now has fine-tuned weights if strategy was 'backbone'
        # For 'head-only' strategy, base_model weights are unchanged (only classifier was trained)
        # For 'lora' strategy, the LoRA adapters are merged into the model
        
        # We only use the base_model for embeddings, not the classifier head
        self.embedding_dim = base_model.get_embedding_dim()
        
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract embedding from fine-tuned model (before classifier head)."""
        return self.base_model.extract_embedding(image)
    
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract batch embeddings from fine-tuned model (before classifier head)."""
        return self.base_model.extract_batch_embeddings(images)
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    def __repr__(self):
        return f"FineTunedEmbeddingModel({self.base_model.__class__.__name__})"


class FineTunedDINOBackbone(EmbeddingModel):
    """
    Special wrapper for DINO models fine-tuned with 'backbone' strategy.
    Loads the fine-tuned backbone weights directly.
    """
    
    def __init__(self, checkpoint_path: str, model_size: str = "small"):
        """
        Args:
            checkpoint_path: Path to saved checkpoint from fine_tune.py with strategy='backbone'
            model_size: 'small', 'base', 'large', or 'giant'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create base DINO model
        base_dino = DINOv2Embedding(model_size=model_size)
        
        # Load fine-tuned backbone weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        base_dino.model.load_state_dict(checkpoint['model'])
        base_dino.model.eval()  # Ensure eval mode after loading weights
        print(f"Loaded fine-tuned DINO backbone from: {checkpoint_path}")
        
        self.base_dino = base_dino
        self.embedding_dim = base_dino.get_embedding_dim()
        
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract embedding from fine-tuned DINO backbone."""
        return self.base_dino.extract_embedding(image)
    
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract batch embeddings from fine-tuned DINO backbone."""
        return self.base_dino.extract_batch_embeddings(images)
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    def __repr__(self):
        return f"FineTunedDINOBackbone(size={self.base_dino.model_size})"


def load_finetuned_embedding_model(
    checkpoint_path: str,
    model_type: str = "dino-small",
    strategy: str = "head-only"
) -> EmbeddingModel:
    """
    Factory function to load the appropriate fine-tuned embedding model.
    
    Args:
        checkpoint_path: Path to checkpoint saved by fine_tune.py
        model_type: "qwen", "dino-small", "dino-base", etc.
        strategy: "head-only", "backbone", or "lora"
        
    Returns:
        EmbeddingModel instance with fine-tuned weights
    """
    if strategy == "backbone":
        # For backbone fine-tuning, load the fine-tuned weights into the model
        if "dino" in model_type.lower():
            size = model_type.split("-")[-1] if "-" in model_type else "small"
            return FineTunedDINOBackbone(checkpoint_path, model_size=size)
        else:
            raise ValueError("Backbone fine-tuning only supported for DINO models")
    
    elif strategy == "head-only":
        # For head-only, the backbone is unchanged, so use the base model
        if model_type == "qwen":
            base_model = QwenVisionEmbedding()
        elif "dino" in model_type.lower():
            size = model_type.split("-")[-1] if "-" in model_type else "small"
            base_model = DINOv2Embedding(model_size=size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return FineTunedEmbeddingModel(checkpoint_path, base_model)
    
    elif strategy == "lora":
        # For LoRA, need to load the merged model
        raise NotImplementedError("LoRA loading not yet implemented - use head-only or backbone")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def parse_tile_coords(image_path: str) -> tuple:
    """
    Parse row and column from tile filename.
    
    Tiles are named: *_tile_row{r}_column{c}_class{label}.png
    Returns (row, col) or raises ValueError if not parseable.
    """
    match = re.search(r'_tile_row(\d+)_column(\d+)_', image_path)
    if not match:
        raise ValueError(f"Cannot parse tile coordinates from: {image_path}")
    return int(match.group(1)), int(match.group(2))





def load_classifier_head(checkpoint_path: str, embedding_dim: int) -> nn.Module:
    """
    Reconstructs the classifier head architecture and loads weights.
    """
    print(f"Loading classifier head from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DEFINITION MUST MATCH fine_tune.py EXACTLY
    classifier = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(embedding_dim // 2, 13),
    )

    # Load the weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'classifier' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'classifier' state_dict")

    classifier.load_state_dict(checkpoint['classifier'])
    classifier.to(device)
    classifier.eval()  # Freeze it
    return classifier


def evaluate_on_test_csv(
        classifier: FENClassifier,
        test_csv_path: str,
        data_root: str = "data",
        method: str = "mahalanobis",
        ood_output_dir: str = "ood_failures"
):
    """
    Evaluate the classifier with detailed OOD metrics.
    Separates 'Safety' (False Rejections) from 'Detection' (Recall on Class 17).
    """
    print(f"\nEvaluating on {test_csv_path}")
    print(f"Method: {method}")

    if os.path.exists(ood_output_dir):
        shutil.rmtree(ood_output_dir)
    os.makedirs(ood_output_dir)
    print(f"Saving OOD inspection images to: {ood_output_dir}/")

    # Load test CSV
    df = pd.read_csv(test_csv_path)
    board_ids = df['board_id'].unique()

    # --- STORAGE FOR METRICS ---
    all_true_labels = []
    all_predictions = []
    all_is_ood = []

    # Define the OOD Label
    OOD_LABEL = 17

    for board_id in tqdm(board_ids, desc="Evaluating boards"):
        board_df = df[df['board_id'] == board_id].copy()

        if len(board_df) != 64:
            continue

        # Sort and Load
        board_df['tile_coords'] = board_df['image'].apply(parse_tile_coords)
        board_df = board_df.sort_values('tile_coords')

        tile_images = []
        true_labels = []
        for _, row in board_df.iterrows():
            img_path = Path(row['image'])
            if not img_path.is_absolute():
                img_path = Path.cwd() / img_path

            with Image.open(img_path) as im:
                tile_images.append(im.convert('RGB').copy())
            true_labels.append(row['label'])

        # Predict
        tile_embeddings = classifier.embedding_extractor.extract_batch_embeddings(tile_images)
        predictions, confidences, is_ood = classifier.predict_with_ood(
            tile_embeddings,
            method=method
        )

        # Store Data
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
        all_is_ood.extend(is_ood)

        # --- SAVE FAILURE IMAGES ---
        # We save images if:
        # 1. False Rejection: It was valid (0-12), but we flagged it as OOD.
        # 2. Missed OOD: It was OOD (17), but we let it pass.

        # Check if we need to inspect this board
        has_false_rejection = np.any((np.array(true_labels) != OOD_LABEL) & is_ood)
        has_missed_ood = np.any((np.array(true_labels) == OOD_LABEL) & (~is_ood))

        if has_false_rejection or has_missed_ood:
            for idx in range(64):
                t_lbl = true_labels[idx]
                p_lbl = predictions[idx]
                flagged = is_ood[idx]

                # Logic to name files helpfully
                if (t_lbl != OOD_LABEL) and flagged:
                    # Case A: False Rejection (Bad for user experience)
                    fname = f"FalseReject_{board_id}_tile{idx}_True{t_lbl}.png"
                    save_path = os.path.join(ood_output_dir, fname)
                    tile_images[idx].resize((224, 224)).save(save_path)

                elif (t_lbl == OOD_LABEL) and not flagged:
                    # Case B: Missed OOD (Bad for safety)
                    fname = f"MissedOOD_{board_id}_tile{idx}_Pred{p_lbl}.png"
                    save_path = os.path.join(ood_output_dir, fname)
                    tile_images[idx].resize((224, 224)).save(save_path)

    # ---------------------------------------------------------
    # CALCULATE METRICS
    # ---------------------------------------------------------
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    is_ood_flag = np.array(all_is_ood)

    # Masks
    ood_mask = (y_true == OOD_LABEL)  # The "Real" 17s
    id_mask = ~ood_mask  # The Normal pieces (0-12)

    # --- 1. OOD Detection Metrics (Ability to catch Class 17) ---
    n_ood = np.sum(ood_mask)
    if n_ood > 0:
        ood_detected = np.sum(is_ood_flag[ood_mask])
        ood_recall = ood_detected / n_ood
    else:
        ood_detected, ood_recall = 0, 0.0

    # --- 2. ID Classification Metrics (Ability to classify 0-12) ---
    n_id = np.sum(id_mask)
    if n_id > 0:
        # False Rejection Rate: How many valid pieces did we accidentally reject?
        id_rejected = np.sum(is_ood_flag[id_mask])
        id_false_ood_rate = id_rejected / n_id

        # Clean Accuracy: Accuracy ONLY on tiles we accepted
        accepted_mask = id_mask & (~is_ood_flag)
        n_accepted = np.sum(accepted_mask)

        if n_accepted > 0:
            correct_preds = np.sum(y_pred[accepted_mask] == y_true[accepted_mask])
            clean_accuracy = correct_preds / n_accepted
        else:
            clean_accuracy = 0.0

        # Overall Strict Accuracy
        # Correct = (ID & Accepted & Correct) + (OOD & Rejected)
        correct_id_cnt = np.sum((y_pred[id_mask] == y_true[id_mask]) & (~is_ood_flag[id_mask]))
        total_correct = correct_id_cnt + ood_detected
        overall_accuracy = total_correct / len(y_true)
    else:
        id_false_ood_rate, clean_accuracy, overall_accuracy = 0.0, 0.0, 0.0

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"DETAILED TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Total Samples: {len(y_true)}")
    print(f"  - Normal Pieces (0-12): {n_id}")
    print(f"  - Anomalies (Class 17): {n_ood}")

    print(f"\n1. OOD DETECTION (Goal: Catch the 17s)")
    print(f"   Correctly Caught:     {ood_detected}/{n_ood}")
    print(f"   OOD Recall:           {ood_recall * 100:.2f}%  (Target: >80-90%)")

    print(f"\n2. CLASSIFIER SAFETY (Goal: Don't reject normal pieces)")
    print(f"   False Rejections:     {id_rejected}/{n_id}")
    print(f"   False Rejection Rate: {id_false_ood_rate * 100:.2f}% (Target: <5%)")

    print(f"\n3. CLASSIFIER ACCURACY (Goal: Predict correct class)")
    print(f"   Clean Accuracy:       {clean_accuracy * 100:.2f}%     (On accepted tiles)")

    print(f"\n4. SYSTEM OVERALL")
    print(f"   Overall Accuracy:     {overall_accuracy * 100:.2f}%")
    print(f"{'=' * 60}")

    return overall_accuracy

def grid_search_softmax(classifier: FENClassifier, test_csv_path: str):
    """
    Efficiently searches for the best Temperature and Threshold.
    Loads data ONCE, then runs pure math loops.
    """
    print(f"\n{'=' * 60}")
    print("STARTING HYPERPARAMETER GRID SEARCH")
    print(f"{'=' * 60}")

    # 1. LOAD ALL DATA INTO MEMORY
    df = pd.read_csv(test_csv_path)
    print(f"Loading {len(df)} test images into memory...")

    all_embeddings = []
    all_labels = []

    image_paths = []
    labels = []

    # This loop is fast (just strings)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading CSV paths"):
        img_path = Path(row['image'])
        if not img_path.is_absolute(): img_path = Path.cwd() / img_path
        image_paths.append(img_path)
        labels.append(row['label'])

    # Process in batches to save RAM/VRAM
    BATCH_SIZE = 32

    # === ADDED TQDM HERE to show "Extracting Embeddings" progress ===
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), total=total_batches,
                  desc="Extracting Embeddings (Slow Step)"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_imgs = [Image.open(p).convert('RGB') for p in batch_paths]
        emb_batch = classifier.embedding_extractor.extract_batch_embeddings(batch_imgs)
        all_embeddings.append(emb_batch.cpu())  # Store on CPU to avoid VRAM overflow
        all_labels.extend(labels[i:i + BATCH_SIZE])

    # Convert to giant Tensors
    X_test = torch.cat(all_embeddings).to(classifier.device)
    y_test = torch.tensor(all_labels).to(classifier.device)

    # 2. PRE-CALCULATE LOGITS
    print("\nPre-calculating raw model outputs...")
    with torch.no_grad():
        classifier.classifier_head.eval()
        base_logits = classifier.classifier_head(X_test)

    # 3. RUN THE GRID SEARCH
    print("\nRunning Grid Search...")

    temperatures = [0.8, 1.0, 1.5, 2.0, 2.5, 5.0]
    thresholds   = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    results = []

    # Use tqdm here too just in case math takes a second
    total_combos = len(temperatures) * len(thresholds)
    with tqdm(total=total_combos, desc="Calculating Metrics") as pbar:
        for temp in temperatures:
            scaled_logits = base_logits / temp
            probs = torch.softmax(scaled_logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            is_correct = (predictions == y_test)

            for thresh in thresholds:
                is_ood = confidences < thresh

                acc = is_correct.float().mean().item()
                ood_rate = is_ood.float().mean().item()
                silent_errors = (~is_correct & ~is_ood).float().mean().item()

                results.append({
                    "Temp": temp,
                    "Thresh": thresh,
                    "Accuracy": acc,
                    "OOD_Rate": ood_rate,
                    "Silent_Err": silent_errors
                })
                pbar.update(1)

    # 4. PRINT RESULTS TABLE
    results_df = pd.DataFrame(results)

    # === NEW: Show ALL rows ===
    pd.set_option('display.max_rows', None)  # Disable truncation

    # Sort by OOD Rate so you can easily find your 3% target
    # Secondary sort by Silent_Err so the best options appear first in that block
    sorted_df = results_df.sort_values(by=["OOD_Rate", "Silent_Err"], ascending=[True, True])

    print("\nFULL GRID SEARCH RESULTS (Sorted by OOD Rate):")
    print(sorted_df.to_string(index=False))


def main():
    """
    Proper evaluation setup to avoid data leakage:
    - Train set was used to fine-tune the embedding backbone
    - Val set is used to build the KNN/Mahalanobis retrieval database
    - Test set is used for final evaluation
    """
    # Configuration
    CHECKPOINT_PATH = "embedding/chess_encoder_finetuned_dino-small_backbone.pt"
    MODEL_TYPE = "dino-small"
    STRATEGY = "backbone"  # "head-only", "backbone", or "lora"
    
    VAL_CSV = "data/splits/val.csv"  # Use val for database (NOT train - that was used for fine-tuning!)
    TEST_CSV = "data/splits/test.csv"

    METHOD = "knn"
    
    print("="*80)
    print("USING FINE-TUNED EMBEDDINGS WITH PER-TILE CLASSIFIER")
    print("="*80)
    print("\nDATA SPLIT STRATEGY (to avoid leakage):")
    print("  - Train set: Used to fine-tune embedding backbone ✓")
    print("  - Val set:   Used to build KNN/Mahalanobis database ← current step")
    print("  - Test set:  Used for final evaluation")
    
    # Step 1: Load fine-tuned embedding model
    print(f"\n1. Loading fine-tuned model...")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Model: {MODEL_TYPE}")
    print(f"   Strategy: {STRATEGY}")
    
    embedding_model = load_finetuned_embedding_model(
        checkpoint_path=CHECKPOINT_PATH,
        model_type=MODEL_TYPE,
        strategy=STRATEGY
    )
    
    # Step 2: Initialize classifier with fine-tuned embeddings
    print(f"\n2. Initializing FENClassifier with fine-tuned embeddings...")
    classifier = FENClassifier(embedding_extractor=embedding_model)

    if METHOD == "softmax":
        head = load_classifier_head(CHECKPOINT_PATH, embedding_model.get_embedding_dim())
        classifier.set_classifier_head(head)

        # === OPTION: RUN GRID SEARCH INSTEAD OF NORMAL TEST ===
        # run_grid_search = True
        run_grid_search = False  # Set to True to optimize parameters

        if run_grid_search:
            grid_search_softmax(classifier, VAL_CSV)
            return  # Stop here after search


    else:

        # Step 3: Load validation data for KNN/Mahalanobis database
        print(f"\n3. Loading validation data from {VAL_CSV}...")
        print(f"   (Using VAL not TRAIN to avoid leakage - train was used for fine-tuning)")
        val_df = pd.read_csv(VAL_CSV)
        board_ids = val_df['board_id'].unique()
        print(f"   Found {len(board_ids)} validation boards")



        # Add validation boards to classifier database
        print(f"\n4. Building per-tile database from validation set...")
        for board_id in tqdm(board_ids, desc="Adding boards"):
            board_df = val_df[val_df['board_id'] == board_id].copy()

            if len(board_df) != 64:
                continue

            # Parse tile coordinates and sort by (row, col) to ensure correct ordering
            board_df['tile_coords'] = board_df['image'].apply(parse_tile_coords)
            board_df = board_df.sort_values('tile_coords')

            # Load tile images and labels in correct spatial order
            tile_images = []
            board_state = np.zeros((8, 8), dtype=int)

            for _, row in board_df.iterrows():
                # Parse row/col from filename (guaranteed by sorting above)
                tile_row, tile_col = parse_tile_coords(row['image'])

                # CSV paths are relative to project root (e.g., 'preprocessed_data/...')
                img_path = Path(row['image'])
                if not img_path.is_absolute():
                    img_path = Path.cwd() / img_path

                with Image.open(img_path) as im:
                    tile_images.append(im.convert('RGB').copy())

                board_state[tile_row, tile_col] = row['label']

            # Extract embeddings using fine-tuned model
            tile_embeddings = embedding_model.extract_batch_embeddings(tile_images)

            # Add to classifier
            classifier.add_fen_position(
                fen=board_id,  # Use board_id as FEN placeholder
                tile_embeddings=tile_embeddings,
                board_state=board_state
            )


        # 1. Save the data you just built (so you can use it next time)
        classifier_path = "classifier.json"
        classifier.save(str(classifier_path))

        # # 2. DO NOT LOAD HERE! (This was wiping your data)
        # classifier.load(str(classifier_path))

        # Step 5: Build index (uses the data currently in memory)
        print(f"\n5. Building KNN indices...")
        classifier.update_thresholds()
    
    # Step 6: Evaluate on test set
    print(f"\n6. Evaluating on test set...")
    accuracy = evaluate_on_test_csv(
        classifier=classifier,
        test_csv_path=TEST_CSV,
        method=METHOD,
        ood_output_dir="ood_inspection_images"
    )
    
    print(f"\n✓ Done! Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()