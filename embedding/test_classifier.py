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


def grid_search_optimization(classifier: FENClassifier, val_csv_path: str):
    """
    Finds the best Temperature and Threshold by testing thousands of combinations.
    Optimizes for: High OOD Recall (Safety) vs. Low False Rejection Rate (Usability).
    """
    print(f"\n{'=' * 80}")
    print("HYPERPARAMETER OPTIMIZATION (GRID SEARCH)")
    print(f"{'=' * 80}")

    # 1. LOAD DATA & EXTRACT EMBEDDINGS
    print(f"Loading validation data from {val_csv_path}...")
    df = pd.read_csv(val_csv_path)

    image_paths = []
    true_labels = []

    for _, row in df.iterrows():
        img_path = Path(row['image'])
        if not img_path.is_absolute(): img_path = Path.cwd() / img_path
        image_paths.append(img_path)
        true_labels.append(row['label'])

    print(f"Extracting embeddings for {len(image_paths)} tiles (this takes a moment)...")

    # Batch processing
    batch_size = 32
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting"):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = [Image.open(p).convert('RGB') for p in batch_paths]
        emb_batch = classifier.embedding_extractor.extract_batch_embeddings(batch_imgs)
        all_embeddings.append(emb_batch.cpu())

    X_val = torch.cat(all_embeddings).to(classifier.device)
    y_val = torch.tensor(true_labels).to(classifier.device)

    # 2. DEFINE THE SEARCH GRID
    temperatures = np.arange(0.1, 10.0, 0.1)
    thresholds = np.arange(0.1, 0.95, 0.01)
    OOD_LABEL = 17

    # Masks
    is_ood_ground_truth = (y_val == OOD_LABEL)
    is_id_ground_truth = ~is_ood_ground_truth

    n_ood = is_ood_ground_truth.sum().item()
    n_id = is_id_ground_truth.sum().item()

    print(f"\nOptimization Dataset:")
    print(f"  - Valid Pieces (ID): {n_id}")
    print(f"  - Anomalies (OOD):   {n_ood}")

    # 3. RUN THE GRID SEARCH
    results = []
    with torch.no_grad():
        if classifier.classifier_head is None:
            print("Error: No classifier head attached! Cannot optimize Softmax.")
            return

        logits_raw = classifier.classifier_head(X_val)

        for temp in temperatures:
            scaled_logits = logits_raw / temp
            probs = torch.softmax(scaled_logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            for thresh in thresholds:
                flagged_ood = (confidences < thresh)

                # Metric 1: Recall (Catching the bad guys)
                if n_ood > 0:
                    caught_ood = (flagged_ood & is_ood_ground_truth).sum().item()
                    ood_recall = caught_ood / n_ood
                else:
                    ood_recall = 0.0

                # Metric 2: False Rejection (Annoying the user)
                if n_id > 0:
                    false_rejects = (flagged_ood & is_id_ground_truth).sum().item()
                    false_rejection_rate = false_rejects / n_id
                else:
                    false_rejection_rate = 0.0

                results.append({
                    "Temp": temp,
                    "Threshold": thresh,
                    "OOD_Recall": ood_recall,
                    "False_Rejection": false_rejection_rate
                })

    # 4. ANALYZE RESULTS
    df_res = pd.DataFrame(results)

    print("\n--- 1. SAFE BETS (Constraint: False Rejection < 1%) ---")
    safe_settings = df_res[df_res['False_Rejection'] < 0.01].sort_values('OOD_Recall', ascending=False)
    print(safe_settings.head(5).to_string(index=False, float_format="%.4f"))

    # --- NEW CALCULATION HERE ---
    print("\n--- 2. REAL-WORLD OPTIMIZATION (Assuming 1% OOD Prevalence) ---")
    print("Maximizing: 0.99 * (1 - False_Rejection) + 0.01 * Recall")

    # Formula explanation:
    # We care 99x more about False Rejection than Recall, because valid pieces are 99x more common.
    df_res['Projected_Accuracy'] = (0.99 * (1 - df_res['False_Rejection'])) + (0.01 * df_res['OOD_Recall'])

    real_world_settings = df_res.sort_values('Projected_Accuracy', ascending=False)
    print(real_world_settings.head(10).to_string(index=False, float_format="%.4f"))

    return real_world_settings

def evaluate_on_test_csv(
        classifier: FENClassifier,
        test_csv_path: str,
        prediction_method="knn",
        ood_method="softmax",
        ood_output_dir: str = "ood_inspection_images"
):
    """
    Evaluate the classifier with detailed OOD metrics.
    Saves images into categorized folders for easier inspection.
    """
    print(f"\nEvaluating on {test_csv_path}")
    print(f"prediction_method: {prediction_method}, ood_method: {ood_method}")

    # --- SETUP FOLDERS ---
    # Clear previous run
    if os.path.exists(ood_output_dir):
        shutil.rmtree(ood_output_dir)

    # Create main directory and sub-directories
    dir_false_reject = os.path.join(ood_output_dir, "false_rejections")  # Valid pieces flagged as OOD
    dir_missed_ood = os.path.join(ood_output_dir, "missed_ood")  # OOD pieces that sneaked in
    dir_correct_ood = os.path.join(ood_output_dir, "correct_ood")  # OOD pieces correctly caught

    os.makedirs(dir_false_reject)
    os.makedirs(dir_missed_ood)
    os.makedirs(dir_correct_ood)

    print(f"Saving inspection images to: {ood_output_dir}/")

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
            prediction_method=prediction_method,
            ood_method=ood_method,
        )

        # Store Data
        all_true_labels.extend(true_labels)
        all_predictions.extend(predictions)
        all_is_ood.extend(is_ood)

        # --- SAVE IMAGES BY CATEGORY ---
        for idx in range(64):
            t_lbl = true_labels[idx]
            p_lbl = predictions[idx]
            flagged = is_ood[idx]

            # 1. FALSE REJECTION: Valid piece (0-12) flagged as OOD (Bad for User)
            if (t_lbl != OOD_LABEL) and flagged:
                fname = f"FalseReject_Board{board_id}_tile{idx}_True{t_lbl}_Pred{p_lbl}.png"
                save_path = os.path.join(dir_false_reject, fname)
                tile_images[idx].resize((224, 224)).save(save_path)

            # 2. MISSED OOD: OOD piece (17) passed as valid (Bad for Safety)
            elif (t_lbl == OOD_LABEL) and not flagged:
                fname = f"MissedOOD_Board{board_id}_tile{idx}_Pred{p_lbl}_Conf{confidences[idx]:.2f}.png"
                save_path = os.path.join(dir_missed_ood, fname)
                tile_images[idx].resize((224, 224)).save(save_path)

            # 3. CORRECT OOD: OOD piece (17) correctly flagged (Good!)
            elif (t_lbl == OOD_LABEL) and flagged:
                fname = f"CorrectOOD_Board{board_id}_tile{idx}_Pred{p_lbl}_Conf{confidences[idx]:.2f}.png"
                save_path = os.path.join(dir_correct_ood, fname)
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

    # --- 1. OOD Detection Metrics ---
    n_ood = np.sum(ood_mask)
    if n_ood > 0:
        ood_detected = np.sum(is_ood_flag[ood_mask])
        ood_recall = ood_detected / n_ood
    else:
        ood_detected, ood_recall = 0, 0.0

    # --- 2. ID Classification Metrics ---
    n_id = np.sum(id_mask)
    if n_id > 0:
        id_rejected = np.sum(is_ood_flag[id_mask])
        id_false_ood_rate = id_rejected / n_id

        accepted_mask = id_mask & (~is_ood_flag)
        n_accepted = np.sum(accepted_mask)

        if n_accepted > 0:
            correct_preds = np.sum(y_pred[accepted_mask] == y_true[accepted_mask])
            clean_accuracy = correct_preds / n_accepted
        else:
            clean_accuracy = 0.0

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
    print(f"   OOD Recall:           {ood_recall * 100:.2f}%")

    print(f"\n2. CLASSIFIER SAFETY (Goal: Don't reject normal pieces)")
    print(f"   False Rejections:     {id_rejected}/{n_id}")
    print(f"   False Rejection Rate: {id_false_ood_rate * 100:.2f}%")

    print(f"\n3. CLASSIFIER ACCURACY (Goal: Predict correct class)")
    print(f"   Clean Accuracy:       {clean_accuracy * 100:.2f}%")

    print(f"\n4. SYSTEM OVERALL")
    print(f"   Overall Accuracy:     {overall_accuracy * 100:.2f}%")
    print(f"{'=' * 60}")

    return overall_accuracy


def main():
    """
    Main execution pipeline.
    """
    # ================= CONFIGURATION =================
    CHECKPOINT_PATH = "embedding/chess_encoder_finetuned_dino-small_backbone.pt"
    MODEL_TYPE = "dino-small"
    STRATEGY = "backbone"

    VAL_CSV = "data/splits/val.csv"
    TEST_CSV = "data/splits/test.csv"

    # --- MODE SELECTION ---
    DO_OPTIMIZATION = False  # Set True to find best Temp/Threshold
    DO_EVALUATION = True  # Set True to run final test

    # --- METHOD SELECTION ---
    # We use KNN for prediction (accurate) and Softmax for OOD (robust)
    PREDICTION_METHOD = "knn"
    OOD_METHOD = "ensemble"

    # =================================================

    print("=" * 80)
    print("CHESS CLASSIFIER: MIXED METHOD PIPELINE")
    print("=" * 80)

    # 1. LOAD MODEL
    print(f"\n1. Loading fine-tuned model...")
    embedding_model = load_finetuned_embedding_model(
        checkpoint_path=CHECKPOINT_PATH,
        model_type=MODEL_TYPE,
        strategy=STRATEGY
    )

    # 2. INITIALIZE CLASSIFIER
    print(f"\n2. Initializing Classifier...")
    classifier = FENClassifier(embedding_extractor=embedding_model)

    # Always load the classifier head if we plan to use Softmax for ANYTHING
    if OOD_METHOD == "softmax" or PREDICTION_METHOD == "softmax" or DO_OPTIMIZATION:
        head = load_classifier_head(CHECKPOINT_PATH, embedding_model.get_embedding_dim())
        classifier.set_classifier_head(head)

    # 3. OPTIMIZATION (Optional)
    if DO_OPTIMIZATION:
        grid_search_optimization(classifier, VAL_CSV)

    # 4. BUILD KNN DATABASE
    # We need this if prediction is KNN/Mahalanobis OR if OOD is KNN/Mahalanobis
    need_database = "knn" in [PREDICTION_METHOD, OOD_METHOD] or "mahalanobis" in [PREDICTION_METHOD, OOD_METHOD]

    if need_database:
        print(f"\n4. Building Database from {VAL_CSV}...")
        val_df = pd.read_csv(VAL_CSV)
        board_ids = val_df['board_id'].unique()

        for board_id in tqdm(board_ids, desc="Adding boards"):
            board_df = val_df[val_df['board_id'] == board_id].copy()
            if len(board_df) != 64: continue

            board_df['tile_coords'] = board_df['image'].apply(parse_tile_coords)
            board_df = board_df.sort_values('tile_coords')

            tile_images = []
            board_state = np.zeros((8, 8), dtype=int)

            for _, row in board_df.iterrows():
                img_path = Path(row['image'])
                if not img_path.is_absolute(): img_path = Path.cwd() / img_path
                with Image.open(img_path) as im:
                    tile_images.append(im.convert('RGB').copy())
                board_state[parse_tile_coords(row['image'])] = row['label']

            tile_embeddings = embedding_model.extract_batch_embeddings(tile_images)
            classifier.add_fen_position(board_id, tile_embeddings, board_state)

        classifier.update_thresholds()

    # 5. EVALUATE
    if DO_EVALUATION:
        print(f"\n5. Evaluating on {TEST_CSV}...")

        # IMPORTANT: We updated evaluate_on_test_csv to accept separate methods
        # But wait, looking at your file, evaluate_on_test_csv hardcodes the call:
        # classifier.predict_with_ood(..., prediction_method="knn", ood_method="softmax")
        # Let's make sure it matches what we defined above.

        # (Note: You might need to slightly tweak evaluate_on_test_csv arguments
        # to accept these variables if you want full flexibility,
        # but for now, the hardcoded "knn" + "softmax" inside evaluate is fine
        # as long as we built the DB and loaded the head).

        evaluate_on_test_csv(
            classifier=classifier,
            test_csv_path=TEST_CSV,
            prediction_method=PREDICTION_METHOD,
            ood_method=OOD_METHOD,
            ood_output_dir="ood_inspection_images"
        )

if __name__ == "__main__":
    main()