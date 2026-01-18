"""
Benchmark Suite for Chess Classifier

This script runs a comprehensive evaluation of different prediction and OOD detection
combinations (KNN, Softmax, Mahalanobis, Binary Model, etc.) on the test set.

Usage:
    cd /home/lotems/Documents/DL_Oren/CSC-BSR
    python embedding/benchmark.py
"""

import os
import sys
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import re
from typing import List

# --- PROJECT SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.embedding_base import EmbeddingModel
from embedding.classifier import FENClassifier
from embedding.dinov2 import DINOv2Embedding
from embedding.qwen3 import QwenVisionEmbedding


# ==================================================================================
# 1. HELPER CLASSES & FUNCTIONS (Replicated for standalone execution)
# ==================================================================================

class FineTunedEmbeddingModel(EmbeddingModel):
    def __init__(self, checkpoint_path: str, base_model: EmbeddingModel):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = base_model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.embedding_dim = base_model.get_embedding_dim()

    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        return self.base_model.extract_embedding(image)

    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        return self.base_model.extract_batch_embeddings(images)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class FineTunedDINOBackbone(EmbeddingModel):
    def __init__(self, checkpoint_path: str, model_size: str = "small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dino = DINOv2Embedding(model_size=model_size)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        base_dino.model.load_state_dict(checkpoint['model'])
        base_dino.model.eval()
        self.base_dino = base_dino
        self.embedding_dim = base_dino.get_embedding_dim()

    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        return self.base_dino.extract_embedding(image)

    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        return self.base_dino.extract_batch_embeddings(images)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


def load_finetuned_embedding_model(checkpoint_path: str, model_type: str = "dino-small",
                                   strategy: str = "head-only") -> EmbeddingModel:
    if strategy == "backbone":
        if "dino" in model_type.lower():
            size = model_type.split("-")[-1] if "-" in model_type else "small"
            return FineTunedDINOBackbone(checkpoint_path, model_size=size)
        else:
            raise ValueError("Backbone fine-tuning only supported for DINO models")
    elif strategy == "head-only":
        if model_type == "qwen":
            base_model = QwenVisionEmbedding()
        elif "dino" in model_type.lower():
            size = model_type.split("-")[-1] if "-" in model_type else "small"
            base_model = DINOv2Embedding(model_size=size)
        return FineTunedEmbeddingModel(checkpoint_path, base_model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def load_classifier_head(checkpoint_path: str, embedding_dim: int) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(embedding_dim // 2, 13),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['classifier'])
    classifier.to(device)
    classifier.eval()
    return classifier


def parse_tile_coords(image_path: str) -> tuple:
    match = re.search(r'_tile_row(\d+)_column(\d+)_', image_path)
    if not match: raise ValueError(f"Cannot parse tile coordinates from: {image_path}")
    return int(match.group(1)), int(match.group(2))


# ==================================================================================
# 2. EVALUATION LOGIC
# ==================================================================================

def evaluate_single_run(
        classifier: FENClassifier,
        test_csv_path: str,
        prediction_method: str,
        ood_method: str,
        output_dir: str
) -> dict:
    """
    Executes one evaluation run and returns a dictionary of metrics.
    """
    # Create directory for failure images
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    dir_false_reject = os.path.join(output_dir, "false_rejections")
    dir_missed_ood = os.path.join(output_dir, "missed_ood")
    os.makedirs(dir_false_reject)
    os.makedirs(dir_missed_ood)

    df = pd.read_csv(test_csv_path)
    board_ids = df['board_id'].unique()
    OOD_LABEL = 17

    all_true = []
    all_pred = []
    all_ood_flags = []

    # Run Prediction Loop
    for board_id in board_ids:
        board_df = df[df['board_id'] == board_id].copy()
        if len(board_df) != 64: continue

        board_df['tile_coords'] = board_df['image'].apply(parse_tile_coords)
        board_df = board_df.sort_values('tile_coords')

        tile_images = []
        true_labels = []
        for _, row in board_df.iterrows():
            img_path = Path(row['image'])
            if not img_path.is_absolute(): img_path = Path.cwd() / img_path
            with Image.open(img_path) as im:
                tile_images.append(im.convert('RGB').copy())
            true_labels.append(row['label'])

        tile_embeddings = classifier.embedding_extractor.extract_batch_embeddings(tile_images)

        # Predict
        preds, confs, is_ood = classifier.predict_with_ood(
            tile_embeddings,
            prediction_method=prediction_method,
            ood_method=ood_method,
            tile_images=tile_images
        )

        all_true.extend(true_labels)
        all_pred.extend(preds)
        all_ood_flags.extend(is_ood)

        # Save specific failure cases (limit to 5 per board to save space/time)
        failures = 0
        for i in range(64):
            if failures > 5: break
            t_lbl, p_lbl, flagged = true_labels[i], preds[i], is_ood[i]

            # Save False Rejections (Valid piece flagged OOD)
            if t_lbl != OOD_LABEL and flagged:
                fname = f"{board_id}_tile{i}_True{t_lbl}_Pred{p_lbl}.png"
                tile_images[i].save(os.path.join(dir_false_reject, fname))
                failures += 1

            # Save Missed OOD (OOD piece allowed in)
            elif t_lbl == OOD_LABEL and not flagged:
                fname = f"{board_id}_tile{i}_Pred{p_lbl}_Conf{confs[i]:.2f}.png"
                tile_images[i].save(os.path.join(dir_missed_ood, fname))
                failures += 1

    # Calculate Metrics
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    is_ood = np.array(all_ood_flags)

    ood_mask = (y_true == OOD_LABEL)
    id_mask = ~ood_mask

    # 1. OOD Recall (Caught / Total OOD)
    n_ood = np.sum(ood_mask)
    ood_recall = np.sum(is_ood[ood_mask]) / n_ood if n_ood > 0 else 0.0

    # 2. False Rejection Rate (Flagged Valid / Total Valid)
    n_id = np.sum(id_mask)
    false_reject_rate = np.sum(is_ood[id_mask]) / n_id if n_id > 0 else 0.0

    # 3. Clean Accuracy (Correct Class / Total Accepted Valid)
    accepted_mask = id_mask & (~is_ood)
    n_accepted = np.sum(accepted_mask)
    clean_acc = np.sum(y_pred[accepted_mask] == y_true[accepted_mask]) / n_accepted if n_accepted > 0 else 0.0

    # 4. Overall Accuracy
    # (Correctly accepted valid ID + Correctly rejected OOD) / Total
    correct_id_cnt = np.sum((y_pred[id_mask] == y_true[id_mask]) & (~is_ood[id_mask]))
    ood_detected_cnt = np.sum(is_ood[ood_mask])
    overall_acc = (correct_id_cnt + ood_detected_cnt) / len(y_true)

    return {
        "Overall_Acc": overall_acc,
        "OOD_Recall": ood_recall,
        "False_Reject": false_reject_rate,
        "Clean_Acc": clean_acc
    }


# ==================================================================================
# 3. MAIN BENCHMARK LOOP
# ==================================================================================

def run_benchmark_suite(classifier: FENClassifier, test_csv_path: str):
    print(f"\n{'#' * 80}")
    print("STARTING BENCHMARK SUITE")
    print(f"{'#' * 80}\n")

    results = []

    # Define combinations
    # Prediction methods: How we decide the class if it IS NOT OOD
    pred_methods = ['knn', 'softmax']

    # OOD methods: How we decide IF it is OOD
    # Note: 'mahalanobis' is an OOD method, usually paired with knn prediction
    ood_methods = ['binary_ood_model', 'softmax', 'knn', 'mahalanobis', 'ensemble']

    total_runs = len(pred_methods) * len(ood_methods)
    pbar = tqdm(total=total_runs, desc="Benchmarking Combinations")

    for pred in pred_methods:
        for ood in ood_methods:
            pbar.set_description(f"Testing: {pred} + {ood}")

            combo_name = f"{pred}_{ood}"
            out_dir = os.path.join("benchmark_results", combo_name)

            try:
                metrics = evaluate_single_run(classifier, test_csv_path, pred, ood, out_dir)

                results.append({
                    "Prediction": pred,
                    "OOD_Method": ood,
                    "Overall Acc": metrics['Overall_Acc'],
                    "OOD Recall": metrics['OOD_Recall'],
                    "False Rejection": metrics['False_Reject'],
                    "Clean Acc": metrics['Clean_Acc'],
                    "Status": "OK"
                })
            except Exception as e:
                # Log error but don't stop the whole suite
                results.append({
                    "Prediction": pred,
                    "OOD_Method": ood,
                    "Overall Acc": 0.0,
                    "OOD Recall": 0.0,
                    "False Rejection": 0.0,
                    "Clean Acc": 0.0,
                    "Status": f"Error: {str(e)}"
                })

            pbar.update(1)

    pbar.close()

    # --- REPORTING ---
    df = pd.DataFrame(results)

    # Formatting for display
    print(f"\n{'=' * 80}")
    print("FINAL LEADERBOARD (Sorted by Overall Accuracy)")
    print(f"{'=' * 80}")

    df_display = df.sort_values("Overall Acc", ascending=False).copy()

    # Convert to percentages
    for col in ["Overall Acc", "OOD Recall", "False Rejection", "Clean Acc"]:
        df_display[col] = (df_display[col] * 100).map("{:.2f}%".format)

    print(df_display.to_string(index=False))

    # Save to CSV
    os.makedirs("benchmark_results", exist_ok=True)
    csv_path = "benchmark_results/leaderboard.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")


# ==================================================================================
# 4. EXECUTION
# ==================================================================================

def main():
    # --- CONFIG ---
    CHECKPOINT_PATH = "embedding/chess_encoder_finetuned_dino-small_backbone.pt"
    MODEL_TYPE = "dino-small"
    STRATEGY = "backbone"
    BINARY_MODEL_PATH = "embedding/binary_ood_dino_small_epoch3.pt"
    BINARY_DINO_SIZE = "small"
    VAL_CSV = "data/splits/val.csv"
    TEST_CSV = "data/splits/test.csv"

    print("--- 1. Loading Models ---")
    embedding_model = load_finetuned_embedding_model(CHECKPOINT_PATH, MODEL_TYPE, STRATEGY)
    classifier = FENClassifier(embedding_extractor=embedding_model)

    # Load Binary Model
    classifier.set_binary_model(BINARY_MODEL_PATH, dino_size=BINARY_DINO_SIZE)

    # Load Softmax Head
    head = load_classifier_head(CHECKPOINT_PATH, embedding_model.get_embedding_dim())
    classifier.set_classifier_head(head)

    print("--- 2. Building Database (for KNN/Mahalanobis) ---")
    val_df = pd.read_csv(VAL_CSV)
    board_ids = val_df['board_id'].unique()

    for board_id in tqdm(board_ids, desc="Indexing"):
        board_df = val_df[val_df['board_id'] == board_id]
        if len(board_df) != 64: continue

        # Prepare batch
        tile_images = []
        board_state = np.zeros((8, 8), dtype=int)
        for _, row in board_df.iterrows():
            img_path = Path(row['image'])
            if not img_path.is_absolute(): img_path = Path.cwd() / img_path
            with Image.open(img_path) as im:
                tile_images.append(im.convert('RGB').copy())
            r, c = parse_tile_coords(row['image'])
            board_state[r, c] = row['label']

        tile_embeddings = embedding_model.extract_batch_embeddings(tile_images)
        classifier.add_fen_position(board_id, tile_embeddings, board_state)

    classifier.update_thresholds()

    # --- 3. RUN ---
    run_benchmark_suite(classifier, TEST_CSV)


if __name__ == "__main__":
    main()