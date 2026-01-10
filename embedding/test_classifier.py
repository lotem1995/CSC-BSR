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


def evaluate_on_test_csv(
    classifier: FENClassifier,
    test_csv_path: str,
    data_root: str = "data",
    method: str = "mahalanobis",
    ood_output_dir: str = "ood_failures"
):
    """
    Evaluate the classifier on test.csv.
    
    Args:
        classifier: Trained FENClassifier instance
        test_csv_path: Path to test.csv
        data_root: Root directory for image paths
        method: "knn" or "mahalanobis"
    """
    print(f"\nEvaluating on {test_csv_path}")
    print(f"Method: {method}")

    if os.path.exists(ood_output_dir):
        shutil.rmtree(ood_output_dir)  # Delete old run
    os.makedirs(ood_output_dir)  # Create new folder
    print(f"Saving OOD images to: {ood_output_dir}/")
    
    # Load test CSV
    df = pd.read_csv(test_csv_path)
    
    # Group by board_id to evaluate whole boards
    board_ids = df['board_id'].unique()
    
    total_tiles = 0
    correct_tiles = 0
    ood_count = 0
    
    for board_id in tqdm(board_ids, desc="Evaluating boards"):
        # Get all 64 tiles for this board
        board_df = df[df['board_id'] == board_id].copy()
        
        if len(board_df) != 64:
            print(f"Warning: Board {board_id} has {len(board_df)} tiles, skipping")
            continue
        
        # Parse tile coordinates and sort by (row, col) to ensure correct ordering
        board_df['tile_coords'] = board_df['image'].apply(parse_tile_coords)
        board_df = board_df.sort_values('tile_coords')
        
        # Load tile images in correct order
        tile_images = []
        true_labels = []
        for _, row in board_df.iterrows():
            # CSV paths are relative to project root (e.g., 'preprocessed_data/...')
            img_path = Path(row['image'])
            if not img_path.is_absolute():
                # Resolve relative to current directory (project root)
                img_path = Path.cwd() / img_path
            
            with Image.open(img_path) as im:
                tile_images.append(im.convert('RGB').copy())
            true_labels.append(row['label'])
        
        true_labels = np.array(true_labels)
        
        # Extract embeddings
        tile_embeddings = classifier.embedding_extractor.extract_batch_embeddings(tile_images)
        
        # # Predict
        # if method == "knn":
        #     predictions, confidences = classifier.predict_knn(tile_embeddings, k=5)
        # else:
        #     predictions, confidences = classifier.predict_mahalanobis(tile_embeddings)
        predictions, confidences, is_ood = classifier.predict_with_ood(
            tile_embeddings,
            method=method
        )

        # Compute accuracy for this board
        correct = (predictions == true_labels).sum()
        total_tiles += 64
        correct_tiles += correct
        ood_count += is_ood.sum()

        if np.any(is_ood):
            # Get the indices (0-63) of the OOD tiles
            ood_indices = np.where(is_ood)[0]

            for idx in ood_indices:
                # Calculate row/col for filename
                row, col = divmod(idx, 8)
                pred_cls = predictions[idx]
                true_cls = true_labels[idx]

                # Create a helpful filename:
                # boardID_tileA1_True(Pawn)_Pred(Empty).png
                tile_name = f"{board_id}_tile{row}{col}_True{true_cls}_Pred{pred_cls}.png"
                save_path = os.path.join(ood_output_dir, tile_name)

                # Save the image
                # (We resize to 224x224 so it's big enough to see clearly)
                img_to_save = tile_images[idx].resize((224, 224))
                img_to_save.save(save_path)
    
    # Overall accuracy
    accuracy = correct_tiles / total_tiles if total_tiles > 0 else 0
    ood_rate = ood_count / total_tiles if total_tiles > 0 else 0
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Correct: {correct_tiles}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  OOD Flags: {ood_count} ({ood_rate * 100:.2f}%)")
    print(f"  Check '{ood_output_dir}' to see the confused images!")
    print(f"{'='*60}")
    
    return accuracy


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
    classifier_path = "classifier.json"

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
        method="mahalanobis",  # or "knn"
        ood_output_dir = "ood_inspection_images"
    )
    
    print(f"\n✓ Done! Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()