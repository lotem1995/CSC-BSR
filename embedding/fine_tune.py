"""
Fine-tuning Vision Models on Chess Data

This module fine-tunes vision models specifically for chess tiles.
Makes the models chess-aware instead of generic.

Supports multiple embedding models:
- QwenVisionEmbedding (Qwen3-VL)
- DINOv2Embedding
- Custom models implementing EmbeddingModel interface

Updated to work with pre-built dataset from build_dataset.py
"""

import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from pathlib import Path

from preprocessing.load_dataset import ChessTilesCSV
from sklearn.metrics import balanced_accuracy_score, f1_score

from embedding_base import EmbeddingModel
from dinov2 import DINOv2Embedding


def _is_qwen_embedding_model(model: Optional[EmbeddingModel]) -> bool:
    return model is not None and model.__class__.__name__ == "QwenVisionEmbedding"


def _is_dino_embedding_model(model: Optional[EmbeddingModel]) -> bool:
    return model is not None and model.__class__.__name__ == "DINOv2Embedding"


def _load_qwen_embedding_class():
    # Lazy import: avoids importing transformers unless Qwen is requested.
    from main import QwenVisionEmbedding
    return QwenVisionEmbedding

# Piece label mapping
PIECE_LABELS = {
    0: "empty",
    1: "white_pawn", 11: "black_pawn",
    2: "white_knight", 12: "black_knight",
    3: "white_bishop", 13: "black_bishop",
    4: "white_rook", 14: "black_rook",
    5: "white_queen", 15: "black_queen",
    6: "white_king", 16: "black_king",
}

PIECE_TO_ID = {v: k for k, v in PIECE_LABELS.items()}

# The dataset labels are stored as raw piece IDs (0..16 with gaps),
# but the classifier predicts 13 contiguous classes.
_RAW_LABELS_IN_ORDER = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
_LABEL_LUT = torch.full((17,), -1, dtype=torch.long)
for _class_idx, _raw_id in enumerate(_RAW_LABELS_IN_ORDER):
    _LABEL_LUT[_raw_id] = _class_idx


def _remap_raw_piece_labels(raw_labels: torch.Tensor) -> torch.Tensor:
    """Map raw piece IDs to contiguous class indices for CrossEntropyLoss."""
    if raw_labels.dtype != torch.long:
        raw_labels = raw_labels.long()
    if raw_labels.numel() == 0:
        return raw_labels
    if raw_labels.min().item() < 0 or raw_labels.max().item() >= _LABEL_LUT.numel():
        raise ValueError(
            f"Found label outside expected range [0, 16]: min={raw_labels.min().item()}, max={raw_labels.max().item()}"
        )
    lut = _LABEL_LUT.to(device=raw_labels.device)
    mapped = lut[raw_labels]
    if (mapped < 0).any():
        bad = raw_labels[mapped < 0].unique().tolist()
        raise ValueError(f"Found unsupported raw labels: {bad}. Expected one of {_RAW_LABELS_IN_ORDER}.")
    return mapped


class FineTuner:
    """
    Fine-tunes any EmbeddingModel on chess tile classification.
    
    Works with QwenVisionEmbedding, DINOv2Embedding, or any custom embedding model.
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize fine-tuner with a specific embedding model.
        
        Args:
            embedding_model: Any EmbeddingModel instance. If None, uses QwenVisionEmbedding.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use provided embedding model or default to Qwen
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            print("Initializing default QwenVisionEmbedding...")
            self.embedding_model = _load_qwen_embedding_class()()
        
        embedding_dim = self.embedding_model.get_embedding_dim()
        print(f"Using {self.embedding_model} for fine-tuning")
        
        # Classification head: embedding_dim -> 13 piece classes
        # Adaptive to different embedding dimensions
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 13),
        ).to(self.device)
        
        # Optimizer for fine-tuning
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        self.criterion = nn.CrossEntropyLoss()
        print(f"Fine-tuner initialized on device: {self.device}")
    
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract embedding using the embedding model"""
        embedding = self.embedding_model.extract_embedding(image).to(self.device)
        target_dtype = next(self.classifier.parameters()).dtype
        if embedding.dtype != target_dtype:
            embedding = embedding.to(dtype=target_dtype)
        return embedding
    
    def train_batch(self, batch: Dict) -> float:
        """
        Fine-tune on a batch of chess tiles.
        
        Args:
            batch: Dict with keys "image" (tensor), "label" (tensor), "board_id", "path"
            
        Returns:
            Loss value
        """
        self.classifier.train()
        
        # Get image tensors and labels from batch
        image_tensors = batch["image"]  # Shape: [batch_size, 3, H, W]
        labels = batch["label"]  # Shape: [batch_size]
        
        # Convert tensors to PIL images
        images = []
        for img_tensor in image_tensors:
            # Convert from [C, H, W] to [H, W, C] and to numpy
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
        
        # Extract embeddings using the embedding model
        embeddings = self.embedding_model.extract_batch_embeddings(images)
        embeddings = embeddings.to(self.device)
        target_dtype = next(self.classifier.parameters()).dtype
        if embeddings.dtype != target_dtype:
            embeddings = embeddings.to(dtype=target_dtype)
        
        # Classify
        labels = labels.to(self.device)
        labels = _remap_raw_piece_labels(labels)
        logits = self.classifier(embeddings)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_batch(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Evaluate on a batch with balanced accuracy metrics.
        
        Args:
            batch: Dict with keys "image" (tensor), "label" (tensor), "board_id", "path"
        
        Returns:
            (loss, balanced_accuracy, f1_score)
        """
        self.classifier.eval()
        
        with torch.no_grad():
            # Get image tensors and labels from batch
            image_tensors = batch["image"]
            labels = batch["label"]
            
            # Convert tensors to PIL images
            images = []
            for img_tensor in image_tensors:
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                images.append(Image.fromarray(img_np))
            
            # Extract embeddings using the embedding model
            embeddings = self.embedding_model.extract_batch_embeddings(images)
            embeddings = embeddings.to(self.device)
            target_dtype = next(self.classifier.parameters()).dtype
            if embeddings.dtype != target_dtype:
                embeddings = embeddings.to(dtype=target_dtype)
            
            # Classify
            labels = labels.to(self.device)
            labels = _remap_raw_piece_labels(labels)
            logits = self.classifier(embeddings)
            loss = self.criterion(logits, labels)
            
            # Predictions for metrics
            preds = logits.argmax(dim=1)
            
            # Convert to numpy for sklearn metrics
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            # Balanced accuracy and F1 (better for imbalanced data)
            balanced_acc = balanced_accuracy_score(labels_np, preds_np)
            f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        
        return loss.item(), balanced_acc, f1
    
    def save(self, path: str):
        """Save fine-tuned classifier head"""
        torch.save({
            'classifier': self.classifier.state_dict(),
            'piece_labels': PIECE_LABELS,
            'embedding_model_name': self.embedding_model.__class__.__name__,
        }, path)
    
    def load(self, path: str):
        """Load fine-tuned classifier head"""
        data = torch.load(path)
        self.classifier.load_state_dict(data['classifier'])


class QwenLoRAClassifierTrainer:
    """LoRA fine-tuning for Qwen visual tower + classifier head with metrics."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Lazy import: avoids importing transformers/peft unless LoRA is requested.
        from lorafinetune import QwenLoRAFineTuner
        self.qwen = QwenLoRAFineTuner()
        # Qwen3-VL vision embedding dimension
        embedding_dim = 2048
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 13),
        ).to(self.device)
        # Optimizer over LoRA trainables + classifier
        trainable_params = [p for p in self.qwen.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            list(self.classifier.parameters()) + trainable_params,
            lr=1e-4,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()

    def _batch_embeddings(self, images):
        # Process each image individually for correctness
        embs = []
        for image in images:
            inputs = self.qwen.processor(text=[""], images=[image], return_tensors="pt").to(self.device)
            grid_thw = inputs.get('image_grid_thw')
            visual_output = self.qwen.model.visual(inputs.pixel_values, grid_thw=grid_thw)
            visual_features = visual_output[0] if isinstance(visual_output, tuple) else visual_output
            emb = torch.mean(visual_features, dim=0, keepdim=True)
            embs.append(emb)
        return torch.cat(embs, dim=0)

    def train_batch(self, batch: Dict) -> float:
        self.qwen.model.train()
        self.classifier.train()
        image_tensors = batch["image"]
        labels = batch["label"].to(self.device)
        labels = _remap_raw_piece_labels(labels)
        # Convert tensors to PIL
        images = []
        for img_tensor in image_tensors:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
        embeddings = self._batch_embeddings(images)
        target_dtype = next(self.classifier.parameters()).dtype
        if embeddings.dtype != target_dtype:
            embeddings = embeddings.to(dtype=target_dtype)
        logits = self.classifier(embeddings)
        loss = self.criterion(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate_batch(self, batch: Dict) -> Tuple[float, float, float]:
        self.qwen.model.eval()
        self.classifier.eval()
        image_tensors = batch["image"]
        labels = batch["label"].to(self.device)
        labels = _remap_raw_piece_labels(labels)
        images = []
        for img_tensor in image_tensors:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
        embeddings = self._batch_embeddings(images)
        target_dtype = next(self.classifier.parameters()).dtype
        if embeddings.dtype != target_dtype:
            embeddings = embeddings.to(dtype=target_dtype)
        logits = self.classifier(embeddings)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = logits.argmax(dim=1)
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        balanced_acc = balanced_accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        return loss.item(), balanced_acc, f1


class DINOBackboneFineTuner:
    """Finetune DINO-v2 backbone together with classifier head."""
    def __init__(self, dino: DINOv2Embedding):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dino = dino
        self.model = dino.model
        self.transform = dino.transform
        embedding_dim = dino.get_embedding_dim()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 13),
        ).to(self.device)
        # Two param groups: smaller LR for backbone
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": 5e-6},
                {"params": self.classifier.parameters(), "lr": 1e-4},
            ],
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()

    def _to_batch(self, image_tensors):
        images = []
        for img_tensor in image_tensors:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
        batch_x = []
        for img in images:
            x = self.transform(img).unsqueeze(0)
            batch_x.append(x)
        return torch.cat(batch_x, dim=0).to(self.device)

    def train_batch(self, batch: Dict) -> float:
        self.model.train()
        self.classifier.train()
        x = self._to_batch(batch["image"])
        labels = batch["label"].to(self.device)
        labels = _remap_raw_piece_labels(labels)
        feats = self.model(x)
        logits = self.classifier(feats)
        loss = self.criterion(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate_batch(self, batch: Dict) -> Tuple[float, float, float]:
        self.model.eval()
        self.classifier.eval()
        x = self._to_batch(batch["image"])
        labels = batch["label"].to(self.device)
        labels = _remap_raw_piece_labels(labels)
        feats = self.model(x)
        logits = self.classifier(feats)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = logits.argmax(dim=1)
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        balanced_acc = balanced_accuracy_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        return loss.item(), balanced_acc, f1


# Backward compatibility alias
QwenFineTuner = FineTuner


def train_fine_tuning(
    splits_dir: str = "data/splits",
    embedding_model: Optional[EmbeddingModel] = None,
    path_root: str = "data",
    epochs: int = 3,
    batch_size: int = 4,
    use_val: bool = True,
    num_workers: int = 0,
    strategy: str = "head-only"
):
    """
    Main training loop for fine-tuning using pre-built dataset.
    
    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv
        embedding_model: EmbeddingModel to use. If None, uses QwenVisionEmbedding.
        path_root: Root directory for image paths (as stored in manifest)
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_val: Whether to evaluate on validation set after each epoch
        num_workers: DataLoader worker count (tune for cluster CPUs)
    """
    print("=" * 80)
    if embedding_model is None:
        print("FINE-TUNING QWEN ON CHESS DATA")
        embedding_model = _load_qwen_embedding_class()()
    else:
        print(f"FINE-TUNING {embedding_model} ON CHESS DATA")
    print("=" * 80)
    
    # Load datasets using ChessTilesCSV
    splits_dir_path = Path(splits_dir)
    path_root_path = Path(path_root)
    
    train_csv = splits_dir_path / "train.csv"
    val_csv = splits_dir_path / "val.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {train_csv}. "
            "Please run build_dataset.py first to create the dataset."
        )
    
    print(f"Loading training data from {train_csv}")
    train_dataset = ChessTilesCSV(
        csv_path=str(train_csv),
        root=str(path_root_path),
        transform=None,
        use_embeddings=False
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Optional validation dataset
    val_loader = None
    if use_val and val_csv.exists():
        print(f"Loading validation data from {val_csv}")
        val_dataset = ChessTilesCSV(
            csv_path=str(val_csv),
            root=str(path_root_path),
            transform=None,
            use_embeddings=False
        )
        print(f"Validation dataset size: {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    # Initialize fine-tuner per strategy
    if strategy == "lora":
        if embedding_model is not None and not _is_qwen_embedding_model(embedding_model):
            print("Warning: --strategy lora is only supported with Qwen; overriding to Qwen.")
        fine_tuner = QwenLoRAClassifierTrainer()
    elif strategy == "backbone":
        if _is_dino_embedding_model(embedding_model):
            fine_tuner = DINOBackboneFineTuner(embedding_model)
        else:
            print("Warning: --strategy backbone is only supported with DINO; falling back to head-only.")
            fine_tuner = FineTuner(embedding_model=embedding_model)
    else:
        fine_tuner = FineTuner(embedding_model=embedding_model)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 80)
        
        # Training phase
        fine_tuner.classifier.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            loss = fine_tuner.train_batch(batch)
            total_loss += loss
            num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1}: Loss = {loss:.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch+1} Training - Avg Loss: {avg_loss:.4f}")
        
        # Validation phase
        if val_loader is not None:
            # Model eval handled inside evaluate_batch as needed
            
            val_loss = 0
            val_balanced_acc = 0
            val_f1 = 0
            val_batches = 0
            
            for batch in tqdm(val_loader, desc="Validation"):
                loss, balanced_acc, f1 = fine_tuner.evaluate_batch(batch)
                val_loss += loss
                val_balanced_acc += balanced_acc
                val_f1 += f1
                val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)
            avg_val_balanced_acc = val_balanced_acc / max(val_batches, 1)
            avg_val_f1 = val_f1 / max(val_batches, 1)
            print(f"Epoch {epoch+1} Validation:")
            print(f"  Loss: {avg_val_loss:.4f}")
            print(f"  Balanced Accuracy: {avg_val_balanced_acc:.4f}")
            print(f"  F1 Score (weighted): {avg_val_f1:.4f}")
        else:
            fine_tuner.classifier.eval()
    
    # Save fine-tuned model
    output_path = str(Path(__file__).resolve().parent / "chess_finetuned.pt")
    fine_tuner.save(output_path)
    print(f"\n✓ Fine-tuned model saved to: {output_path}")
    
    return fine_tuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune embedding models on chess tiles (cluster-friendly)")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory with train/val/test CSV splits")
    parser.add_argument("--path-root", default="data", help="Root folder for image paths in CSVs")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers (tune to cluster CPUs)")
    parser.add_argument("--embedding-model", choices=["qwen", "dino-small", "dino-base"], default="qwen",
                        help="Embedding backbone to use")
    parser.add_argument("--no-val", action="store_true", help="Skip validation loop")
    parser.add_argument("--strategy", choices=["head-only", "lora", "backbone"], default="head-only",
                        help="Training strategy: classifier only, Qwen LoRA, or finetune backbone (DINO only)")
    args = parser.parse_args()

    embedding_model = None
    if args.strategy == "lora":
        # LoRA path uses its own trainer, embedding_model not needed
        embedding_model = None
    else:
        if args.embedding_model == "qwen":
            embedding_model = _load_qwen_embedding_class()()
        elif args.embedding_model == "dino-small":
            embedding_model = DINOv2Embedding(model_size="small")
        elif args.embedding_model == "dino-base":
            embedding_model = DINOv2Embedding(model_size="base")

    train_fine_tuning(
        splits_dir=args.splits_dir,
        embedding_model=embedding_model,
        path_root=args.path_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_val=not args.no_val,
        num_workers=args.num_workers,
        strategy=args.strategy
    )

