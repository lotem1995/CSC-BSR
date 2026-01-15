import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your corrected loaders
from preprocessing.load_dataset import get_train_dataloader, get_val_dataloader
from embedding.dinov2 import DINOv2Embedding
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


class BinaryDINOBackboneFineTuner:
    """
    Fine-tunes DINO-v2 backbone for Binary OOD Classification.
    Class 0: In-Distribution (Normal Chess Pieces)
    Class 1: Out-of-Distribution (Hands, Objects, etc.)
    """

    def __init__(self, dino_model: DINOv2Embedding):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dino = dino_model
        self.model = dino_model.model

        embedding_dim = dino_model.get_embedding_dim()

        # Binary Classifier Head (2 classes: ID vs OOD)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, 2),
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": 5e-6},  # Backbone (Slow)
                {"params": self.classifier.parameters(), "lr": 1e-4},  # Head (Fast)
            ],
            weight_decay=0.01
        )

        # Since get_train_dataloader balances the batch (50% OOD),
        # we treat classes equally (1.0).
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(self.device))

        print(f"Initialized Binary Tuner on {self.device}")

    def _get_binary_labels(self, raw_labels):
        """
        Map {0..16} -> 0 (ID)
        Map {17}    -> 1 (OOD)
        """
        # Create binary mask: 1 if label is 17, else 0
        binary_labels = (raw_labels == 17).long()
        return binary_labels.to(self.device)

    def train_batch(self, batch: Dict) -> float:
        self.model.train()
        self.classifier.train()

        # Image is already a tensor from your get_train_dataloader
        x = batch["image"].to(self.device)
        labels = self._get_binary_labels(batch["label"])

        # Forward Pass
        # DINO expects specific normalization, but with fine-tuning backbone + augmentation
        # standard 0-1 tensors often work well enough.
        # Ideally, we'd add DINO normalization in the dataset transform,
        # but the backbone will adapt to the current distribution.
        feats = self.model(x)
        logits = self.classifier(feats)

        loss = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate_batch(self, batch: Dict) -> Tuple[float, Dict]:
        self.model.eval()
        self.classifier.eval()

        x = batch["image"].to(self.device)
        labels = self._get_binary_labels(batch["label"])

        feats = self.model(x)
        logits = self.classifier(feats)
        loss = self.criterion(logits, labels).item()

        # Predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        return loss, {"y_true": y_true, "y_pred": y_pred}

    def save(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'classifier': self.classifier.state_dict(),
            'type': 'binary_ood'
        }, path)


def train_binary_ood(
        epochs: int = 5,
        batch_size: int = 32,  # Larger batch size since we use sampling
        dino_size: str = "small",
        num_workers: int = 2
):
    # 1. Initialize Model
    logger.info(f"Initializing DINOv2-{dino_size} for Binary OOD...")
    dino_model = DINOv2Embedding(model_size=dino_size)
    trainer = BinaryDINOBackboneFineTuner(dino_model)

    # 2. Data Loaders (Using your corrected loader)
    logger.info("Loading Datasets via get_train_dataloader...")
    train_loader = get_train_dataloader(batch_size=batch_size, num_workers=num_workers, resize=518)
    val_loader = get_val_dataloader(batch_size=batch_size, num_workers=num_workers)

    # 3. Training Loop
    logger.info("Starting Binary Training...")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # Train
        total_loss = 0
        train_batches = 0
        for batch in tqdm(train_loader, desc="Training"):
            loss = trainer.train_batch(batch)
            total_loss += loss
            train_batches += 1

        logger.info(f"  Avg Train Loss: {total_loss / train_batches:.4f}")

        # Validation
        val_loss = 0
        all_true = []
        all_pred = []

        for batch in tqdm(val_loader, desc="Validation"):
            loss, metrics = trainer.evaluate_batch(batch)
            val_loss += loss
            all_true.extend(metrics["y_true"])
            all_pred.extend(metrics["y_pred"])

        avg_val_loss = val_loss / len(val_loader)

        # Metrics
        y_true = np.array(all_true)
        y_pred = np.array(all_pred)

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # OOD Recall
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)  # OOD Precision

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  OOD Recall (Caught): {rec:.4f}")
        logger.info(f"  False Alarms: {cm[0, 1]}")

    # 4. Save
    output_path = f"embedding/binary_ood_dino_{dino_size}.pt"
    trainer.save(output_path)
    logger.info(f"✓ Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-size", default="small", choices=["small", "base"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    train_binary_ood(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dino_size=args.dino_size,
        num_workers=args.num_workers
    )