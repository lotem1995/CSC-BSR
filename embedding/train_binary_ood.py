import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image  # Added for _to_batch
from tqdm import tqdm
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your loaders
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
        self.transform = dino_model.transform  # Store transform for _to_batch

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

        # Weighted Loss: OOD (Class 1) is rare (~7%), so we weight it heavily (8.0)
        ood_weight = 8.0
        weights = torch.tensor([1.0, ood_weight]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        print(f"Initialized Binary Tuner on {self.device} with OOD Weight: {ood_weight}")

    def _to_batch(self, image_tensors):
        """
        Robustly handles image resizing.
        Takes whatever the DataLoader gives (e.g., 224x224 Tensors),
        converts back to PIL, and applies DINO's specific transform (518x518).
        """
        images = []
        # 1. Convert Tensor back to PIL
        for img_tensor in image_tensors:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))

        # 2. Apply DINO's transform (Resizes to 518, Normalizes, etc.)
        batch_x = []
        for img in images:
            x = self.transform(img).unsqueeze(0)
            batch_x.append(x)

        # 3. Return correct tensor on device
        return torch.cat(batch_x, dim=0).to(self.device)

    def train_batch(self, batch: Dict) -> float:
        self.model.train()
        self.classifier.train()

        # [CHANGED] Use _to_batch to handle resizing automatically
        x = self._to_batch(batch["image"])
        labels = batch["is_ood"].long().to(self.device)

        # Forward Pass
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

        # [CHANGED] Use _to_batch here too
        x = self._to_batch(batch["image"])
        labels = batch["is_ood"].long().to(self.device)

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
        batch_size: int = 8,  # Kept small (8) because 518px images use a lot of VRAM
        dino_size: str = "small",
        num_workers: int = 2
):
    # 1. Initialize Model
    logger.info(f"Initializing DINOv2-{dino_size} for Binary OOD...")
    dino_model = DINOv2Embedding(model_size=dino_size)
    trainer = BinaryDINOBackboneFineTuner(dino_model)

    # 2. Data Loaders
    # NOTE: We can now use standard 224 resize in loader, saving memory/time
    # The trainer's _to_batch will handle the upscale to 518.
    logger.info("Loading Datasets via get_train_dataloader...")

    # Passing resize=224 is safe now!
    train_loader = get_train_dataloader(batch_size=batch_size, num_workers=num_workers, consider_ood_as_class=True)
    val_loader = get_val_dataloader(batch_size=batch_size, num_workers=num_workers, consider_ood_as_class=True)

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

        epoch_path = f"embedding/binary_ood_dino_{dino_size}_epoch{epoch + 1}.pt"
        trainer.save(epoch_path)
        logger.info(f"  -> Saved Checkpoint: {epoch_path}")

    # 4. Save
    output_path = f"embedding/binary_ood_dino_{dino_size}.pt"
    trainer.save(output_path)
    logger.info(f"✓ Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-size", default="small", choices=["small", "base"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    train_binary_ood(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dino_size=args.dino_size,
        num_workers=args.num_workers
    )