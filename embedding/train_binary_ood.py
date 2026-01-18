import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
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

    def __init__(self, dino_model: DINOv2Embedding, ood_lookup: Dict[str, int]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dino = dino_model
        self.model = dino_model.model
        self.transform = dino_model.transform

        # We use this to fix the labels without touching load_dataset.py
        self.ood_lookup = ood_lookup

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
                {"params": self.model.parameters(), "lr": 5e-6},
                {"params": self.classifier.parameters(), "lr": 1e-4},
            ],
            weight_decay=0.01
        )

        # Weighted Loss
        ood_weight = 8.0
        weights = torch.tensor([1.0, ood_weight]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        print(f"Initialized Binary Tuner on {self.device} with OOD Weight: {ood_weight}")

    def _to_batch(self, image_tensors):
        """
        Takes 224x224 tensors from loader, converts to PIL,
        then applies DINO's 518x518 transform.
        """
        images = []
        for img_tensor in image_tensors:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))

        batch_x = []
        for img in images:
            x = self.transform(img).unsqueeze(0)
            batch_x.append(x)

        return torch.cat(batch_x, dim=0).to(self.device)

    def _get_binary_labels_from_paths(self, paths):
        """
        Ignores the incoming label and looks up the TRUE OOD status via file path.
        """
        clean_labels = []
        for p in paths:
            # Default to 0 (ID) if path not found, but it should be found
            label = self.ood_lookup.get(p, 0)
            clean_labels.append(label)

        return torch.tensor(clean_labels, dtype=torch.long).to(self.device)

    def train_batch(self, batch: Dict) -> float:
        self.model.train()
        self.classifier.train()

        x = self._to_batch(batch["image"])

        # [CHANGED] Look up label by path instead of trusting batch['label']
        labels = self._get_binary_labels_from_paths(batch["path"])

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

        x = self._to_batch(batch["image"])

        # [CHANGED] Look up label by path
        labels = self._get_binary_labels_from_paths(batch["path"])

        feats = self.model(x)
        logits = self.classifier(feats)
        loss = self.criterion(logits, labels).item()

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


def build_ood_lookup(dataset):
    """
    Scans the dataset's internal DataFrame to map absolute file paths to OOD status.
    This bypasses the logic in __getitem__.
    """
    lookup = {}
    print(f"Building OOD lookup for {len(dataset)} items...")

    # Access the dataframe and root directly from the dataset object
    # (Assuming Dataset class has .df and .root as per your code)
    root = dataset.root

    for _, row in dataset.df.iterrows():
        # Reconstruct the exact path string that __getitem__ returns
        full_path = str(root / row['image'])

        # Robustly check is_ood (handle bool, int, or string "True")
        val = row.get('is_ood', False)
        if isinstance(val, str):
            is_ood = val.lower() == 'true' or val == '1'
        else:
            is_ood = bool(val)

        lookup[full_path] = 1 if is_ood else 0

    return lookup


def train_binary_ood(
        epochs: int = 5,
        batch_size: int = 8,
        dino_size: str = "small",
        num_workers: int = 2
):
    # 1. Initialize Loaders first to get access to the CSV data
    logger.info("Loading Datasets...")
    train_loader = get_train_dataloader(batch_size=batch_size, num_workers=num_workers)
    val_loader = get_val_dataloader(batch_size=batch_size, num_workers=num_workers)

    # 2. Build the Global OOD Lookup Map
    # We combine train and val lookups into one giant dictionary
    train_lookup = build_ood_lookup(train_loader.dataset)
    val_lookup = build_ood_lookup(val_loader.dataset)
    global_lookup = {**train_lookup, **val_lookup}

    # 3. Initialize Model with the lookup
    logger.info(f"Initializing DINOv2-{dino_size} for Binary OOD...")
    dino_model = DINOv2Embedding(model_size=dino_size)

    # Pass the lookup to the trainer
    trainer = BinaryDINOBackboneFineTuner(dino_model, ood_lookup=global_lookup)

    # 4. Training Loop
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
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # Safe access to confusion matrix elements
        false_alarms = cm[0, 1] if cm.shape == (2, 2) else 0

        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  OOD Recall (Caught): {rec:.4f}")
        logger.info(f"  False Alarms: {false_alarms}")

        epoch_path = f"embedding/binary_ood_dino_{dino_size}_epoch{epoch + 1}.pt"
        trainer.save(epoch_path)
        logger.info(f"  -> Saved Checkpoint: {epoch_path}")

    # 5. Save Final
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