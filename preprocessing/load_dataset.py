import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from torchvision import transforms

# Adjust paths if your project structure is different
splits_dir = Path("data/splits")
path_root = Path(".")  # stored in manifest as config.path_root; adjust if you move things


class ChessTilesCSV(Dataset):
    def __init__(self, csv_path, root, transform=None, use_embeddings=False):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.use_embeddings = use_embeddings

        self.label_map = {
            # Empty Square
            0: 0,
            # White Pieces (1-6) -> Classes 1-6
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
            # Black Pieces (11-16) -> Classes 7-12
            11: 7, 12: 8, 13: 9, 14: 10, 15: 11, 16: 12,
            # OOD (Class 17) -> Class 17 [FIXED: Added this line]
            17: 17
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row.image
        raw_label = int(row.label)

        # Default to 0 if label not found (safety)
        label = self.label_map.get(raw_label, 0)

        emb = row.embedding if isinstance(row.embedding, str) and row.embedding else None

        # CASE A: Using Pre-calculated Embeddings
        if self.use_embeddings and emb:
            features = torch.as_tensor(np.load(self.root / emb))
            # Note: Transforms are usually not applied to embeddings
            return {"image": features, "label": label, "board_id": row.board_id, "path": str(img_path)}

        # CASE B: Loading Images
        else:
            with Image.open(img_path) as img:
                img = img.convert("RGB")

                # [FIXED] Apply transforms here, while 'img' exists
                if self.transform:
                    image_tensor = self.transform(img)
                else:
                    # Default if no transform provided
                    image_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            return {"image": image_tensor, "label": label, "board_id": row.board_id, "path": str(img_path)}


def paint_camel():
    # We use 'r' before the string to denote a raw string
    # This prevents python from interpreting backslashes as special characters
    art = r"""
           \ | /
         '-.;;;.-'
        -==;;;;;==-   (The Blazing Sun)
         .-';;;'-.
           / | \

              //
            _oo\
           (__/ \  _  _
              \  \/ \/ \
              (         )    (The Ship of the Desert)
               \_______/
               //     \\
              //       \\
    ~^~^~^~^~^~^~^~^~^~^~^~^~
         (The Hot Sand)
    """
    print(art)


def get_train_dataloader(batch_size, num_workers):
    # --- CONFIGURATION ---
    rotation_jitter = 5  # Change this number to increase/decrease the "wiggle"

    # --- TRANSFORM DEFINITION ---
    jittered_rotation = transforms.RandomChoice([
        transforms.RandomRotation(degrees=(-rotation_jitter, rotation_jitter)),
        transforms.RandomRotation(degrees=(90 - rotation_jitter, 90 + rotation_jitter)),
        transforms.RandomRotation(degrees=(180 - rotation_jitter, 180 + rotation_jitter)),
        transforms.RandomRotation(degrees=(270 - rotation_jitter, 270 + rotation_jitter)),
    ])
    train_transforms = transforms.Compose([
        # we simply force the image to the correct size for the model.
        transforms.Resize((224, 224)),
        # 1. Geometric Flips
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # 2. Rotations (Discrete 90s + Jitter)
        jittered_rotation,
        # 3. Color/Light
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        # 4. Finalize
        transforms.ToTensor()
    ])

    # 1. Instantiate your dataset
    dataset = ChessTilesCSV(splits_dir / "train.csv", root=path_root, transform=train_transforms)
    labels = dataset.df['label'].values

    class_counts = dataset.df['label'].value_counts().sort_index()

    # Calculate weight per class
    class_weights = 1.0 / class_counts
    class_weights_dict = class_weights.to_dict()

    # Map weights to each sample
    sample_weights = [class_weights_dict.get(label, 0) for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    # Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create the Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create the DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers
    )

    paint_camel()
    return train_loader


def get_val_dataloader(batch_size=64, num_workers=4):

    # Simple transform for validation (Resize + ToTensor)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Pass the clean transforms to the dataset
    val_dataset = ChessTilesCSV(splits_dir / "val.csv", root=path_root, transform=val_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return val_loader