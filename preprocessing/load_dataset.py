import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
<<<<<<< HEAD
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
=======
>>>>>>> b5e13acf (pre processed data can be a ready-to-go dataset using the provided script)
import matplotlib.pyplot as plt 
from collections import Counter
splits_dir = Path("data/splits")
path_root = Path("data")  # stored in manifest as config.path_root; adjust if you move things

class ChessTilesCSV(Dataset):
    def __init__(self, csv_path, root, transform=None, use_embeddings=False):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.use_embeddings = use_embeddings

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row.image
        label = int(row.label)
        emb = row.embedding if isinstance(row.embedding, str) and row.embedding else None

        if self.use_embeddings and emb:
            features = torch.as_tensor(np.load(self.root / emb))
            image_tensor = features
        else:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                image_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0
<<<<<<< HEAD
        if self.transform:
            image_tensor = self.transform(img)
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

def get_train_dataloader(batch_size,num_workers):
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
        # we simply force the image
        # to the correct size for the model.
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

    # Calculate the weight for each class
    class_weights = 1.0 / class_counts  # no need to normalize it - WeightedRandomSampler does it for you
    class_weights_dict = class_weights.to_dict()
    sample_weights = [class_weights_dict[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    # Create the Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Essential: allows oversampling the minority class
    )

    # Create the DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Adjust as needed
        sampler=sampler,  # Pass the sampler here
        shuffle=False,  # CRITICAL: Shuffle must be False when using a any sampler - in our case the sampler already shuffle
        num_workers=num_workers  # Adjust based on your CPU
    )

    paint_camel()
    return train_loader


if __name__ == "__main__":
    # example of some pictures sampled from the new distribution after some transforms
    train_loader = get_train_dataloader(64,1)
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision


    # Function to un-normalize and display an image
    def imshow(tensor, title=None):
        # 1. Clone tensor so we don't modify the original
        img = tensor.clone().detach().cpu()

        # 4. Convert (C, H, W) -> (H, W, C) for Matplotlib
        img = img.permute(1, 2, 0).numpy()

        # 5. Plot
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.axis('off')


    # --- MAIN EXECUTION ---

    # 1. Get a single batch
    print("Fetching a batch...")
    batch = next(iter(train_loader))
    images = batch['image']  # Access by key (based on your dataset class)
    labels = batch['label']

    # 2. Create a grid of images
    # We'll show the first 16 images of the batch
    num_show = 16
    grid_img = torchvision.utils.make_grid(images[:num_show], nrow=4, padding=2)

    # 3. Plot the grid
    plt.figure(figsize=(10, 10))
    imshow(grid_img, title=f"Batch Sample (Labels: {labels[:num_show].tolist()})")
    plt.show()

    # 4. Print Distribution Check (Text)
    print(f"\nBatch Label Counts: {torch.bincount(labels)}")
=======
        return {"image": image_tensor, "label": label, "board_id": row.board_id, "path": str(img_path)}

train_ds = ChessTilesCSV(splits_dir/"train.csv", root=path_root)

# show train_ds as pandas dataframe
print(train_ds.df.head())
>>>>>>> b5e13acf (pre processed data can be a ready-to-go dataset using the provided script)
