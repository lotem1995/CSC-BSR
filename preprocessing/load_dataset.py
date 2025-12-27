import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
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
        return {"image": image_tensor, "label": label, "board_id": row.board_id, "path": str(img_path)}

train_ds = ChessTilesCSV(splits_dir/"train.csv", root=path_root)

# show train_ds as pandas dataframe
print(train_ds.df.head())