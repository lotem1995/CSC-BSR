import os
import re

from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class FilenameDataset(Dataset):
    def __init__(self, img_dir, transform=transforms.Compose([transforms.ToTensor()])):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")  # Ensure it's 3-channel

        label=self.get_class_label(img_name)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_label(self, img_name):
        # Search for 'class' followed by one or more digits (\d+)
        # The pattern finds 'class' and captures the digits after it
        match = re.search(r'class(\d+)', img_name)

        if match:
            # match.group(1) contains just the digits (e.g., "16" or "2")
            return int(match.group(1))
        else:
            raise ValueError(f"No class label found in filename: {img_name}")


if __name__ == '__main__':
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(parent_dir, 'preprocessed_data')
    print(data_dir)


    my_dataset = FilenameDataset(img_dir=data_dir)

    # Create the DataLoader
    train_loader = DataLoader(my_dataset, batch_size=32, shuffle=True)

    # Grab one batch to see if it works
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")


