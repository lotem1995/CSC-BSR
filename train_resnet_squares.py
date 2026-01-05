import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from preprocessing.load_dataset import ChessTilesCSV
from simple_models_attempt import ResNet18SquareClassifier


# ----- label remap -----
def remap_label(y: int) -> int:
    # Original labels: 0, 1..6, 11..16  ->  Needed: 0..12
    if y == 0:
        return 0
    if 1 <= y <= 6:
        return y
    if 11 <= y <= 16:
        return 7 + (y - 11)
    raise ValueError(f"Bad label: {y}")


# ImageNet normalization (required for pretrained ResNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def preprocess(x: torch.Tensor, img_size: int = 96) -> torch.Tensor:
    """
    x: (B,3,H,W) float tensor in [0,1]
    returns: resized + normalized tensor
    """
    x = F.resize(x, [img_size, img_size])
    x = F.normalize(x, mean=MEAN, std=STD)
    return x


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, out_dir="runs"):
    # Loss plot
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss vs Epoch")
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(train_accs, label="train acc")
    plt.plot(val_accs, label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy vs Epoch")
    plt.savefig(os.path.join(out_dir, "accuracy.png"))
    plt.close()


def train_one_epoch(model, loader, optimizer, loss_fn, device, img_size=96):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        x = batch["image"].to(device)
        y = torch.tensor([remap_label(int(v)) for v in batch["label"]], device=device)

        x = preprocess(x, img_size=img_size)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device, img_size=96):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        x = batch["image"].to(device)
        y = torch.tensor([remap_label(int(v)) for v in batch["label"]], device=device)

        x = preprocess(x, img_size=img_size)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    # ---- simple settings ----
    TRAIN_CSV = "data/splits/train.csv"
    VAL_CSV   = "data/splits/val.csv"
    ROOT      = "data"

    EPOCHS = 15
    BATCH  = 128
    LR     = 3e-4
    IMG    = 96
    NUM_WORKERS = 2
    # -------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    os.makedirs("runs", exist_ok=True)

    # datasets + loaders
    train_ds = ChessTilesCSV(TRAIN_CSV, root=ROOT)
    val_ds   = ChessTilesCSV(VAL_CSV,   root=ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device == "cuda"))

    # model
    model = ResNet18SquareClassifier(
        num_classes=13,
        pretrained=True,
        dropout=0.2,
        freeze_backbone=(device == "cpu")
    ).to(device)

    # loss + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, img_size=IMG)
        va_loss, va_acc = eval_one_epoch(model, val_loader, loss_fn, device, img_size=IMG)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), "runs/best_model.pt")
            print("saved runs/best_model.pt")

    # plots
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, out_dir="runs")
    print("saved plots: runs/loss.png and runs/accuracy.png")
    print("best val acc =", best_acc)


if __name__ == "__main__":
    main()
