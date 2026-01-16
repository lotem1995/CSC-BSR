import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy

from CNN.chess_CNN import ChessPieceClassifier
# internal imports
from preprocessing.load_dataset import get_train_dataloader, get_val_dataloader

# from CNN_classifier import ChessPieceClassifier

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10  # Stop if val loss doesn't improve
NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def validate(model, val_loader, criterion):
    model.eval()  # Set to evaluation mode (turns off Dropout/BatchNorm updating)
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():  # No gradients needed for validation
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            val_loss += loss.item()
            val_acc += acc.item()

    avg_loss = val_loss / len(val_loader)
    avg_acc = val_acc / len(val_loader)
    return avg_loss, avg_acc


def train_with_validation(train_loader, val_loader):
    model = ChessPieceClassifier(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # 1. Weight Decay: Adds L2 Regularization (penalizes large weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 2. Scheduler: Reduce learning rate if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels).item()

            pbar.set_postfix(train_loss=train_loss / (batch_idx + 1))

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # --- VALIDATION PHASE ---
        avg_val_loss, avg_val_acc = validate(model, val_loader, criterion)

        # Step the scheduler
        scheduler.step(avg_val_loss)

        print(f"\n\tTrain Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2%}")
        print(f"\tVal Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.2%}")

        # --- EARLY STOPPING & CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_chess_classifier.pth")
            print("\t-> New Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"\t-> No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered!")
                break

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model


def main():
    print("--- Loading Data ---")
    train_loader = get_train_dataloader(batch_size=BATCH_SIZE)
    val_loader = get_val_dataloader(batch_size=BATCH_SIZE)

    print(f"Training on {len(train_loader)} batches.")
    print(f"Validating on {len(val_loader)} batches.")
    print(f"Device: {DEVICE}")

    # --- Run Training ---
    train_with_validation(train_loader, val_loader)

    print("\nProcess Complete. Best model weights loaded and saved to disk.")


if __name__ == "__main__":
    main()
