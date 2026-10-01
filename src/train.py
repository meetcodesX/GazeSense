from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.gazefollow_dataset import GazeFollowDataset
from src.models.gaze_model import GazeModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_distance = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        head_bbox = batch["head_bbox"].to(device)
        eye_point = batch["eye_point"].to(device)
        targets = batch["gaze_target"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predictions = model(images, head_bbox, eye_point)
            loss = criterion(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        distances = torch.linalg.vector_norm(
            predictions - targets, dim=1
        )

        total_loss += loss.item() * batch_size
        total_distance += distances.sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    average_distance = total_distance / total_samples

    return average_loss, average_distance


def main():
    set_seed(42)

    ROOT = Path(__file__).resolve().parents[1]
    SPLIT_DIR = ROOT / "data" / "gazefollow" / "grouped_splits"

    OUTPUT_DIR = ROOT / "outputs" / "gazefollow_baseline"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = GazeFollowDataset(
        csv_path=SPLIT_DIR / "train.csv",
        project_root=ROOT,
    )

    val_dataset = GazeFollowDataset(
        csv_path=SPLIT_DIR / "val.csv",
        project_root=ROOT,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = GazeModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_distance = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_distance = run_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Train Mean Distance: {train_distance:.5f} | "
            f"Val Mean Distance: {val_distance:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = OUTPUT_DIR / "best_model.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint: {checkpoint_path}")

    print("Training completed.")
    print(f"Best validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    main()