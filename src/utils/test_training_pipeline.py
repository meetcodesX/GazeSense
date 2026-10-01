from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.gazefollow_dataset import GazeFollowDataset
from src.models.gaze_model import GazeModel


def main():
    ROOT = Path(__file__).resolve().parents[2]

    # Based on your VS Code screenshot.
    SPLIT_DIR = ROOT / "data" / "gazefollow" / "grouped_splits"

    train_csv = SPLIT_DIR / "train.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Could not find {train_csv}. "
            "Update SPLIT_DIR to the folder containing your fresh split CSVs."
        )

    dataset = GazeFollowDataset(
        csv_path=train_csv,
        project_root=ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GazeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Load one real batch
    batch = next(iter(loader))

    images = batch["image"].to(device)
    head_bbox = batch["head_bbox"].to(device)
    eye_point = batch["eye_point"].to(device)
    targets = batch["gaze_target"].to(device)

    # Forward pass
    model.train()
    predictions = model(images, head_bbox, eye_point)

    # Confirm prediction and target dimensions match
    print("Image batch:", images.shape)
    print("Head bbox batch:", head_bbox.shape)
    print("Eye point batch:", eye_point.shape)
    print("Target batch:", targets.shape)
    print("Prediction batch:", predictions.shape)

    assert predictions.shape == targets.shape, (
        f"Shape mismatch: predictions {predictions.shape}, "
        f"targets {targets.shape}"
    )

    # Loss and backward pass
    loss = criterion(predictions, targets)

    optimizer.zero_grad()
    loss.backward()

    # Check that gradients were calculated
    has_gradients = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert has_gradients, "No gradients were calculated."

    # Update model parameters once
    optimizer.step()

    print(f"Smoke-test loss: {loss.item():.6f}")
    print("Gradient check: passed")
    print("Optimizer step: passed")
    print("Training pipeline smoke test: SUCCESS")


if __name__ == "__main__":
    main()