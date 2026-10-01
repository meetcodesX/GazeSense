from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.gazefollow_dataset import GazeFollowDataset
from src.models.gaze_model import GazeModel


def main():
    ROOT = Path(__file__).resolve().parents[1]

    SPLIT_DIR = ROOT / "data" / "gazefollow" / "grouped_splits"
    CHECKPOINT = (
        ROOT / "outputs" / "gazefollow_baseline" / "best_model.pth"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Evaluation device:", device)

    test_dataset = GazeFollowDataset(
        csv_path=SPLIT_DIR / "test.csv",
        project_root=ROOT,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = GazeModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.MSELoss(reduction="sum")

    total_squared_error = 0.0
    total_distance = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            head_bbox = batch["head_bbox"].to(device)
            eye_point = batch["eye_point"].to(device)
            targets = batch["gaze_target"].to(device)

            predictions = model(images, head_bbox, eye_point)

            total_squared_error += criterion(
                predictions, targets
            ).item()

            distances = torch.linalg.vector_norm(
                predictions - targets, dim=1
            )

            total_distance += distances.sum().item()
            total_samples += images.size(0)

    test_mse = total_squared_error / (total_samples * 2)
    test_mean_distance = total_distance / total_samples

    print("\n--- TEST RESULTS ---")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test samples:", total_samples)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test Mean Euclidean Distance: {test_mean_distance:.6f}")


if __name__ == "__main__":
    main()