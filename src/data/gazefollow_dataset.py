from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GazeFollowDataset(Dataset):
    def __init__(self, csv_path, project_root, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.project_root = Path(project_root)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.project_root / row["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Normalized head bounding box
        head_bbox = torch.tensor([
            row["head_xmin"],
            row["head_ymin"],
            row["head_xmax"],
            row["head_ymax"],
        ], dtype=torch.float32)

        # Normalized eye position
        eye_point = torch.tensor([
            row["eye_x"],
            row["eye_y"],
        ], dtype=torch.float32)

        # Normalized gaze target
        gaze_target = torch.tensor([
            row["gaze_x"],
            row["gaze_y"],
        ], dtype=torch.float32)

        return {
            "image": image,
            "head_bbox": head_bbox,
            "eye_point": eye_point,
            "gaze_target": gaze_target,
            "image_id": int(row["image_id"]),
        }


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    SPLIT_DIR = ROOT / "data" / "gazefollow" / "grouped_splits"

    train_dataset = GazeFollowDataset(
        csv_path=SPLIT_DIR / "train.csv",
        project_root=ROOT,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    print("Dataset size:", len(train_dataset))

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")