"""GazeFollow dataset with a head crop, a head mask and a gaussian target heatmap.

Returns, per annotation row:
  scene       (3, S, S)   ImageNet-normalized full image
  head_mask   (1, S, S)   binary box mask of the head location
  head_crop   (3, C, C)   ImageNet-normalized crop around the head
  gaze_target (2,)        normalized (x, y) in [0, 1]
  heatmap     (1, H, H)   gaussian target (sums to 1), H = heatmap_size
  image_id, head_bbox     (for min-distance grouping at eval time)

Training augmentation (train=True): horizontal flip (labels flipped too)
and colour jitter on the full image.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_gaussian_heatmap(x, y, size, sigma):
    """Normalized gaussian (sums to 1) centred at (x, y) given in [0, 1]."""
    cx, cy = x * size, y * size
    coords = (np.arange(size, dtype=np.float32) + 0.5)
    gx = np.exp(-((coords - cx) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((coords - cy) ** 2) / (2 * sigma ** 2))
    hm = np.outer(gy, gx)
    hm /= hm.sum() + 1e-8
    return torch.from_numpy(hm).unsqueeze(0)


class GazeFollowSpatialDataset(Dataset):
    def __init__(
        self,
        csv_path,
        project_root,
        train=False,
        image_size=224,
        crop_size=112,
        heatmap_size=56,
        heatmap_sigma=2.0,
        crop_expand=0.2,
        cache_max_side=512,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.root = Path(project_root)
        # Decoding full-size JPEGs for every annotation row is the main
        # bottleneck, so each image is downscaled once (long side <= 512) and
        # reused. Coordinates are normalized, so labels are unaffected.
        self.paths = self._build_small_cache(cache_max_side) if cache_max_side else list(
            self.df["image_path"]
        )
        self.train = train
        self.S = image_size
        self.C = crop_size
        self.H = heatmap_size
        self.sigma = heatmap_sigma
        self.expand = crop_expand

        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.df)

    def _build_small_cache(self, max_side):
        small_dir = self.root / "data" / "gazefollow" / "images_small"
        small_dir.mkdir(parents=True, exist_ok=True)
        mapping = {}
        todo = [p for p in self.df["image_path"].unique()
                if not (small_dir / Path(p).name).exists()]
        if todo:
            print(f"Caching {len(todo)} downscaled images to {small_dir} (one-time)...")
        for p in todo:
            with Image.open(self.root / p) as im:
                im.draft("RGB", (max_side, max_side))  # fast JPEG downscale on decode
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side))
                im.save(small_dir / Path(p).name, quality=95)
        for p in self.df["image_path"].unique():
            mapping[p] = str(Path("data") / "gazefollow" / "images_small" / Path(p).name)
        return [mapping[p] for p in self.df["image_path"]]

    def _crop_head(self, img, xmin, ymin, xmax, ymax):
        w, h = img.size
        bw, bh = (xmax - xmin), (ymax - ymin)
        x0 = max(0.0, xmin - self.expand * bw) * w
        y0 = max(0.0, ymin - self.expand * bh) * h
        x1 = min(1.0, xmax + self.expand * bw) * w
        y1 = min(1.0, ymax + self.expand * bh) * h
        # guard against degenerate boxes
        if x1 - x0 < 4:
            x1 = min(w, x0 + 4)
        if y1 - y0 < 4:
            y1 = min(h, y0 + 4)
        return img.crop((int(x0), int(y0), int(x1), int(y1)))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / self.paths[idx]).convert("RGB")

        xmin, ymin = float(row["head_xmin"]), float(row["head_ymin"])
        xmax, ymax = float(row["head_xmax"]), float(row["head_ymax"])
        gx, gy = float(row["gaze_x"]), float(row["gaze_y"])

        if self.train:
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
                xmin, xmax = 1.0 - xmax, 1.0 - xmin
                gx = 1.0 - gx
            img = self.color_jitter(img)

        head_crop = self._crop_head(img, xmin, ymin, xmax, ymax)
        head_crop = TF.to_tensor(head_crop.resize((self.C, self.C)))
        head_crop = self.normalize(head_crop)

        scene = TF.to_tensor(img.resize((self.S, self.S)))
        scene = self.normalize(scene)

        mask = torch.zeros(1, self.S, self.S)
        x0, x1 = int(xmin * self.S), max(int(xmax * self.S), int(xmin * self.S) + 1)
        y0, y1 = int(ymin * self.S), max(int(ymax * self.S), int(ymin * self.S) + 1)
        mask[:, y0:y1, x0:x1] = 1.0

        return {
            "scene": scene,
            "head_mask": mask,
            "head_crop": head_crop,
            "gaze_target": torch.tensor([gx, gy], dtype=torch.float32),
            "heatmap": make_gaussian_heatmap(gx, gy, self.H, self.sigma),
            "image_id": int(row["image_id"]),
            "head_bbox": torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32),
        }
