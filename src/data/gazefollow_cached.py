"""Fast, GPU-resident GazeFollow data pipeline.

The PIL/DataLoader pipeline was CPU-bound (~46 samples/s on Colab's 2 CPUs).
Here every unique person (image + head box) is decoded ONCE, stored as uint8
tensors (scene 224x224 and head crop 112x112), and kept on the GPU. Batching,
flip, brightness/contrast jitter, head mask and gaussian heatmap targets are
all computed on the GPU, so training becomes GPU-bound.

Cache files live in data/gazefollow/cache/*.pt (git-ignored via *.pt).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _open_image(root, image_path):
    small = root / "data" / "gazefollow" / "images_small" / Path(image_path).name
    path = small if small.exists() else root / image_path
    img = Image.open(path)
    img.draft("RGB", (512, 512))  # fast JPEG downscale if the original is large
    return img.convert("RGB")


def _crop_head(img, xmin, ymin, xmax, ymax, expand):
    w, h = img.size
    bw, bh = xmax - xmin, ymax - ymin
    x0 = max(0.0, xmin - expand * bw) * w
    y0 = max(0.0, ymin - expand * bh) * h
    x1 = min(1.0, xmax + expand * bw) * w
    y1 = min(1.0, ymax + expand * bh) * h
    if x1 - x0 < 4:
        x1 = min(w, x0 + 4)
    if y1 - y0 < 4:
        y1 = min(h, y0 + 4)
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def build_or_load_cache(split, root, image_size=224, crop_size=112, expand=0.2):
    root = Path(root)
    cache_dir = root / "data" / "gazefollow" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_{image_size}_{crop_size}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)

    df = pd.read_csv(root / "data" / "gazefollow" / "grouped_splits" / f"{split}.csv")
    cols = ["image_path", "head_xmin", "head_ymin", "head_xmax", "head_ymax"]
    key = df[cols].round(5).astype(str).agg("|".join, axis=1)
    uid, uniques = pd.factorize(key)
    first_row = pd.Series(np.arange(len(df))).groupby(uid).first().to_numpy()

    print(f"[{split}] caching {len(first_row)} unique people from {len(df)} rows (one-time)...")
    scenes = np.zeros((len(first_row), image_size, image_size, 3), dtype=np.uint8)
    crops = np.zeros((len(first_row), crop_size, crop_size, 3), dtype=np.uint8)
    boxes = np.zeros((len(first_row), 4), dtype=np.float32)

    last_path, img = None, None
    for u, r in enumerate(first_row):
        row = df.iloc[r]
        if row["image_path"] != last_path:
            img, last_path = _open_image(root, row["image_path"]), row["image_path"]
        xmin, ymin = float(row["head_xmin"]), float(row["head_ymin"])
        xmax, ymax = float(row["head_xmax"]), float(row["head_ymax"])
        scenes[u] = np.asarray(img.resize((image_size, image_size)))
        crops[u] = np.asarray(
            _crop_head(img, xmin, ymin, xmax, ymax, expand).resize((crop_size, crop_size))
        )
        boxes[u] = (xmin, ymin, xmax, ymax)

    data = {
        "scene": torch.from_numpy(scenes).permute(0, 3, 1, 2).contiguous(),  # (U,3,S,S) uint8
        "crop": torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous(),    # (U,3,C,C) uint8
        "box": torch.from_numpy(boxes),                                      # (U,4)
        "uid": torch.from_numpy(uid.astype(np.int64)),                       # (N,)
        "gaze": torch.tensor(df[["gaze_x", "gaze_y"]].to_numpy(dtype=np.float32)),  # (N,2)
        "image_id": torch.tensor(df["image_id"].astype(int).to_numpy(), dtype=torch.long),
    }
    torch.save(data, cache_path)
    return data


class GPUBatcher:
    """Iterates over annotation rows in batches, entirely on `device`."""

    def __init__(self, data, device, batch_size=32, train=False, heatmap_size=56,
                 heatmap_sigma=2.0, seed=0):
        self.device, self.bs, self.train = device, batch_size, train
        self.H, self.sigma = heatmap_size, heatmap_sigma
        self.scene = data["scene"].to(device)
        self.crop = data["crop"].to(device)
        self.box = data["box"].to(device)
        self.uid = data["uid"].to(device)
        self.gaze = data["gaze"].to(device)
        self.image_id = data["image_id"]  # stays on CPU (used for grouping only)
        self.S, self.C = self.scene.shape[-1], self.crop.shape[-1]
        self.n = len(self.uid)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.grid_h = torch.arange(self.H, device=device, dtype=torch.float32) + 0.5
        self.grid_s = torch.arange(self.S, device=device)

    def __len__(self):
        full, rem = divmod(self.n, self.bs)
        return full + (1 if rem and not self.train else 0)

    def _heatmap(self, gaze):
        cx, cy = gaze[:, 0:1] * self.H, gaze[:, 1:2] * self.H
        gx = torch.exp(-((self.grid_h[None] - cx) ** 2) / (2 * self.sigma ** 2))
        gy = torch.exp(-((self.grid_h[None] - cy) ** 2) / (2 * self.sigma ** 2))
        hm = gy[:, :, None] * gx[:, None, :]
        return (hm / (hm.sum((1, 2), keepdim=True) + 1e-8)).unsqueeze(1)

    def _mask(self, box):
        x0 = (box[:, 0] * self.S).long()
        x1 = torch.maximum((box[:, 2] * self.S).long(), x0 + 1)
        y0 = (box[:, 1] * self.S).long()
        y1 = torch.maximum((box[:, 3] * self.S).long(), y0 + 1)
        g = self.grid_s[None]
        mx = (g >= x0[:, None]) & (g < x1[:, None])  # (B,S)
        my = (g >= y0[:, None]) & (g < y1[:, None])
        return (my[:, :, None] & mx[:, None, :]).float().unsqueeze(1)

    def __iter__(self):
        if self.train:
            order = torch.randperm(self.n, generator=self.gen).to(self.device)
        else:
            order = torch.arange(self.n, device=self.device)
        for start in range(0, self.n, self.bs):
            idx = order[start:start + self.bs]
            if self.train and len(idx) < self.bs:
                break  # drop last (BatchNorm-friendly)
            u = self.uid[idx]
            scene = self.scene[u].float() / 255.0
            crop = self.crop[u].float() / 255.0
            box = self.box[u].clone()
            gaze = self.gaze[idx].clone()

            if self.train:
                b = idx.numel()
                flip = torch.rand(b, device=self.device) < 0.5
                f4 = flip[:, None, None, None]
                scene = torch.where(f4, scene.flip(3), scene)
                crop = torch.where(f4, crop.flip(3), crop)
                box = torch.where(
                    flip[:, None], torch.stack([1 - box[:, 2], box[:, 1], 1 - box[:, 0], box[:, 3]], 1), box
                )
                gaze = torch.where(flip[:, None], torch.stack([1 - gaze[:, 0], gaze[:, 1]], 1), gaze)
                # brightness / contrast jitter (same factors for scene and crop)
                br = (torch.rand(b, 1, 1, 1, device=self.device) * 0.6 + 0.7)
                ct = (torch.rand(b, 1, 1, 1, device=self.device) * 0.6 + 0.7)
                scene, crop = scene * br, crop * br
                scene = ((scene - scene.mean((1, 2, 3), keepdim=True)) * ct
                         + scene.mean((1, 2, 3), keepdim=True)).clamp(0, 1)
                crop = ((crop - crop.mean((1, 2, 3), keepdim=True)) * ct
                        + crop.mean((1, 2, 3), keepdim=True)).clamp(0, 1)

            yield {
                "scene": (scene - self.mean) / self.std,
                "head_crop": (crop - self.mean) / self.std,
                "head_mask": self._mask(box),
                "gaze_target": gaze,
                "heatmap": self._heatmap(gaze),
                "image_id": self.image_id[idx.cpu()],
                "head_bbox": box,
            }