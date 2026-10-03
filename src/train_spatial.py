"""Train + evaluate the spatial-preserving gaze models.

Examples (run from the project root):
  python -m src.train_spatial --run-id E1A --variant scene_mask  --epochs 30
  python -m src.train_spatial --run-id E1B --variant two_pathway --epochs 30

Every run appends one row to experiments/log.csv and saves its best
checkpoint to outputs/<run-id>/best_model.pth (checkpoints are git-ignored).
Model selection uses val mean distance; the test split is only evaluated
once, at the end, with the best checkpoint.
"""
import argparse
import csv
import os
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.gazefollow_spatial_dataset import GazeFollowSpatialDataset
from src.models.gaze_spatial import GazeSpatialModel

LOG_FIELDS = [
    "run_id", "date", "model", "lr", "batch", "epochs", "seed",
    "val_dist", "test_mse", "test_dist", "test_min_dist", "notes",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(model, logits, batch, device, coord_weight):
    """Cross-entropy to the gaussian target + MSE on the soft-argmax point."""
    target_hm = batch["heatmap"].to(device).flatten(1)  # sums to 1
    log_p = F.log_softmax(logits.flatten(1), dim=1)
    ce = -(target_hm * log_p).sum(1).mean()
    coords = model.soft_argmax(logits)
    mse = F.mse_loss(coords, batch["gaze_target"].to(device))
    return ce + coord_weight * mse, coords


def forward_batch(model, batch, device, variant):
    scene = batch["scene"].to(device)
    mask = batch["head_mask"].to(device)
    crop = batch["head_crop"].to(device) if variant == "two_pathway" else None
    return model(scene, mask, crop)


def run_train_epoch(model, loader, optimizer, scaler, device, variant, coord_weight, max_batches):
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if i % 100 == 0:
            print(f"  batch {i}/{len(loader)}  ({time.time() - t0:.0f}s)", flush=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = forward_batch(model, batch, device, variant)
        loss, _ = compute_loss(model, logits.float(), batch, device, coord_weight)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * logits.size(0)
        n += logits.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, variant, max_batches=0):
    """Returns dict with row-level MSE / mean dist (soft + hard argmax) and
    GazeFollow-style min distance (best of all annotations of the same person)."""
    model.eval()
    ids, boxes, soft, hard, tgt = [], [], [], [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        logits = forward_batch(model, batch, device, variant).float()
        soft.append(model.soft_argmax(logits).cpu())
        hard.append(model.hard_argmax(logits).cpu())
        tgt.append(batch["gaze_target"])
        ids.append(batch["image_id"])
        boxes.append(batch["head_bbox"])
    soft, hard, tgt = torch.cat(soft), torch.cat(hard), torch.cat(tgt)
    ids, boxes = torch.cat(ids).numpy(), torch.cat(boxes).numpy().round(3)

    d_soft = (soft - tgt).norm(dim=1).numpy()
    d_hard = (hard - tgt).norm(dim=1).numpy()

    # min distance: group annotations of the same person (same image + head box)
    df = pd.DataFrame({"img": ids, "b": [tuple(b) for b in boxes], "d": d_soft})
    min_dist = df.groupby(["img", "b"])["d"].min().mean()

    return {
        "mse": float(((soft - tgt) ** 2).mean()),
        "dist": float(d_soft.mean()),
        "dist_hard": float(d_hard.mean()),
        "min_dist": float(min_dist),
    }


def append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    old_rows = []
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            # older logs (e.g. BASE01) have fewer columns: keep rows, add columns
            old_rows = [r for r in reader if r.get("run_id")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        for r in old_rows + [row]:
            w.writerow({k: r.get(k, "") for k in LOG_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant", choices=["scene_mask", "two_pathway"], default="two_pathway")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="lr for decoder/head layers")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--coord-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--root", default=None, help="project root (default: repo root)")
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: limit batches/epoch")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    split_dir = root / "data" / "gazefollow" / "grouped_splits"
    out_dir = root / "outputs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| run:", args.run_id, "| variant:", args.variant)

    def make_loader(name, train):
        ds = GazeFollowSpatialDataset(split_dir / f"{name}.csv", root, train=train)
        return DataLoader(
            ds, batch_size=args.batch, shuffle=train,
            num_workers=args.workers, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            drop_last=train and len(ds) > args.batch,
        )

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    model = GazeSpatialModel(args.variant, pretrained=not args.no_pretrained).to(device)

    backbone = list(model.scene.parameters()) + (
        list(model.head.parameters()) if model.head is not None else []
    )
    backbone_ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.lr * args.backbone_lr_mult},
            {"params": rest, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val, best_epoch = float("inf"), 0
    ckpt = out_dir / "best_model.pth"
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(
            model, train_loader, optimizer, scaler, device,
            args.variant, args.coord_weight, args.max_batches,
        )
        scheduler.step()
        val = evaluate(model, val_loader, device, args.variant, args.max_batches)
        print(
            f"Epoch [{epoch}/{args.epochs}] train loss {train_loss:.4f} | "
            f"val MSE {val['mse']:.5f} | val dist {val['dist']:.4f} "
            f"(argmax {val['dist_hard']:.4f}, min {val['min_dist']:.4f}) | "
            f"{(time.time() - t0) / 60:.1f} min"
        )
        if val["dist"] < best_val:
            best_val, best_epoch = val["dist"], epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "variant": args.variant, "val_dist": val["dist"]},
                ckpt,
            )
            print("  saved best checkpoint")

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    test = evaluate(model, test_loader, device, args.variant, args.max_batches)
    print("\n--- TEST RESULTS (best val checkpoint, epoch %d) ---" % best_epoch)
    print(f"Test MSE:                 {test['mse']:.6f}")
    print(f"Test mean distance:       {test['dist']:.6f}")
    print(f"Test mean dist (argmax):  {test['dist_hard']:.6f}")
    print(f"Test min distance:        {test['min_dist']:.6f}")

    append_log(root / "experiments" / "log.csv", {
        "run_id": args.run_id, "date": date.today().isoformat(),
        "model": f"ResNet18 {args.variant} heatmap",
        "lr": args.lr, "batch": args.batch, "epochs": args.epochs, "seed": args.seed,
        "val_dist": f"{best_val:.6f}", "test_mse": f"{test['mse']:.6f}",
        "test_dist": f"{test['dist']:.6f}", "test_min_dist": f"{test['min_dist']:.6f}",
        "notes": f"best ep {best_epoch}; {args.notes}".strip("; "),
    })


if __name__ == "__main__":
    main()
