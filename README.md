# GazeSense — GazeFollow Gaze Estimation

A gaze-target estimation project built on the GazeFollow dataset.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Baseline Training

```bash
python -m src.train
python -m src.evaluate
```

## Results (GazeFollow, same test split, 4,373 samples)

| Run | Model | Test mean distance | Test min distance | Test MSE |
|-----|-------|--------------------|-------------------|----------|
| Constant guess (image center) | none | 0.317 | n/a | n/a |
| BASE01 | 4-layer CNN, global avg pooling | 0.253 | n/a | 0.0404 |
| E1A | ResNet18 scene + head mask, heatmap output | 0.211 | 0.137 | 0.0401 |
| E1B | E1A + head-crop pathway | **0.1851 ± 0.0013** | **0.1130 ± 0.0013** | **0.0297 ± 0.0013** |

- BASE01 and E1A are reported from a single seed (42). E1B was evaluated using three seeds (42, 123, 456).
- E1B results are reported as mean ± sample standard deviation over 3 seeds.
- Distance is computed per annotation row against normalized (x, y) in [0, 1]. This is not the standard GazeFollow protocol (which compares against the average of annotators), so these numbers are not comparable to published results.
- E1A has a similar MSE to the baseline but a lower mean distance, because the heatmap model sometimes lands on a wrong region, which gives larger errors.
- Training data is only 3,824 images (the HF copy contains just the GazeFollow test split, which we split by image into train/val/test).

## How to reproduce

Setup:

```bash
pip install -r requirements.txt
python download_gazefollow.py
python src/utils/prepare_gazefollow.py
```

Run (about 1.5 min/epoch on a Colab T4 GPU, about 45 min for 30 epochs):

```bash
python -m src.train_spatial --run-id E1A --variant scene_mask --epochs 30
python -m src.train_spatial --run-id E1B --variant two_pathway --epochs 30
```

Every run appends a row to `experiments/log.csv`. Checkpoints are saved to `outputs/<run-id>/best_model.pth` and are not committed to Git.

## Files

- `src/data/gazefollow_cached.py` — GPU-resident data pipeline (one-time cache, GPU augmentation)
- `src/models/gaze_spatial.py` — `scene_mask` and `two_pathway` models with soft-argmax heatmap output
- `src/train_spatial.py` — training, evaluation, and logging
