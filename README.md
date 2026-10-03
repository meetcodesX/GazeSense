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

## Experiments

Spatial-preserving models that output a heatmap (soft-argmax readout) instead
of direct coordinate regression. Two variants are provided:

| Run ID | Variant        | Description                              |
|--------|----------------|------------------------------------------|
| E1A    | `scene_mask`   | Scene + head-location mask (ablation)    |
| E1B    | `two_pathway`  | Scene + head mask + head-crop pathway    |

```bash
# E1A – scene + head mask only (ablation)
python -m src.train_spatial --run-id E1A --variant scene_mask --epochs 30

# E1B – two-pathway (main model)
python -m src.train_spatial --run-id E1B --variant two_pathway --epochs 30
```

Results are logged to `experiments/log.csv`.
Best checkpoints are saved to `outputs/<run-id>/best_model.pth` (git-ignored).
