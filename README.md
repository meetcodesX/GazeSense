# GazeSense — GazeFollow Gaze Estimation

A gaze-target estimation project built on the **GazeFollow dataset**.

GazeSense estimates gaze targets in **image-space coordinates** using a spatial heatmap-based deep learning architecture.

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Baseline Training

To train and evaluate the original baseline model:

```bash
python -m src.train
python -m src.evaluate
```

---

## Results

Results are reported on the same test split containing **4,373 samples**.

| Run | Model | Test Mean Distance | Test Min Distance | Test MSE |
|-----|-------|-------------------:|------------------:|---------:|
| Constant guess (image center) | None | 0.317 | N/A | N/A |
| BASE01 | 4-layer CNN, global average pooling | 0.253 | N/A | 0.0404 |
| E1A | ResNet18 scene + head mask, heatmap output | 0.211 | 0.137 | 0.0401 |
| E1B | E1A + head-crop pathway | **0.1851 ± 0.0013** | **0.1130 ± 0.0013** | **0.0297 ± 0.0013** |

### Notes

- BASE01 and E1A are reported from a single seed (`42`).
- E1B was evaluated using three seeds: `42`, `123`, and `456`.
- E1B results are reported as **mean ± sample standard deviation** over the three seeds.
- Distance is computed per annotation row against normalized `(x, y)` coordinates in `[0, 1]`.
- This is **not the standard GazeFollow evaluation protocol**, which compares predictions against the average of annotators. Therefore, these numbers are **not directly comparable to published GazeFollow results**.
- E1A has a similar MSE to the baseline but a lower mean distance because the heatmap model can sometimes place its prediction in an incorrect region, resulting in larger coordinate errors.
- The available Hugging Face copy contains only the GazeFollow test split. We split it by image into our own train, validation, and test sets.

---

## Demo

GazeSense performs **image-space gaze-target estimation** using the E1B two-pathway model trained on GazeFollow.

The video demo shows:

- Face detection
- Estimated head region
- E1B gaze target
- Temporal gaze trajectory

> **Note:** The current system estimates gaze targets in the camera-image coordinate space. It does **not** perform calibrated screen-coordinate eye tracking.

### Run Video Inference

Place a demo video at:

```text
data/demo/gaze_video.mp4
```

Then run:

```bash
python -m src.inference.video_inference
```

The generated demo video is saved to:

```text
outputs/gaze_demo_final.mp4
```

---

## How to Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the GazeFollow dataset

```bash
python download_gazefollow.py
```

### 3. Prepare the dataset

```bash
python src/utils/prepare_gazefollow.py
```

### 4. Train E1A

```bash
python -m src.train_spatial \
    --run-id E1A \
    --variant scene_mask \
    --epochs 30
```

### 5. Train E1B

```bash
python -m src.train_spatial \
    --run-id E1B \
    --variant two_pathway \
    --epochs 30
```

Training takes approximately **1.5 minutes per epoch on a Colab T4 GPU**, or approximately **45 minutes for 30 epochs**.

Every training run appends a row to:

```text
experiments/log.csv
```

Checkpoints are saved to:

```text
outputs/<run-id>/best_model.pth
```

Model checkpoints are **not committed to Git**.

---

## Model Variants

### BASE01

A simple 4-layer CNN baseline using:

- Convolutional feature extraction
- Global average pooling
- Bounding-box features
- Coordinate regression

### E1A — Scene + Head Mask

A ResNet18-based spatial model that uses:

- RGB scene image
- Head-region mask
- Spatial heatmap prediction
- Soft-argmax for gaze coordinates

### E1B — Two-Pathway Model

E1B extends E1A with an additional **head-crop pathway**.

It uses:

- ResNet18 scene encoder
- Head-region mask
- Dedicated head-crop encoder
- Spatial heatmap decoder
- Soft-argmax gaze estimation
- ImageNet-pretrained ResNet18 backbones

The head-crop pathway provides additional information about the person's head region and improves gaze-target estimation performance.

---

## Dataset Splits

The project uses grouped splits by image ID to prevent the same image from appearing across multiple splits.

| Split | Annotation Rows | Images |
|-------|----------------:|-------:|
| Train | 35,254 | 3,824 |
| Validation | 4,417 | 479 |
| Test | 4,373 | 479 |

The splits are created at the **image level**, while multiple gaze annotations for the same image are preserved.

---

## Project Structure

```text
GazeSense/
│
├── data/
│   ├── demo/
│   │   └── gaze_video.mp4
│   │
│   └── gazefollow/
│       ├── images/
│       ├── annotations.csv
│       └── grouped_splits/
│
├── experiments/
│   └── log.csv
│
├── outputs/
│   └── <run-id>/
│       └── best_model.pth
│
├── src/
│   ├── data/
│   │   └── gazefollow_cached.py
│   │
│   ├── models/
│   │   └── gaze_spatial.py
│   │
│   ├── inference/
│   │   └── video_inference.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── train_spatial.py
│
├── download_gazefollow.py
├── requirements.txt
└── README.md
```