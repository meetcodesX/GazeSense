# Dataset

This folder contains the processed GazeFollow dataset files used in the project.

## Dataset

- Dataset: GazeFollow
- Source split: Test
- Images: 4,782
- Gaze annotations: 44,191

## Files

- `gaze_dataset.csv` – Processed gaze annotations
- `gaze_dataset_normalized.csv` – Normalized dataset
- `train.csv` – Project-generated training split
- `val.csv` – Project-generated validation split
- `test.csv` – Project-generated test split
- `experiment_log.csv` – Dataset preparation experiment record

## Important

The original GazeFollow dataset files are not included in this repository because of their size and redistribution considerations.

The train, validation, and test CSV files are project-generated splits created from the available dataset split.

## Access

The original dataset can be downloaded using `download_gazefollow.py`.