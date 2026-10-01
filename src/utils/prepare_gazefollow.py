import io
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


# Project paths
ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = ROOT / "gazefollow" / "data" / "test-00000-of-00001.parquet"

OUTPUT_DIR = ROOT / "data" / "gazefollow"
IMAGE_DIR = OUTPUT_DIR / "images"
CSV_PATH = OUTPUT_DIR / "annotations.csv"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

if not PARQUET_PATH.exists():
    raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")

table = pq.read_table(PARQUET_PATH)
records = table.to_pylist()

rows = []
saved_images = set()

for record in records:
    image_info = record["image"]
    image_name = Path(image_info["path"]).name
    image_id = Path(image_name).stem
    image_output_path = IMAGE_DIR / image_name

    # Save each image only once
    image_bytes = image_info.get("bytes")

    if image_name not in saved_images and image_bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.convert("RGB").save(image_output_path)

        saved_images.add(image_name)

    # One CSV row per gaze annotation
    for annotation in record["gazes"]:
        head = annotation["head_bbox"]
        eye = annotation["eye"]
        gaze = annotation["gaze"]
        body = annotation["body_bbox"]

        rows.append({
            "image_id": image_id,
            "image_path": f"data/gazefollow/images/{image_name}",

            "head_xmin": head["xmin"],
            "head_ymin": head["ymin"],
            "head_xmax": head["xmax"],
            "head_ymax": head["ymax"],

            "eye_x": eye["x"],
            "eye_y": eye["y"],

            "gaze_x": gaze["x"],
            "gaze_y": gaze["y"],

            "body_x": body["x"],
            "body_y": body["y"],
            "body_w": body["w"],
            "body_h": body["h"],

            "in_out": annotation["in_out"]
        })

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)

print("Preparation complete!")
print("Image records in Parquet:", len(records))
print("Images saved:", len(saved_images))
print("Total gaze annotations:", len(df))
print("CSV saved:", CSV_PATH)

print("\nMissing values:")
print(df.isnull().sum())

coordinate_columns = [
    "head_xmin", "head_ymin", "head_xmax", "head_ymax",
    "eye_x", "eye_y", "gaze_x", "gaze_y",
    "body_x", "body_y", "body_w", "body_h"
]

print("\nCoordinate ranges:")
print(df[coordinate_columns].agg(["min", "max"]).T)

print("\nFirst 5 annotation rows:")
print(df.head().to_string(index=False))
