from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Project root: C:\GazeSense
ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "data" / "gazefollow" / "annotations.csv"
OUTPUT_DIR = ROOT / "outputs" / "gazefollow_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Select a few images that have annotations
image_ids = df["image_id"].drop_duplicates().head(5)

for image_id in image_ids:
    rows = df[df["image_id"] == image_id]

    # Use image path from CSV
    image_path = ROOT / rows.iloc[0]["image_path"]

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        continue

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)

    # Draw head bounding box once
    row = rows.iloc[0]

    x1 = row["head_xmin"] * width
    y1 = row["head_ymin"] * height
    x2 = row["head_xmax"] * width
    y2 = row["head_ymax"] * height

    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
            label="Head bounding box"
        )
    )

    # Draw all gaze target points for this image
    gaze_x = rows["gaze_x"] * width
    gaze_y = rows["gaze_y"] * height

    ax.scatter(
        gaze_x,
        gaze_y,
        color="red",
        s=35,
        marker="x",
        label="Gaze target"
    )

    # Draw eye location
    eye_x = row["eye_x"] * width
    eye_y = row["eye_y"] * height

    ax.scatter(
        [eye_x],
        [eye_y],
        color="cyan",
        s=45,
        marker="o",
        label="Eye location"
    )

    ax.set_title(
        f"Image ID: {image_id} | Gaze annotations: {len(rows)}"
    )
    ax.legend()
    ax.axis("off")

    output_path = OUTPUT_DIR / f"{image_id}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")

print("Visualization complete.")
