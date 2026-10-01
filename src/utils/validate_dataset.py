import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

files = ["train.csv", "val.csv", "test.csv"]

required_columns = {
    "image_id",
    "head_xmin", "head_ymin", "head_xmax", "head_ymax",
    "eye_x", "eye_y",
    "gaze_x", "gaze_y",
    "body_x", "body_y", "body_w", "body_h"
}

for file in files:
    path = DATA_DIR / file

    if not path.exists():
        print(f"\n❌ Missing file: {path}")
        continue

    df = pd.read_csv(path)

    print(f"\n{'=' * 45}")
    print(f"FILE: {file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    missing_columns = required_columns - set(df.columns)
    print("Missing columns:", missing_columns or "None")

    print("Null values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Unique image IDs:", df["image_id"].nunique())

    coordinate_columns = list(required_columns - {"image_id"})

    out_of_range = (
        (df[coordinate_columns] < 0) |
        (df[coordinate_columns] > 1)
    ).sum().sum()

    print("Coordinate values outside [0, 1]:", out_of_range)

    print("\nSample rows:")
    print(df.head(3).to_string(index=False))