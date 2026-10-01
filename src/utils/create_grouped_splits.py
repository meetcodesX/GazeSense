from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]

SOURCE_CSV = ROOT / "data" / "gazefollow" / "annotations.csv"
OUTPUT_DIR = ROOT / "data" / "gazefollow" / "grouped_splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE_CSV)

# Preserve source image IDs and all distinct annotations.
# Remove only exact duplicate rows.
df = df.drop_duplicates().reset_index(drop=True)

print("Source rows:", len(df))
print("Unique source images:", df["image_id"].nunique())
print("Image ID range:", df["image_id"].min(), "to", df["image_id"].max())

# Group by image_id so annotations for one image stay together.
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=42
)

train_val_idx, test_idx = next(
    split1.split(df, groups=df["image_id"])
)

train_val = df.iloc[train_val_idx].copy()
test_df = df.iloc[test_idx].copy()

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=1 / 9,
    random_state=42
)

train_idx, val_idx = next(
    split2.split(train_val, groups=train_val["image_id"])
)

train_df = train_val.iloc[train_idx].copy()
val_df = train_val.iloc[val_idx].copy()

# Save fresh splits, including image_path.
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Verify IDs and image-level separation.
train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])
test_ids = set(test_df["image_id"])

print("\n--- SPLIT SUMMARY ---")
for name, split_df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(
        f"{name}: {len(split_df)} rows, "
        f"{split_df['image_id'].nunique()} images"
    )

print("\n--- OVERLAP ---")
print("Train ∩ Val:", len(train_ids & val_ids))
print("Train ∩ Test:", len(train_ids & test_ids))
print("Val ∩ Test:", len(val_ids & test_ids))

print("\n--- IMAGE PATH CHECK ---")
for image_id in sorted(df["image_id"].unique()):
    if image_id == 0:
        print("Source contains image_id 0")
        break
else:
    print("Source has no image_id 0")