"""
src/training/augment_minority.py

Balances the training split for the 3-class compliance classifier via
stratified oversampling of minority classes and controlled undersampling
of the dominant majority class.

Constraints:
  - NO data leakage: justification column is never used as input text.
  - NO modification of val.csv or test.csv.
  - All randomness is seeded (random_state=42) for full reproducibility.

Output: datasets/splits/train_augmented.csv
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPLITS_DIR = Path("datasets/splits")
INPUT_PATH = SPLITS_DIR / "train.csv"
OUTPUT_PATH = SPLITS_DIR / "train_augmented.csv"

RANDOM_STATE = 42

# Target counts after rebalancing
NOT_ADDRESSED_CAP = 250        # Down-sample from 470 -> 250
MINORITY_TARGET = 150          # Oversample Compliant and Partially Compliant to 150

LABEL_COL = "verdict"

TARGET_COUNTS = {
    "Not Addressed":       NOT_ADDRESSED_CAP,
    "Partially Compliant": MINORITY_TARGET,
    "Compliant":           MINORITY_TARGET,
}

# ---------------------------------------------------------------------------
# Load & Validate
# ---------------------------------------------------------------------------
train_df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("ORIGINAL TRAINING DISTRIBUTION")
print("=" * 60)
print(train_df[LABEL_COL].value_counts().to_string())
print(f"Total: {len(train_df)}\n")

# Drop any solitary edge-case labels (e.g. Non-Compliant = 1 row)
train_df = train_df[train_df[LABEL_COL].isin(TARGET_COUNTS.keys())].copy()

# ---------------------------------------------------------------------------
# Resample each class independently
# ---------------------------------------------------------------------------
resampled_parts = []

for label, target_n in TARGET_COUNTS.items():
    class_df = train_df[train_df[LABEL_COL] == label]
    n_available = len(class_df)

    if n_available == 0:
        print(f"  WARNING: No samples found for '{label}' - skipping.")
        continue

    if n_available >= target_n:
        # Downsample (majority class) - sample WITHOUT replacement
        resampled = class_df.sample(n=target_n, replace=False, random_state=RANDOM_STATE)
        action = f"downsampled {n_available} -> {target_n}"
    else:
        # Oversample (minority class) - sample WITH replacement
        resampled = class_df.sample(n=target_n, replace=True, random_state=RANDOM_STATE)
        action = f"oversampled {n_available} -> {target_n} (x{target_n/n_available:.1f})"

    resampled_parts.append(resampled)
    print(f"  {label}: {action}")

# ---------------------------------------------------------------------------
# Combine, shuffle, reset index
# ---------------------------------------------------------------------------
augmented_df = (
    pd.concat(resampled_parts, ignore_index=True)
    .sample(frac=1.0, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
augmented_df.to_csv(OUTPUT_PATH, index=False)

print()
print("=" * 60)
print("AUGMENTED TRAINING DISTRIBUTION")
print("=" * 60)
print(augmented_df[LABEL_COL].value_counts().to_string())
print(f"Total: {len(augmented_df)}")

old_ratio = 470 / 25
new_ratio = NOT_ADDRESSED_CAP / MINORITY_TARGET
print(f"\nImbalance ratio (NA : Compliant)")
print(f"  Before: {old_ratio:.0f}:1")
print(f"  After:  {new_ratio:.1f}:1")
print(f"\nSaved to: {OUTPUT_PATH}")
print("Val/Test splits were NOT modified.")
