"""
src/training/split_dataset.py
Standard 70/15/15 stratified train/val/test split across all 771 policy clauses.
Handles single-instance edge cases safely and exports both to datasets/splits/ and corpus/.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("corpus/labeled_clauses_bootstrap.csv")
OUTPUT_DIR = Path("datasets/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

TARGET_COL = "verdict"
RANDOM_STATE = 42

# 1. Isolate rare single-instance classes (like Non-Compliant = 1 row) to avoid ValueError
class_counts = df[TARGET_COL].value_counts()
rare_classes = class_counts[class_counts < 2].index

rare_df = df[df[TARGET_COL].isin(rare_classes)]
stratifiable_df = df[~df[TARGET_COL].isin(rare_classes)]

# 2. Stage 1: 70% Train, 30% Temp (Val + Test) on stratifiable data
train_rest, temp_df = train_test_split(
    stratifiable_df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=stratifiable_df[TARGET_COL],
)

# Route the rare class directly into train set so the model encounters it
train_df = pd.concat([train_rest, rare_df], ignore_index=True)

# 3. Stage 2: Split remaining 30% equally into Val (15%) and Test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df[TARGET_COL],
)

# Export to datasets/splits/
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Mirror to corpus/ for backward compatibility with downstream training scripts
train_df.to_csv(Path("corpus/labeled_clauses_final_train.csv"), index=False)
val_df.to_csv(Path("corpus/labeled_clauses_val.csv"), index=False)
test_df.to_csv(Path("corpus/labeled_clauses_gold.csv"), index=False)

print("=" * 60)
print("STRATIFIED SPLIT COMPLETE (All 771 Clauses Preserved)")
print("=" * 60)
print(f"Total dataset: {len(df)} clauses")
print(f"  Train:      {len(train_df)} rows ({len(train_df)/len(df):.1%})")
print(f"  Val:        {len(val_df)} rows ({len(val_df)/len(df):.1%})")
print(f"  Test (Gold):{len(test_df)} rows ({len(test_df)/len(df):.1%})")

print("\n--- Train Class Distribution ---")
print(train_df[TARGET_COL].value_counts().to_string())

print("\n--- Val Class Distribution ---")
print(val_df[TARGET_COL].value_counts().to_string())

print("\n--- Test (Gold) Class Distribution ---")
print(test_df[TARGET_COL].value_counts().to_string())