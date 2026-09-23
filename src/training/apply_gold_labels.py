"""
src/training/apply_gold_labels.py
Applies legally audited gold promotions to create corpus/labeled_clauses_gold.csv.
"""

from pathlib import Path
import pandas as pd

INPUT_PATH = Path("corpus/labeled_clauses_final_train.csv")
if not INPUT_PATH.exists():
    INPUT_PATH = Path("corpus/labeled_clauses_bootstrap.csv")

OUTPUT_PATH = Path("corpus/labeled_clauses_gold.csv")
df = pd.read_csv(INPUT_PATH)

# Audited Ground Truth Promotions
PROMOTED_COMPLIANT = {
    # Data Principal Rights (Access, Portability, Erasure)
    "S000331": "Data Principal Rights",
    "S001152": "Data Principal Rights",
    "S001585": "Data Principal Rights",
    "S001147": "Data Retention & Erasure",
    "S001589": "Data Retention & Erasure",
    "S002364": "Data Retention & Erasure",
    # Children's Data Safeguards (DPDP Sec 9)
    "S001014": "Significant Data Fiduciary Obligations",
    "S001223": "Significant Data Fiduciary Obligations",
    # Security Safeguards
    "S001518": "Security Safeguards",
    "S002129": "Security Safeguards",
    # Grievance Redressal
    "S000368": "Grievance Redressal",
    "S001919": "Grievance Redressal",
}

# Apply promotions to Compliant
for cid, category in PROMOTED_COMPLIANT.items():
    mask = df["clause_id"] == cid
    df.loc[mask, "verdict"] = "Compliant"
    df.loc[mask, "verdict_score"] = 1.0
    df.loc[mask, "dpdp_category"] = category
    df.loc[mask, "label_status"] = "HUMAN_AUDITED_GOLD"

# Correct false positive to Not Addressed
df.loc[df["clause_id"] == "S002023", "verdict"] = "Not Addressed"
df.loc[df["clause_id"] == "S002023", "verdict_score"] = 0.0
df.loc[df["clause_id"] == "S002023", "dpdp_category"] = "Not Applicable"

# Clean any residual NaN labels
df["verdict"] = df["verdict"].fillna("Not Addressed")
df.loc[df["verdict"] == "Not Addressed", "verdict_score"] = 0.0

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 60)
print("GOLD DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"Total dataset: {len(df)} clauses saved to '{OUTPUT_PATH}'\n")
print("Updated Ground-Truth Class Distribution:")
print(df["verdict"].value_counts().to_string())