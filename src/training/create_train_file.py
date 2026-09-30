import json
from pathlib import Path
import numpy as np
import pandas as pd

SPLITS_DIR = Path("datasets/splits")
OUTPUT_PATH = SPLITS_DIR / "train_augmented_perfect.csv"
TAXONOMY_PATH = Path("corpus/dpdp_taxonomy_final.json")

train_df = pd.read_csv(SPLITS_DIR / "train.csv")
VALID_LABELS = ["Not Addressed", "Partially Compliant", "Compliant"]
train_df = train_df[train_df["verdict"].isin(VALID_LABELS)].copy()

df_na = train_df[train_df["verdict"] == "Not Addressed"].copy()
df_pc = train_df[train_df["verdict"] == "Partially Compliant"].copy()
df_c = train_df[train_df["verdict"] == "Compliant"].copy()

# Sample 200 Not Addressed to rebalance class ratio
df_na_sub = df_na.sample(n=200, random_state=42)

# Load real statutory anchor concepts from dpdp_taxonomy_final.json
tax_data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
tax_compliant_anchors = [
    f"{item['requirement']} {item['checkable_test']}" for item in tax_data
]

def augment_in_domain(base_df, anchor_texts, target_count, label):
    real_texts = list(base_df["clause_text"])
    rng = np.random.default_rng(42)
    
    rows = [{"clause_text": t, "verdict": label} for t in real_texts]
    
    # Prefix variations using legitimate consumer policy phrasings
    prefixes = [
        "In accordance with our user privacy policy, ",
        "To protect your account and personal data, ",
        "Under applicable data protection regulations, ",
        "As part of our commitment to user rights, ",
        "With respect to personal information handling, "
    ]
    
    while len(rows) < target_count:
        # 70% chance to sample and vary real policy clauses, 30% chance from statutory requirements
        if rng.random() < 0.70 and len(real_texts) > 0:
            base = rng.choice(real_texts)
        else:
            base = rng.choice(anchor_texts)
            
        p = rng.choice(prefixes)
        augmented_text = f"{p}{base[0].lower() + base[1:]}"
        rows.append({"clause_text": augmented_text, "verdict": label})
        
    return pd.DataFrame(rows[:target_count])

df_c_aug = augment_in_domain(df_c, tax_compliant_anchors, target_count=160, label="Compliant")
df_pc_aug = augment_in_domain(df_pc, tax_compliant_anchors, target_count=160, label="Partially Compliant")

perfect_df = pd.concat([df_na_sub, df_pc_aug, df_c_aug], ignore_index=True)
perfect_df = perfect_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
perfect_df = perfect_df[["clause_text", "verdict"]]
perfect_df.to_csv(OUTPUT_PATH, index=False)

print(f"[SUCCESS] Saved balanced statutory dataset to: {OUTPUT_PATH}")
print(perfect_df["verdict"].value_counts())