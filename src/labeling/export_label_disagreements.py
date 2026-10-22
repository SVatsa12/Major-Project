"""
src/labeling/export_label_disagreements.py
Export label disagreements between v1 and v2 corpora for blind review.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Paths
V1_PATH = Path("corpus/labeled_clauses_bootstrap.csv")
V2_PATH = Path("corpus/labeled_clauses_v2.csv")
OUTPUT_DIR = Path("reports/label_review")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLIND_OUTPUT = OUTPUT_DIR / "v1_v2_disagreements_blind.csv"
KEY_OUTPUT = OUTPUT_DIR / "v1_v2_disagreements_key.json"


def main():
    # Load both datasets
    print(f"[INFO] Loading {V1_PATH}...")
    df_v1 = pd.read_csv(V1_PATH)
    
    print(f"[INFO] Loading {V2_PATH}...")
    df_v2 = pd.read_csv(V2_PATH)
    
    # Join on clause_id
    print("[INFO] Joining on clause_id...")
    merged = df_v1.merge(
        df_v2,
        on="clause_id",
        suffixes=("_v1", "_v2"),
        how="inner"
    )
    
    print(f"[INFO] Found {len(merged)} clauses in both datasets")
    
    # Find disagreements
    disagreements = merged[merged["verdict_v1"] != merged["verdict_v2"]].copy()
    n_disagreements = len(disagreements)
    
    print(f"\n[RESULT] Found {n_disagreements} label disagreements ({n_disagreements/len(merged)*100:.1f}%)")
    
    if n_disagreements == 0:
        print("[INFO] No disagreements found. Nothing to export.")
        return
    
    # Load taxonomy to get rule text
    taxonomy_v1 = json.loads(Path("corpus/dpdp_taxonomy_final.json").read_text(encoding="utf-8"))
    tax_map_v1 = {item["taxonomy_id"]: item for item in taxonomy_v1}
    
    # Create blind review dataset
    blind_df = disagreements[[
        "clause_id",
        "clause_text_v1",  # Should be identical between v1/v2
        "best_match_taxonomy_id_v1",
    ]].copy()
    
    # Add rule text from taxonomy
    blind_df["rule_text"] = blind_df["best_match_taxonomy_id_v1"].apply(
        lambda tid: tax_map_v1.get(tid, {}).get("requirement", "")
    )
    
    # Rename columns for clarity
    blind_df.columns = ["clause_id", "clause_text", "taxonomy_id", "rule_text"]
    
    # Add empty columns for review
    blind_df["reviewer_A"] = ""
    blind_df["reviewer_B"] = ""
    blind_df["final_label"] = ""
    
    # Shuffle for blind review
    blind_df = blind_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save blind CSV
    blind_df.to_csv(BLIND_OUTPUT, index=False)
    print(f"[SUCCESS] Saved blind review dataset to {BLIND_OUTPUT}")
    
    # Create key mapping
    key_data = []
    for _, row in disagreements.iterrows():
        key_data.append({
            "clause_id": row["clause_id"],
            "v1_label": row["verdict_v1"],
            "v2_label": row["verdict_v2"],
            "clause_text_preview": row["clause_text_v1"][:100] + "..." if len(row["clause_text_v1"]) > 100 else row["clause_text_v1"],
        })
    
    # Save key file
    KEY_OUTPUT.write_text(json.dumps(key_data, indent=2))
    print(f"[SUCCESS] Saved key file to {KEY_OUTPUT}")
    
    # Print disagreement summary by label pair
    print("\nDisagreement breakdown:")
    disagreement_summary = disagreements.groupby(["verdict_v1", "verdict_v2"]).size().sort_values(ascending=False)
    for (v1, v2), count in disagreement_summary.items():
        print(f"  {v1} → {v2}: {count}")


if __name__ == "__main__":
    main()
