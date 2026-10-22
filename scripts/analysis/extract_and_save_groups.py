"""
src/evaluation/extract_and_save_groups.py
Extract the original group assignment from fixed_folds_v1.json and save it.
"""

import json
from pathlib import Path
import pandas as pd


def extract_groups():
    """Extract groups from fixed_folds_v1.json."""
    
    # Load fixed folds
    folds_path = Path("datasets/folds/fixed_folds_v1.json")
    folds_data = json.loads(folds_path.read_text())
    
    # Get group assignment
    groups = folds_data["groups"]
    
    print(f"[INFO] Loaded fixed_folds_v1.json")
    print(f"[INFO] Groups: {len(set(groups))} unique, {sum(1 for g in set(groups) if groups.count(g) > 1)} multi-member")
    
    # Load corpus to get clause IDs
    corpus_path = Path("corpus/labeled_clauses_bootstrap.csv")
    df = pd.read_csv(corpus_path)
    
    assert len(df) == len(groups), f"Length mismatch: {len(df)} clauses, {len(groups)} group assignments"
    
    # Create clause_id → group_id mapping
    group_mapping = {}
    for idx, group_id in enumerate(groups):
        clause_id = df.iloc[idx].get("clause_id", idx)  # Use clause_id if exists, else index
        group_mapping[str(clause_id)] = int(group_id)
    
    # Save to groups_v1.json
    output_data = {
        "description": "Near-duplicate group assignments for labeled_clauses_bootstrap.csv",
        "source": "Extracted from fixed_folds_v1.json (generated with all-MiniLM-L6-v2)",
        "corpus": "labeled_clauses_bootstrap.csv",
        "n_samples": len(groups),
        "n_groups": len(set(groups)),
        "n_multi_member_groups": sum(1 for g in set(groups) if groups.count(g) > 1),
        "grouping_method": "all-MiniLM-L6-v2 context embeddings, cosine >= 0.90, connected components",
        "groups": groups,
        "clause_id_to_group_id": group_mapping
    }
    
    output_path = Path("datasets/folds/groups_v1.json")
    output_path.write_text(json.dumps(output_data, indent=2))
    
    print(f"[SUCCESS] Saved group assignment to {output_path}")
    print(f"[INFO] Statistics:")
    print(f"  Total samples: {len(groups)}")
    print(f"  Unique groups: {len(set(groups))}")
    print(f"  Multi-member groups: {sum(1 for g in set(groups) if groups.count(g) > 1)}")
    
    return output_path


if __name__ == "__main__":
    extract_groups()
