"""
src/evaluation/regenerate_confirm_folds.py
Regenerate confirmation fold assignments using IDENTICAL groups from groups_v1.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


def regenerate_confirm_folds(seeds=[45, 46, 47, 48, 49], n_splits=5):
    """Regenerate confirmation folds using saved group assignment."""
    
    # Load data
    corpus_path = Path("corpus/labeled_clauses_bootstrap.csv")
    df = pd.read_csv(corpus_path)
    
    print(f"[INFO] Loaded {len(df)} clauses from {corpus_path}")
    
    # Load ORIGINAL groups from groups_v1.json
    groups_path = Path("datasets/folds/groups_v1.json")
    if not groups_path.exists():
        raise FileNotFoundError(f"Groups file not found: {groups_path}")
    
    groups_data = json.loads(groups_path.read_text())
    groups = np.array(groups_data["groups"])
    
    print(f"[INFO] Loaded groups from {groups_path}")
    print(f"[INFO] Groups: {groups_data['n_groups']} unique, {groups_data['n_multi_member_groups']} multi-member")
    
    assert len(groups) == len(df), f"Length mismatch: {len(groups)} groups, {len(df)} clauses"
    
    # ASSERT: Groups should be identical to fixed_folds_v1
    fixed_folds = json.loads(Path("datasets/folds/fixed_folds_v1.json").read_text())
    fixed_groups = np.array(fixed_folds["groups"])
    assert np.array_equal(groups, fixed_groups), "Groups do not match fixed_folds_v1!"
    print(f"[SUCCESS] Verified groups match fixed_folds_v1.json")
    
    # Get labels
    y = df["verdict"].values
    
    # Generate folds for each seed
    all_seeds_data = {}
    
    for seed in seeds:
        print(f"\n[INFO] Generating folds for seed {seed}...")
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        fold_assignments = []
        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(df, y, groups)):
            fold_assignments.append({
                "fold": fold_idx,
                "train": train_idx.tolist(),
                "test": test_idx.tolist(),
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_groups": len(set(groups[train_idx])),
                "test_groups": len(set(groups[test_idx]))
            })
            
            # Verify no group appears in both train and test
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = train_groups & test_groups
            assert len(overlap) == 0, f"Seed {seed}, Fold {fold_idx}: Groups overlap! {overlap}"
        
        all_seeds_data[str(seed)] = fold_assignments
        
        print(f"[SUCCESS] Seed {seed}: {n_splits} folds generated, no group overlap")
    
    # Save to file
    output_data = {
        "description": "Confirmation fold assignments using IDENTICAL groups from fixed_folds_v1",
        "corpus": "labeled_clauses_bootstrap.csv",
        "n_clauses": len(df),
        "n_splits": n_splits,
        "groups_source": "datasets/folds/groups_v1.json",
        "n_groups": groups_data["n_groups"],
        "n_multi_member_groups": groups_data["n_multi_member_groups"],
        "grouping_method": groups_data["grouping_method"],
        "seeds": all_seeds_data
    }
    
    output_path = Path("datasets/folds/confirm_folds_v1.json")
    output_path.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n[SUCCESS] Saved confirmation folds to {output_path}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Total folds: {len(seeds)} seeds × {n_splits} folds = {len(seeds) * n_splits} folds")
    print(f"[INFO] Groups: {groups_data['n_groups']} unique (IDENTICAL to fixed_folds_v1)")
    
    return output_path


if __name__ == "__main__":
    regenerate_confirm_folds(
        seeds=[45, 46, 47, 48, 49],
        n_splits=5
    )
