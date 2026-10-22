"""
src/evaluation/generate_confirm_folds.py
Generate NEW fold assignments with seeds 45-49 for confirmation testing.

Uses same near-duplicate grouping (all-MiniLM-L6-v2, cosine >= 0.90) but
completely different train/test splits to confirm model generalization.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def compute_near_duplicate_groups(df, similarity_threshold=0.90):
    """
    Compute near-duplicate groups using all-MiniLM-L6-v2 embeddings.
    Same algorithm as fixed_folds_v1.json.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    clause_texts = df["clause_text"].tolist()
    print(f"[INFO] Computing embeddings for {len(clause_texts)} clauses...")
    embeddings = model.encode(clause_texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"[INFO] Computing pairwise cosine similarities...")
    similarities = cosine_similarity(embeddings)
    
    # Build adjacency list
    adjacency = defaultdict(set)
    for i in range(len(clause_texts)):
        for j in range(i + 1, len(clause_texts)):
            if similarities[i, j] >= similarity_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
    
    # Find connected components (groups)
    visited = set()
    groups = []
    
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for i in range(len(clause_texts)):
        if i not in visited:
            component = []
            dfs(i, component)
            groups.append(sorted(component))
    
    # Assign group IDs
    group_ids = np.zeros(len(clause_texts), dtype=int)
    for group_id, component in enumerate(groups):
        for idx in component:
            group_ids[idx] = group_id
    
    multi_member_groups = [g for g in groups if len(g) > 1]
    
    print(f"[INFO] Found {len(groups)} total groups")
    print(f"[INFO] {len(multi_member_groups)} groups have multiple members")
    
    return group_ids, groups


def generate_confirm_folds(seeds=[45, 46, 47, 48, 49], similarity_threshold=0.90, n_splits=5):
    """Generate confirmation fold assignments with new seeds."""
    
    # Load data
    corpus_path = Path("corpus/labeled_clauses_bootstrap.csv")
    df = pd.read_csv(corpus_path)
    
    print(f"[INFO] Loaded {len(df)} clauses from {corpus_path}")
    print(f"[INFO] Label distribution:")
    print(df["verdict"].value_counts())
    
    # Compute groups (same as fixed_folds_v1)
    print(f"\n[INFO] Computing near-duplicate groups (threshold={similarity_threshold})...")
    group_ids, groups = compute_near_duplicate_groups(df, similarity_threshold)
    
    # Store group information
    group_metadata = {
        "total_groups": len(groups),
        "multi_member_groups": sum(1 for g in groups if len(g) > 1),
        "singleton_groups": sum(1 for g in groups if len(g) == 1),
        "largest_group_size": max(len(g) for g in groups),
        "groups": [{"group_id": i, "member_indices": g} for i, g in enumerate(groups)]
    }
    
    # Generate folds for each seed
    y = df["verdict"].values
    all_seeds_data = {}
    
    for seed in seeds:
        print(f"\n[INFO] Generating folds for seed {seed}...")
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        fold_assignments = []
        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(df, y, group_ids)):
            fold_assignments.append({
                "fold": fold_idx,
                "train": train_idx.tolist(),
                "test": test_idx.tolist(),
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_groups": len(set(group_ids[train_idx])),
                "test_groups": len(set(group_ids[test_idx]))
            })
            
            # Verify no group appears in both train and test
            train_groups = set(group_ids[train_idx])
            test_groups = set(group_ids[test_idx])
            overlap = train_groups & test_groups
            assert len(overlap) == 0, f"Seed {seed}, Fold {fold_idx}: Groups overlap! {overlap}"
        
        all_seeds_data[str(seed)] = fold_assignments
        
        print(f"[SUCCESS] Seed {seed}: {n_splits} folds generated")
    
    # Save to file
    output_data = {
        "description": "Confirmation fold assignments for model validation on unseen splits",
        "corpus": "labeled_clauses_bootstrap.csv",
        "n_clauses": len(df),
        "n_splits": n_splits,
        "similarity_threshold": similarity_threshold,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "group_metadata": group_metadata,
        "seeds": all_seeds_data
    }
    
    output_path = Path("datasets/folds/confirm_folds_v1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n[SUCCESS] Saved confirmation folds to {output_path}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Total folds: {len(seeds)} seeds × {n_splits} folds = {len(seeds) * n_splits} folds")
    
    return output_path


if __name__ == "__main__":
    generate_confirm_folds(
        seeds=[45, 46, 47, 48, 49],
        similarity_threshold=0.90,
        n_splits=5
    )
