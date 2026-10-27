"""
src/evaluation/generate_fixed_folds.py
Generate fixed cross-validation fold assignments using near-duplicate grouping.

Uses all-MiniLM-L6-v2 embeddings for grouping regardless of which encoder
experiments use. This ensures all experiments use identical train/test splits.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.model_inputs import build_input_texts, model_inputs


def find_near_duplicate_groups(embeddings, threshold=0.90):
    """Find near-duplicate groups using cosine similarity."""
    similarity_matrix = embeddings @ embeddings.T
    adjacency = (similarity_matrix >= threshold).astype(int)
    n_components, labels = connected_components(
        csgraph=csr_matrix(adjacency),
        directed=False,
        return_labels=True
    )
    unique, counts = np.unique(labels, return_counts=True)
    n_multi_member = np.sum(counts > 1)
    return labels, n_multi_member


def main():
    # Load bootstrap data
    corpus_path = Path("corpus/labeled_clauses_bootstrap.csv")
    taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")
    
    print(f"[INFO] Loading {corpus_path}...")
    df = pd.read_csv(corpus_path)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    
    print(f"[INFO] Loaded {len(df)} clauses")
    print("\nClass distribution:")
    print(df["verdict"].value_counts().to_string())
    
    # Label-free texts for grouping. NOTE: the committed fixed_folds_v1.json was
    # grouped on the old label-leaking context texts; re-running this script
    # would produce different groups, so the committed file is kept as-is.
    context_texts = build_input_texts(model_inputs(df))
    
    # Generate embeddings with all-MiniLM-L6-v2
    print(f"\n[INFO] Generating all-MiniLM-L6-v2 embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        context_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Find near-duplicate groups
    print(f"\n[INFO] Computing near-duplicate groups (cosine >= 0.90)...")
    groups, n_multi_member = find_near_duplicate_groups(embeddings, threshold=0.90)
    n_unique_groups = len(np.unique(groups))
    
    print(f"[INFO] Found {n_unique_groups} groups, {n_multi_member} with size > 1")
    
    # Generate fold assignments for each seed
    y = df["verdict"].values
    seeds = [42, 43, 44]
    
    fold_assignments = {
        "metadata": {
            "corpus": str(corpus_path),
            "n_samples": len(df),
            "n_groups": int(n_unique_groups),
            "n_multi_member_groups": int(n_multi_member),
            "grouping_method": "all-MiniLM-L6-v2, cosine >= 0.90, connected components",
            "n_folds": 5,
            "seeds": seeds,
        },
        "groups": groups.tolist(),
        "seeds": {}
    }
    
    for seed in seeds:
        print(f"\n[INFO] Generating folds for seed {seed}...")
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        
        fold_indices = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, y, groups), 1):
            fold_indices.append({
                "fold": fold_idx,
                "train": train_idx.tolist(),
                "test": test_idx.tolist(),
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            })
            
            # Check class distribution
            train_dist = pd.Series(y[train_idx]).value_counts().to_dict()
            test_dist = pd.Series(y[test_idx]).value_counts().to_dict()
            
            print(f"  Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
            print(f"    Train dist: {train_dist}")
            print(f"    Test dist: {test_dist}")
        
        fold_assignments["seeds"][str(seed)] = fold_indices
    
    # Save to file
    output_dir = Path("datasets/folds")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fixed_folds_v1.json"
    
    output_path.write_text(json.dumps(fold_assignments, indent=2))
    print(f"\n[SUCCESS] Saved fixed folds to {output_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Total groups: {n_unique_groups}")
    print(f"Multi-member groups: {n_multi_member}")
    print(f"Seeds: {seeds}")
    print(f"Folds per seed: 5")
    print(f"Total fold assignments: {len(seeds) * 5}")


if __name__ == "__main__":
    main()
