"""
src/evaluation/evaluate_5fold_cv.py
Stratified 5-Fold Cross-Validation per Research Paper Specification.

Implements leak-free threshold calibration with:
- Context-formatted string-level conditioning
- all-MiniLM-L6-v2 embeddings (384 dims, mean pooling, L2-normalized)
- 8 actionability features, scaled with RobustScaler
- LogisticRegression with specified hyperparameters
- Inner stratified CV for threshold calibration on training fold only
- Optional near-duplicate grouping to prevent leakage
- Multi-seed support for robustness testing
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import joblib

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.feature_extraction import extract_actionability_features
from src.training.train_classifier import train_and_evaluate_fold
from src.training.model_inputs import build_input_texts, model_inputs

# Default paths
DEFAULT_CORPUS_PATH = "corpus/labeled_clauses_bootstrap.csv"
DEFAULT_TAXONOMY_PATH = "corpus/dpdp_taxonomy_final.json"
OUTPUT_DIR = Path("reports")
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
MODELS_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def find_near_duplicate_groups(embeddings, threshold=0.90):
    """
    Find near-duplicate groups using cosine similarity.
    Returns group assignments where each connected component is a group.
    
    Args:
        embeddings: numpy array of L2-normalized embeddings (n_samples, n_dims)
        threshold: cosine similarity threshold (default 0.90)
        
    Returns:
        groups: numpy array of group IDs (n_samples,)
        n_multi_member: number of groups with size > 1
    """
    # Compute cosine similarity matrix (embeddings already L2-normalized)
    similarity_matrix = embeddings @ embeddings.T
    
    # Create adjacency matrix (edges where similarity >= threshold)
    adjacency = (similarity_matrix >= threshold).astype(int)
    
    # Find connected components
    n_components, labels = connected_components(
        csgraph=csr_matrix(adjacency),
        directed=False,
        return_labels=True
    )
    
    # Count multi-member groups
    unique, counts = np.unique(labels, return_counts=True)
    n_multi_member = np.sum(counts > 1)
    
    return labels, n_multi_member

def run_single_seed_cv(df, taxonomy, embedder, groups, args, seed):
    """Run 5-fold CV for a single seed."""
    tax_map = {item["taxonomy_id"]: item for item in taxonomy}
    
    # Build label-free input texts (clause_text only; MiniLM takes no prefix)
    context_texts = build_input_texts(model_inputs(df))
    
    # Extract actionability features
    action_features = extract_actionability_features(df["clause_text"].values)
    
    # Get class labels
    y_all = df["verdict"].values
    
    # Generate embeddings
    embeddings = embedder.encode(
        context_texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Choose cross-validator
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = skf.split(embeddings, y_all, groups)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = skf.split(embeddings, y_all)
    
    fold_results = []
    all_true_labels = []
    all_pred_labels = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        # Split data
        X_train_embedding = embeddings[train_idx]
        X_test_embedding = embeddings[test_idx]
        X_train_features = action_features[train_idx]
        X_test_features = action_features[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        # Prepare data dict
        X_train_dict = {"embedding": X_train_embedding, "features": X_train_features}
        X_test_dict = {"embedding": X_test_embedding, "features": X_test_features}
        
        # Train and calibrate
        result = train_and_evaluate_fold(X_train_dict, y_train, X_test_dict, y_test, inner_cv_splits=3)
        
        # Get predictions
        test_preds = result["predictions"]
        
        # Compute metrics
        f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test, test_preds)
        
        all_true_labels.extend(y_test)
        all_pred_labels.extend(test_preds)
        
        fold_results.append({
            "fold": fold_idx,
            "seed": seed,
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "tau_compliant": float(result["tau_compliant"]),
            "tau_partial": float(result["tau_partial"]),
        })
    
    # Compute per-class metrics
    class_report_dict = classification_report(
        all_true_labels, all_pred_labels,
        labels=["Not Addressed", "Partially Compliant", "Compliant"],
        output_dict=True,
        zero_division=0
    )
    
    return fold_results, class_report_dict


def main():
    parser = argparse.ArgumentParser(description="5-Fold CV Evaluation with Enhanced Options")
    parser.add_argument("--data", type=str, default=DEFAULT_CORPUS_PATH,
                        help="Path to labeled clauses CSV")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds (e.g., '42,43,44')")
    parser.add_argument("--group-near-dupes", action="store_true",
                        help="Use StratifiedGroupKFold to keep near-duplicates in same fold")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run name for experiment tracking")
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Generate tag if not provided
    if args.tag is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.tag = f"run_{timestamp}"
    
    # Determine taxonomy path based on data file
    corpus_path = Path(args.data)
    if "v2" in corpus_path.stem:
        taxonomy_path = Path("corpus/dpdp_taxonomy_v2.json")
    else:
        taxonomy_path = Path(DEFAULT_TAXONOMY_PATH)
    
    # Load data
    df = pd.read_csv(corpus_path)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    
    print(f"[INFO] Loaded {len(df)} clauses from {corpus_path}")
    print(f"[INFO] Using taxonomy: {taxonomy_path}")
    print("\nClass distribution:")
    print(df["verdict"].value_counts().to_string())
    
    actual_dist = df["verdict"].value_counts().to_dict()
    
    # Load embedder
    print("\n[INFO] Loading all-MiniLM-L6-v2 embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Handle near-duplicate grouping
    groups = None
    n_multi_member_groups = 0
    if args.group_near_dupes:
        print("\n[INFO] Computing near-duplicate groups (cosine >= 0.90)...")
        # Pre-compute embeddings for grouping
        tax_map = {item["taxonomy_id"]: item for item in taxonomy}
        context_texts = build_input_texts(model_inputs(df))
        embeddings_for_grouping = embedder.encode(
            context_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        groups, n_multi_member_groups = find_near_duplicate_groups(embeddings_for_grouping, threshold=0.90)
        print(f"[INFO] Found {len(np.unique(groups))} groups, {n_multi_member_groups} with size > 1")
    
    # Run CV for each seed
    print(f"\n{'='*80}")
    print(f"RUNNING {len(seeds)} SEED(S) × 5 FOLDS = {len(seeds)*5} TOTAL EVALUATIONS")
    print(f"{'='*80}")
    
    all_fold_results = []
    all_class_reports = []
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n--- Seed {seed} ({seed_idx}/{len(seeds)}) ---")
        fold_results, class_report = run_single_seed_cv(df, taxonomy, embedder, groups, args, seed)
        all_fold_results.extend(fold_results)
        all_class_reports.append(class_report)
    
    # Aggregate metrics across all folds and seeds
    all_f1s = [r["macro_f1"] for r in all_fold_results]
    all_accs = [r["accuracy"] for r in all_fold_results]
    
    mean_f1 = float(np.mean(all_f1s))
    std_f1 = float(np.std(all_f1s))
    mean_acc = float(np.mean(all_accs))
    std_acc = float(np.std(all_accs))
    
    # Aggregate per-class metrics (average across seeds)
    metrics_by_class = {}
    for cls in ["Not Addressed", "Partially Compliant", "Compliant"]:
        precisions = [report[cls]["precision"] for report in all_class_reports]
        recalls = [report[cls]["recall"] for report in all_class_reports]
        f1s = [report[cls]["f1-score"] for report in all_class_reports]
        
        metrics_by_class[cls] = {
            "precision": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "f1": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }
    
    # Compute weighted and macro precision
    weighted_precision = float(np.mean([report["weighted avg"]["precision"] for report in all_class_reports]))
    weighted_precision_std = float(np.std([report["weighted avg"]["precision"] for report in all_class_reports]))
    macro_precision = float(np.mean([report["macro avg"]["precision"] for report in all_class_reports]))
    macro_precision_std = float(np.std([report["macro avg"]["precision"] for report in all_class_reports]))
    
    # Compute weighted F1
    weighted_f1 = float(np.mean([report["weighted avg"]["f1-score"] for report in all_class_reports]))
    weighted_f1_std = float(np.std([report["weighted avg"]["f1-score"] for report in all_class_reports]))
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY (All Seeds × All Folds)")
    print(f"{'='*80}")
    print(f"Accuracy:          {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro-F1:          {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f} ± {weighted_f1_std:.4f}")
    print(f"Macro-Precision:   {macro_precision:.4f} ± {macro_precision_std:.4f}")
    print(f"Weighted-Precision: {weighted_precision:.4f} ± {weighted_precision_std:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for cls in ["Not Addressed", "Partially Compliant", "Compliant"]:
        m = metrics_by_class[cls]
        print(f"  {cls}:")
        print(f"    Precision: {m['precision']:.4f} ± {m['precision_std']:.4f}")
        print(f"    Recall:    {m['recall']:.4f} ± {m['recall_std']:.4f}")
        print(f"    F1:        {m['f1']:.4f} ± {m['f1_std']:.4f}")
    
    # Get git commit
    git_commit = get_git_commit()
    
    # Save experiment results
    experiment_data = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "config": {
            "data_path": str(corpus_path),
            "taxonomy_path": str(taxonomy_path),
            "seeds": seeds,
            "group_near_dupes": args.group_near_dupes,
            "n_multi_member_groups": int(n_multi_member_groups) if groups is not None else None,
        },
        "data_distribution": actual_dist,
        "aggregate_metrics": {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "macro_f1_mean": mean_f1,
            "macro_f1_std": std_f1,
            "weighted_f1_mean": weighted_f1,
            "weighted_f1_std": weighted_f1_std,
            "macro_precision_mean": macro_precision,
            "macro_precision_std": macro_precision_std,
            "weighted_precision_mean": weighted_precision,
            "weighted_precision_std": weighted_precision_std,
        },
        "per_class_metrics": metrics_by_class,
        "per_fold_results": all_fold_results,
    }
    
    experiment_file = EXPERIMENTS_DIR / f"{args.tag}.json"
    experiment_file.write_text(json.dumps(experiment_data, indent=2))
    print(f"\n[SUCCESS] Saved experiment to {experiment_file}")
    
    # Append to summary CSV
    summary_csv = EXPERIMENTS_DIR / "summary.csv"
    summary_row = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit[:8],
        "data": corpus_path.stem,
        "seeds": args.seeds,
        "grouped": args.group_near_dupes,
        "n_multi_member_groups": n_multi_member_groups if groups is not None else 0,
        "accuracy_mean": f"{mean_acc:.4f}",
        "accuracy_std": f"{std_acc:.4f}",
        "macro_f1_mean": f"{mean_f1:.4f}",
        "macro_f1_std": f"{std_f1:.4f}",
        "weighted_f1_mean": f"{weighted_f1:.4f}",
        "weighted_f1_std": f"{weighted_f1_std:.4f}",
        "NA_f1": f"{metrics_by_class['Not Addressed']['f1']:.4f}",
        "PC_f1": f"{metrics_by_class['Partially Compliant']['f1']:.4f}",
        "C_f1": f"{metrics_by_class['Compliant']['f1']:.4f}",
    }
    
    # Append to CSV (create if doesn't exist)
    df_summary = pd.DataFrame([summary_row])
    if summary_csv.exists():
        df_summary.to_csv(summary_csv, mode="a", header=False, index=False)
    else:
        df_summary.to_csv(summary_csv, mode="w", header=True, index=False)
    
    print(f"[SUCCESS] Appended to {summary_csv}")
    
    return experiment_data


if __name__ == "__main__":
    main()
