"""
src/evaluation/run_feature_experiments.py
Run feature variant experiments with consistent methodology.
"""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.feature_variants import FEATURE_EXTRACTORS
from src.training.train_classifier import train_and_evaluate_fold
from src.training.two_stage_classifier import train_two_stage_fold

OUTPUT_DIR = Path("reports/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_fixed_folds(seed):
    """Load pre-computed fixed fold assignments."""
    folds_path = Path("datasets/folds/fixed_folds_v1.json")
    
    if not folds_path.exists():
        raise FileNotFoundError(
            f"Fixed folds file not found: {folds_path}\n"
            f"Run: python src/evaluation/generate_fixed_folds.py"
        )
    
    folds_data = json.loads(folds_path.read_text())
    
    if str(seed) not in folds_data["seeds"]:
        raise ValueError(f"Seed {seed} not found in fixed folds. Available: {list(folds_data['seeds'].keys())}")
    
    return folds_data["seeds"][str(seed)], folds_data["groups"]


def run_single_seed_cv(df, taxonomy, feature_data, args, seed, two_stage=False):
    """Run 5-fold CV for a single seed with given features using FIXED FOLDS."""
    embeddings = feature_data["embeddings"]
    structured_features = feature_data["structured"]
    y_all = df["verdict"].values
    
    # Load fixed fold assignments
    fold_assignments, groups = load_fixed_folds(seed)
    groups = np.array(groups)
    
    print(f"[INFO] Using fixed folds from datasets/folds/fixed_folds_v1.json")
    print(f"[INFO] Seed {seed}: {len(fold_assignments)} folds with {len(np.unique(groups))} groups")
    
    fold_results = []
    all_true_labels = []
    all_pred_labels = []
    
    for fold_assignment in fold_assignments:
        fold_idx = fold_assignment["fold"]
        train_idx = np.array(fold_assignment["train"])
        test_idx = np.array(fold_assignment["test"])
        # Split data
        X_train_dict = {
            "embedding": embeddings[train_idx],
            "features": structured_features[train_idx]
        }
        X_test_dict = {
            "embedding": embeddings[test_idx],
            "features": structured_features[test_idx]
        }
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        # Train and evaluate
        if two_stage:
            result = train_two_stage_fold(X_train_dict, y_train, X_test_dict, y_test, inner_cv_splits=3)
        else:
            result = train_and_evaluate_fold(X_train_dict, y_train, X_test_dict, y_test, inner_cv_splits=3)
        
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
    parser = argparse.ArgumentParser(description="Feature Variant Experiments")
    parser.add_argument("--data", type=str, default="corpus/labeled_clauses_bootstrap.csv")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--features", type=str, required=True,
                        choices=list(FEATURE_EXTRACTORS.keys()) + ["two_stage_bge", "two_stage_e5", "two_stage_inlegal"],
                        help="Feature variant to test")
    parser.add_argument("--tag", type=str, required=True, help="Experiment tag")
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Determine taxonomy path
    corpus_path = Path(args.data)
    if "v2" in corpus_path.stem:
        taxonomy_path = Path("corpus/dpdp_taxonomy_v2.json")
    else:
        taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")
    
    # Load data
    print(f"[INFO] Loading {corpus_path}...")
    df = pd.read_csv(corpus_path)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    
    print(f"[INFO] Using taxonomy: {taxonomy_path}")
    print(f"\nClass distribution:")
    print(df["verdict"].value_counts().to_string())
    
    # Extract features
    print(f"\n[INFO] Extracting features: {args.features}")
    
    two_stage = args.features.startswith("two_stage_")
    if two_stage:
        # Two-stage uses pairing features from best encoder
        encoder_name = args.features.replace("two_stage_", "pairing_")
        feature_data = FEATURE_EXTRACTORS[encoder_name](df, taxonomy)
        description = feature_data["description"] + " [TWO-STAGE]"
    else:
        feature_data = FEATURE_EXTRACTORS[args.features](df, taxonomy)
        description = feature_data["description"]
    
    print(f"[INFO] Feature description: {description}")
    print(f"[INFO] Embedding dim: {feature_data['embeddings'].shape[1]}, Structured: {feature_data['structured'].shape[1]}")
    
    # Fixed folds will be loaded per-seed (groups are pre-computed)
    print(f"\n[INFO] Will use fixed folds from datasets/folds/fixed_folds_v1.json")
    
    # Run CV for each seed
    print(f"\n{'='*80}")
    print(f"RUNNING {len(seeds)} SEED(S) × 5 FOLDS = {len(seeds)*5} TOTAL EVALUATIONS")
    print(f"{'='*80}")
    
    all_fold_results = []
    all_class_reports = []
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n--- Seed {seed} ({seed_idx}/{len(seeds)}) ---")
        fold_results, class_report = run_single_seed_cv(
            df, taxonomy, feature_data, args, seed, two_stage=two_stage
        )
        all_fold_results.extend(fold_results)
        all_class_reports.append(class_report)
    
    # Aggregate metrics
    all_f1s = [r["macro_f1"] for r in all_fold_results]
    all_accs = [r["accuracy"] for r in all_fold_results]
    
    mean_f1 = float(np.mean(all_f1s))
    std_f1 = float(np.std(all_f1s))
    mean_acc = float(np.mean(all_accs))
    std_acc = float(np.std(all_accs))
    
    # Per-class metrics
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
    
    # Aggregate precision
    macro_precision = float(np.mean([report["macro avg"]["precision"] for report in all_class_reports]))
    macro_precision_std = float(np.std([report["macro avg"]["precision"] for report in all_class_reports]))
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY (All Seeds × All Folds)")
    print(f"{'='*80}")
    print(f"Accuracy:        {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro-F1:        {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Macro-Precision: {macro_precision:.4f} ± {macro_precision_std:.4f}")
    
    print(f"\nPer-Class F1:")
    for cls in ["Not Addressed", "Partially Compliant", "Compliant"]:
        m = metrics_by_class[cls]
        print(f"  {cls}: {m['f1']:.4f} ± {m['f1_std']:.4f}")
    
    # Save experiment
    git_commit = get_git_commit()
    
    # Load groups info from fixed folds
    folds_data = json.loads(Path("datasets/folds/fixed_folds_v1.json").read_text())
    n_multi_member = folds_data["metadata"]["n_multi_member_groups"]
    
    experiment_data = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "config": {
            "data_path": str(corpus_path),
            "taxonomy_path": str(taxonomy_path),
            "seeds": seeds,
            "feature_variant": args.features,
            "feature_description": description,
            "two_stage": two_stage,
            "n_multi_member_groups": int(n_multi_member),
        },
        "aggregate_metrics": {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "macro_f1_mean": mean_f1,
            "macro_f1_std": std_f1,
            "macro_precision_mean": macro_precision,
            "macro_precision_std": macro_precision_std,
        },
        "per_class_metrics": metrics_by_class,
        "per_fold_results": all_fold_results,
    }
    
    experiment_file = OUTPUT_DIR / f"{args.tag}.json"
    experiment_file.write_text(json.dumps(experiment_data, indent=2))
    print(f"\n[SUCCESS] Saved experiment to {experiment_file}")
    
    # Append to summary CSV
    summary_csv = OUTPUT_DIR / "summary.csv"
    summary_row = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit[:8],
        "features": args.features,
        "seeds": args.seeds,
        "grouped": True,
        "n_multi_member_groups": n_multi_member,
        "accuracy_mean": f"{mean_acc:.4f}",
        "accuracy_std": f"{std_acc:.4f}",
        "macro_f1_mean": f"{mean_f1:.4f}",
        "macro_f1_std": f"{std_f1:.4f}",
        "macro_precision_mean": f"{macro_precision:.4f}",
        "macro_precision_std": f"{macro_precision_std:.4f}",
        "C_f1": f"{metrics_by_class['Compliant']['f1']:.4f}",
        "PC_f1": f"{metrics_by_class['Partially Compliant']['f1']:.4f}",
    }
    
    df_summary = pd.DataFrame([summary_row])
    if summary_csv.exists():
        df_summary.to_csv(summary_csv, mode="a", header=False, index=False)
    else:
        df_summary.to_csv(summary_csv, mode="w", header=True, index=False)
    
    print(f"[SUCCESS] Appended to {summary_csv}")


if __name__ == "__main__":
    main()
