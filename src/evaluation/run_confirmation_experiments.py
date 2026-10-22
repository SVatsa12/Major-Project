"""
src/evaluation/run_confirmation_experiments.py
Evaluate variants on NEW fold assignments (seeds 45-49) to confirm generalization.

Variants to test:
1. fixed_baseline_v1: MiniLM + 8 structured
2. fixed_A_e5: E5 + 8 structured
3. incr_1_all_rules: E5 + 8 + all-rules similarity (WINNER)
4. allrules_desc_only: Same as #3 but explicit about using only statutory text

Reports per-fold performance for paired comparison.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.feature_variants import extract_encoder_swap_features
from src.evaluation.incremental_features import extract_all_rules_similarity


OUTPUT_DIR = Path("reports/experiments/confirmation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_confirm_folds(seed):
    """Load confirmation fold assignments for given seed."""
    folds_path = Path("datasets/folds/confirm_folds_v1.json")
    if not folds_path.exists():
        raise FileNotFoundError(f"Confirmation folds not found: {folds_path}")
    
    folds_data = json.loads(folds_path.read_text())
    if str(seed) not in folds_data["seeds"]:
        raise ValueError(f"Seed {seed} not in confirmation folds")
    
    # Return fold assignments only (group metadata not needed for execution)
    return folds_data["seeds"][str(seed)], None


def run_single_fold(X_train, y_train, X_test, y_test, fold_idx, seed):
    """Train and evaluate on a single fold."""
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    report_dict = classification_report(
        y_test, y_pred,
        labels=["Not Addressed", "Partially Compliant", "Compliant"],
        output_dict=True,
        zero_division=0
    )
    
    return {
        "fold": fold_idx,
        "seed": seed,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "not_addressed_f1": float(report_dict["Not Addressed"]["f1-score"]),
        "partial_f1": float(report_dict["Partially Compliant"]["f1-score"]),
        "compliant_precision": float(report_dict["Compliant"]["precision"]),
        "compliant_recall": float(report_dict["Compliant"]["recall"]),
        "compliant_f1": float(report_dict["Compliant"]["f1-score"]),
    }


def run_confirmation(variant_name, all_features, df, seeds):
    """Run confirmation evaluation for a variant across all seeds."""
    
    y_all = df["verdict"].values
    
    all_fold_results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        fold_assignments, _ = load_confirm_folds(seed)
        
        for fold_assignment in fold_assignments:
            fold_idx = fold_assignment["fold"]
            train_idx = np.array(fold_assignment["train"])
            test_idx = np.array(fold_assignment["test"])
            
            X_train = all_features[train_idx]
            X_test = all_features[test_idx]
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            
            fold_result = run_single_fold(X_train, y_train, X_test, y_test, fold_idx, seed)
            all_fold_results.append(fold_result)
            
            print(f"  Fold {fold_idx}: F1={fold_result['macro_f1']:.4f}, Acc={fold_result['accuracy']:.4f}")
    
    # Aggregate
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall", 
               "not_addressed_f1", "partial_f1", 
               "compliant_precision", "compliant_recall", "compliant_f1"]
    
    aggregates = {}
    for metric in metrics:
        values = [r[metric] for r in all_fold_results]
        aggregates[f"{metric}_mean"] = float(np.mean(values))
        aggregates[f"{metric}_std"] = float(np.std(values))
    
    return all_fold_results, aggregates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, 
                       choices=["baseline", "fixed_A_e5", "all_rules", "allrules_desc_only"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", default="45,46,47,48,49")
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Load data
    corpus_path = Path("corpus/labeled_clauses_bootstrap.csv")
    taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")
    
    df = pd.read_csv(corpus_path)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    
    print(f"[INFO] Loaded {len(df)} clauses")
    print(f"[INFO] Variant: {args.variant}")
    print(f"[INFO] Confirmation seeds: {seeds}")
    
    # Extract features based on variant
    if args.variant == "baseline":
        # MiniLM + 8 structured
        print("\n[INFO] Extracting baseline features (MiniLM + 8 structured)...")
        feature_data = extract_encoder_swap_features(df, taxonomy, "sentence-transformers/all-MiniLM-L6-v2")
        all_features = np.hstack([feature_data["embeddings"], feature_data["structured"]])
        description = "MiniLM clause + 8 structured"
    
    elif args.variant == "fixed_A_e5":
        # E5 + 8 structured
        print("\n[INFO] Extracting E5 features (E5 + 8 structured)...")
        feature_data = extract_encoder_swap_features(df, taxonomy, "intfloat/e5-base-v2")
        all_features = np.hstack([feature_data["embeddings"], feature_data["structured"]])
        description = "E5 clause + 8 structured"
    
    elif args.variant == "all_rules":
        # E5 + 8 + all-rules (WINNER)
        print("\n[INFO] Extracting winner features (E5 + 8 + all-rules)...")
        feature_data = extract_encoder_swap_features(df, taxonomy, "intfloat/e5-base-v2")
        all_rules_sim = extract_all_rules_similarity(df, taxonomy, "intfloat/e5-base-v2", rule_text_mode="full")
        all_features = np.hstack([feature_data["embeddings"], feature_data["structured"], all_rules_sim])
        description = "E5 clause + 8 structured + all-rules similarity"
    
    elif args.variant == "allrules_desc_only":
        # E5 + 8 + all-rules (desc_only mode - same as full for current taxonomy)
        print("\n[INFO] Extracting desc_only features (E5 + 8 + all-rules statutory only)...")
        feature_data = extract_encoder_swap_features(df, taxonomy, "intfloat/e5-base-v2")
        all_rules_sim = extract_all_rules_similarity(df, taxonomy, "intfloat/e5-base-v2", rule_text_mode="desc_only")
        all_features = np.hstack([feature_data["embeddings"], feature_data["structured"], all_rules_sim])
        description = "E5 clause + 8 structured + all-rules similarity (statutory requirement only)"
    
    print(f"[INFO] Total features: {all_features.shape[1]}")
    
    # Run confirmation evaluation
    print(f"\n{'='*80}")
    print(f"CONFIRMATION EVALUATION: {len(seeds)} seeds × 5 folds = {len(seeds) * 5} folds")
    print(f"{'='*80}")
    
    fold_results, aggregates = run_confirmation(args.variant, all_features, df, seeds)
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*80}")
    print(f"Accuracy:        {aggregates['accuracy_mean']:.4f} ± {aggregates['accuracy_std']:.4f}")
    print(f"Macro-F1:        {aggregates['macro_f1_mean']:.4f} ± {aggregates['macro_f1_std']:.4f}")
    print(f"Macro-Precision: {aggregates['macro_precision_mean']:.4f} ± {aggregates['macro_precision_std']:.4f}")
    print(f"Macro-Recall:    {aggregates['macro_recall_mean']:.4f} ± {aggregates['macro_recall_std']:.4f}")
    
    print(f"\nPer-Class F1:")
    print(f"  Not Addressed:       {aggregates['not_addressed_f1_mean']:.4f} ± {aggregates['not_addressed_f1_std']:.4f}")
    print(f"  Partially Compliant: {aggregates['partial_f1_mean']:.4f} ± {aggregates['partial_f1_std']:.4f}")
    print(f"  Compliant:           {aggregates['compliant_f1_mean']:.4f} ± {aggregates['compliant_f1_std']:.4f}")
    
    print(f"\nCompliant Precision/Recall:")
    print(f"  Precision: {aggregates['compliant_precision_mean']:.4f} ± {aggregates['compliant_precision_std']:.4f}")
    print(f"  Recall:    {aggregates['compliant_recall_mean']:.4f} ± {aggregates['compliant_recall_std']:.4f}")
    
    # Save results
    experiment_data = {
        "tag": args.tag,
        "variant": args.variant,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": seeds,
            "n_features": all_features.shape[1],
            "fold_type": "confirmation",
            "total_folds": len(fold_results)
        },
        "aggregates": aggregates,
        "per_fold_results": fold_results
    }
    
    output_file = OUTPUT_DIR / f"{args.tag}.json"
    output_file.write_text(json.dumps(experiment_data, indent=2))
    print(f"\n[SUCCESS] Saved to {output_file}")


if __name__ == "__main__":
    main()
