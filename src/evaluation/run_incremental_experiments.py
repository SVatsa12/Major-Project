"""
src/evaluation/run_incremental_experiments.py
Test incremental feature additions starting from fixed_A_e5 baseline.
"""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.feature_variants import extract_encoder_swap_features
from src.evaluation.incremental_features import extract_all_rules_similarity, extract_rubric_features
from src.training.feature_extraction import extract_actionability_features
from src.training.model_inputs import model_inputs
from src.evaluation.fold_metrics import fold_metrics

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARNING] LightGBM not installed - skipping LGBM experiments")

OUTPUT_DIR = Path("reports/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return "unknown"


def load_fixed_folds(seed):
    """Load pre-computed fixed fold assignments."""
    folds_path = Path("datasets/folds/fixed_folds_v1.json")
    if not folds_path.exists():
        raise FileNotFoundError(f"Fixed folds not found: {folds_path}")
    
    folds_data = json.loads(folds_path.read_text())
    if str(seed) not in folds_data["seeds"]:
        raise ValueError(f"Seed {seed} not in fixed folds")
    
    return folds_data["seeds"][str(seed)], folds_data["groups"]


def train_with_config(X_train, y_train, X_val, y_val, clf_type="logreg", C=1.0, n_estimators=100):
    """Train classifier with given config."""
    if clf_type == "logreg":
        clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    elif clf_type == "linearsvc":
        base_clf = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
        clf = CalibratedClassifierCV(base_clf, cv=3)
    elif clf_type == "lightgbm" and HAS_LIGHTGBM:
        # Compute class weights
        classes, counts = np.unique(y_train, return_counts=True)
        weights = len(y_train) / (len(classes) * counts)
        class_weight_dict = dict(zip(classes, weights))
        
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            class_weight=class_weight_dict,
            random_state=42,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")
    
    clf.fit(X_train, y_train)
    
    # Evaluate on validation
    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    
    return clf, val_f1


def run_single_seed_cv(df, all_features, args, seed, clf_configs):
    """Run 5-fold CV for a single seed with classifier tuning."""
    y_all = df["verdict"].values
    
    # Load fixed folds
    fold_assignments, groups = load_fixed_folds(seed)
    
    fold_results = []
    all_true_labels = []
    all_pred_labels = []
    
    for fold_assignment in fold_assignments:
        fold_idx = fold_assignment["fold"]
        train_idx = np.array(fold_assignment["train"])
        test_idx = np.array(fold_assignment["test"])
        
        # Split features
        X_train_raw = all_features[train_idx]
        X_test_raw = all_features[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Inner CV for classifier tuning (use 20% of train as validation)
        val_size = int(0.2 * len(X_train))
        X_train_inner = X_train[:-val_size]
        y_train_inner = y_train[:-val_size]
        X_val_inner = X_train[-val_size:]
        y_val_inner = y_train[-val_size:]
        
        # Try all classifier configs
        best_clf = None
        best_val_f1 = 0
        
        for config in clf_configs:
            try:
                clf, val_f1 = train_with_config(
                    X_train_inner, y_train_inner,
                    X_val_inner, y_val_inner,
                    clf_type=config["type"],
                    C=config.get("C", 1.0),
                    n_estimators=config.get("n_estimators", 100)
                )
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_clf = clf
            except Exception as e:
                print(f"[WARNING] Config {config} failed: {e}")
                continue
        
        if best_clf is None:
            # Fallback to default
            best_clf, _ = train_with_config(X_train_inner, y_train_inner, X_val_inner, y_val_inner)
        
        # Retrain on full training set
        best_clf.fit(X_train, y_train)
        
        # Predict on test
        test_preds = best_clf.predict(X_test)
        
        all_true_labels.extend(y_test)
        all_pred_labels.extend(test_preds)
        
        fold_results.append({
            "fold": fold_idx,
            "seed": seed,
            **fold_metrics(y_test, test_preds),
        })
    
    # Compute per-class metrics
    class_report_dict = classification_report(
        all_true_labels, all_pred_labels,
        labels=["Not Addressed", "Partially Compliant", "Compliant"],
        output_dict=True,
        zero_division=0
    )
    
    return fold_results, class_report_dict


def run_ensemble_cv(df, all_features, args, data_seed, ensemble_seeds):
    """
    Run 5-fold CV with ensemble of classifiers trained with different seeds.
    For each fold: train N classifiers with different random seeds, average their
    predicted probabilities, then convert to final predictions.
    """
    y_all = df["verdict"].values
    
    # Load fixed folds
    fold_assignments, groups = load_fixed_folds(data_seed)
    
    fold_results = []
    all_true_labels = []
    all_pred_labels = []
    
    for fold_assignment in fold_assignments:
        fold_idx = fold_assignment["fold"]
        train_idx = np.array(fold_assignment["train"])
        test_idx = np.array(fold_assignment["test"])
        
        # Split features
        X_train_raw = all_features[train_idx]
        X_test_raw = all_features[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Train ensemble of classifiers with different seeds
        all_proba = []
        for clf_seed in ensemble_seeds:
            clf = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=clf_seed
            )
            clf.fit(X_train, y_train)
            
            # Get predicted probabilities (aligned to clf.classes_)
            proba = clf.predict_proba(X_test)
            all_proba.append(proba)
        
        # Average probabilities across all ensemble members
        avg_proba = np.mean(all_proba, axis=0)
        
        # Convert to final predictions using the first classifier's classes
        # (all classifiers will have the same class order for the same training data)
        test_preds = clf.classes_[np.argmax(avg_proba, axis=1)]
        
        # Metrics
        f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test, test_preds)
        
        all_true_labels.extend(y_test)
        all_pred_labels.extend(test_preds)
        
        fold_results.append({
            "fold": fold_idx,
            "seed": data_seed,
            "ensemble_size": len(ensemble_seeds),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["all_rules_sim", "rubric", "all_rules_sim+rubric", "tuned", "ensemble"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Load data
    corpus_path = Path("corpus/labeled_clauses_bootstrap.csv")
    taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")
    
    df = pd.read_csv(corpus_path)
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    
    print(f"[INFO] Loaded {len(df)} clauses")
    print(f"[INFO] Variant: {args.variant}")
    
    # Feature builders only ever see clause_text/app_name (no LLM output columns)
    inputs = model_inputs(df)
    
    # Extract base features (E5 + 8 structured)
    print("\n[INFO] Extracting base features (E5 clause + 8 structured)...")
    base_feature_data = extract_encoder_swap_features(inputs, taxonomy, "intfloat/e5-base-v2")
    e5_embeddings = base_feature_data["embeddings"]
    structured_features = base_feature_data["structured"]
    
    # Add incremental features based on variant
    if args.variant == "all_rules_sim":
        print("\n[INFO] Adding all-rules similarity features...")
        all_rules_sim = extract_all_rules_similarity(inputs, taxonomy, "intfloat/e5-base-v2")
        all_features = np.hstack([e5_embeddings, structured_features, all_rules_sim])
        description = "E5 clause + 8 structured + all-rules similarity"
    
    elif args.variant == "rubric":
        print("\n[INFO] Adding rubric features...")
        rubric_feats, rubric_names = extract_rubric_features(inputs, labels=df["verdict"].values)
        all_features = np.hstack([e5_embeddings, structured_features, rubric_feats])
        description = "E5 clause + 8 structured + rubric features"
    
    elif args.variant == "all_rules_sim+rubric":
        print("\n[INFO] Adding all-rules similarity + rubric features...")
        all_rules_sim = extract_all_rules_similarity(inputs, taxonomy, "intfloat/e5-base-v2")
        rubric_feats, rubric_names = extract_rubric_features(inputs, labels=df["verdict"].values)
        all_features = np.hstack([e5_embeddings, structured_features, all_rules_sim, rubric_feats])
        description = "E5 clause + 8 structured + all-rules sim + rubric"
    
    elif args.variant in ["tuned", "ensemble"]:
        # Use best feature combo (all-rules similarity only, NOT rubric)
        print("\n[INFO] Using best feature combination (all-rules similarity only)...")
        all_rules_sim = extract_all_rules_similarity(inputs, taxonomy, "intfloat/e5-base-v2")
        all_features = np.hstack([e5_embeddings, structured_features, all_rules_sim])
        description = f"E5 + 8 + all-rules [{args.variant}]"
    
    else:
        # Baseline (fixed_A_e5)
        all_features = np.hstack([e5_embeddings, structured_features])
        description = "E5 clause + 8 structured (baseline)"
    
    print(f"[INFO] Total features: {all_features.shape[1]}")
    
    # Classifier configs
    if args.variant == "tuned":
        clf_configs = [
            {"type": "logreg", "C": 0.1},
            {"type": "logreg", "C": 0.3},
            {"type": "logreg", "C": 1.0},
            {"type": "logreg", "C": 3.0},
            {"type": "logreg", "C": 10.0},
            {"type": "linearsvc", "C": 1.0},
        ]
        if HAS_LIGHTGBM:
            clf_configs.append({"type": "lightgbm", "n_estimators": 100})
    else:
        clf_configs = [{"type": "logreg", "C": 1.0}]  # Default
    
    # Run experiments
    print(f"\n{'='*80}")
    if args.variant == "ensemble":
        print(f"RUNNING {len(seeds)} SEEDS × 5 FOLDS with ENSEMBLE")
    else:
        print(f"RUNNING {len(seeds)} SEEDS × 5 FOLDS")
    print(f"{'='*80}")
    
    all_fold_results = []
    all_class_reports = []
    
    if args.variant == "ensemble":
        # Use ensemble of 5 classifier seeds for each data seed
        ensemble_seeds = [42, 43, 44, 45, 46]
        
        for seed_idx, data_seed in enumerate(seeds, 1):
            print(f"\n--- Data Seed {data_seed} ({seed_idx}/{len(seeds)}) with Ensemble {ensemble_seeds} ---")
            
            fold_results, class_report = run_ensemble_cv(df, all_features, args, data_seed, ensemble_seeds)
            all_fold_results.extend(fold_results)
            all_class_reports.append(class_report)
    else:
        # Standard single-classifier per fold
        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n--- Seed {seed} ({seed_idx}/{len(seeds)}) ---")
            print(f"[INFO] Using fixed folds, seed {seed}")
            
            fold_results, class_report = run_single_seed_cv(df, all_features, args, seed, clf_configs)
            all_fold_results.extend(fold_results)
            all_class_reports.append(class_report)
    
    # Aggregate
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
    
    macro_precision = float(np.mean([report["macro avg"]["precision"] for report in all_class_reports]))
    macro_precision_std = float(np.std([report["macro avg"]["precision"] for report in all_class_reports]))
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*80}")
    print(f"Accuracy:        {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro-F1:        {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Macro-Precision: {macro_precision:.4f} ± {macro_precision_std:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for cls in ["Compliant", "Partially Compliant"]:
        m = metrics_by_class[cls]
        print(f"  {cls}:")
        print(f"    Precision: {m['precision']:.4f} ± {m['precision_std']:.4f}")
        print(f"    Recall:    {m['recall']:.4f} ± {m['recall_std']:.4f}")
        print(f"    F1:        {m['f1']:.4f} ± {m['f1_std']:.4f}")
    
    # Save
    git_commit = get_git_commit()
    folds_data = json.loads(Path("datasets/folds/fixed_folds_v1.json").read_text())
    
    experiment_data = {
        "tag": args.tag,
        "variant": args.variant,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "description": description,
        "config": {
            "seeds": seeds,
            "n_features": all_features.shape[1],
            "clf_configs": clf_configs if args.variant == "tuned" else "default",
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
    
    output_file = OUTPUT_DIR / f"{args.tag}.json"
    output_file.write_text(json.dumps(experiment_data, indent=2))
    print(f"\n[SUCCESS] Saved to {output_file}")
    
    # Append to summary
    summary_csv = OUTPUT_DIR / "summary.csv"
    summary_row = {
        "tag": args.tag,
        "variant": args.variant,
        "seeds": args.seeds,
        "grouped": True,
        "n_features": all_features.shape[1],
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
