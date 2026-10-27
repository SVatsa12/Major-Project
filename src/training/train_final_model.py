"""
src/training/train_final_model.py
Train the final model (E5+8+all-rules, 825 features) on all 826 clauses.
Save the model, config, thresholds, feature list, and git commit hash.

Features are built by the same functions as evaluation
(run_confirmation_experiments.py --variant all_rules):
    input text = "query: " + clause_text  (label-free, src/training/model_inputs.py)
Thresholds are tuned with real inner CV (a fresh classifier is refit on each
inner training split) via tune_thresholds_inner_cv, as in train_and_evaluate_fold.
"""

import json
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import joblib

# Import project modules
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.model_inputs import model_inputs, encoder_prefix
from src.training.train_classifier import tune_thresholds_inner_cv
from src.evaluation.feature_variants import extract_encoder_swap_features
from src.evaluation.incremental_features import extract_all_rules_similarity, get_active_rules

ENCODER_NAME = "intfloat/e5-base-v2"
INNER_CV_SPLITS = 3

# Names of the 8 actionability features, in the order of extract_actionability_features
ACTIONABILITY_FEATURE_NAMES = [
    "hedge_density",
    "action_verb_count",
    "has_ui_path",
    "has_contact_mechanism",
    "has_timeline_sla",
    "commitment_ratio",
    "clause_length_norm",
    "parental_verification_marker",
]


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


def make_final_classifier():
    """LogisticRegression configuration of the final model (also refit per inner CV fold)."""
    return LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )


def build_feature_matrix(inputs, taxonomy):
    """
    Build the 825-dim feature matrix [E5 (768) | actionability (8) | all-rules (49)]
    with the evaluation feature functions. inputs must be model_inputs(df).
    """
    feature_data = extract_encoder_swap_features(inputs, taxonomy, ENCODER_NAME)
    all_rules_features = extract_all_rules_similarity(inputs, taxonomy, ENCODER_NAME, rule_text_mode="full")
    return feature_data["embeddings"], feature_data["structured"], all_rules_features


def main():
    print("=" * 120)
    print("TRAINING FINAL MODEL ON ALL 826 CLAUSES")
    print("=" * 120)

    # Load data
    print("\n[INFO] Loading data...")
    data_path = Path("corpus/labeled_clauses_bootstrap.csv")
    taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")

    df = pd.read_csv(data_path)
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)

    print(f"[INFO] Loaded {len(df)} clauses and {len(taxonomy)} taxonomy items")

    # Build features (only clause_text/app_name reach the feature builders)
    embeddings, action_features, all_rules_features = build_feature_matrix(model_inputs(df), taxonomy)
    print(f"[INFO] Embedding shape: {embeddings.shape}")
    print(f"[INFO] Actionability features shape: {action_features.shape}")
    print(f"[INFO] All-rules similarity features shape: {all_rules_features.shape}")

    # Combine features
    X_all = np.hstack([embeddings, action_features, all_rules_features])
    y_all = df["verdict"].values

    print(f"\n[INFO] Final feature matrix shape: {X_all.shape}")
    print(f"[INFO] Feature breakdown: {embeddings.shape[1]} (E5) + {action_features.shape[1]} (actionability) + {all_rules_features.shape[1]} (all-rules) = {X_all.shape[1]} total")

    # Scale features
    print("\n[INFO] Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Train model
    print("[INFO] Training LogisticRegression (C=1.0, class_weight='balanced', max_iter=1000)...")
    model = make_final_classifier()
    model.fit(X_scaled, y_all)

    print(f"[INFO] Model classes: {model.classes_.tolist()}")
    print(f"[INFO] Model coefficients shape: {model.coef_.shape}")

    # Tune thresholds with real inner CV (refit per inner fold)
    print(f"\n[INFO] Tuning decision thresholds with {INNER_CV_SPLITS}-fold inner CV (refit per inner fold)...")
    tau_compliant, tau_partial, inner_f1 = tune_thresholds_inner_cv(
        X_scaled, y_all, make_final_classifier, inner_cv_splits=INNER_CV_SPLITS
    )
    constraint_met = inner_f1 > 0.0
    if not constraint_met:
        print("[WARNING] No threshold pair met Compliant precision >= 0.70 on any inner fold; using defaults")

    print(f"[INFO] Tuned thresholds: tau_compliant={tau_compliant:.3f}, tau_partial={tau_partial:.3f} (best inner macro-F1={inner_f1:.4f})")

    # Save model and artifacts
    output_dir = Path("models/final_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving final model to {output_dir}...")

    # Save model
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Saved model to {model_path}")

    # Save scaler
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[SUCCESS] Saved scaler to {scaler_path}")

    # Save thresholds
    thresholds_dict = {
        "tau_compliant": float(tau_compliant),
        "tau_partial": float(tau_partial),
        "inner_cv_best_macro_f1": float(inner_f1),
        "compliant_precision_constraint_met": bool(constraint_met),
    }
    thresholds_path = output_dir / "thresholds.json"
    with open(thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"[SUCCESS] Saved thresholds to {thresholds_path}")

    # Save feature configuration
    n_rules = len(get_active_rules(taxonomy))
    feature_config = {
        "total_features": int(X_all.shape[1]),
        "feature_breakdown": {
            "e5_embeddings": int(embeddings.shape[1]),
            "actionability_features": int(action_features.shape[1]),
            "all_rules_similarity": int(all_rules_features.shape[1])
        },
        "encoder_model": ENCODER_NAME,
        "input_text": {
            "format": "encoder_prefix + clause_text",
            "encoder_prefix": encoder_prefix(ENCODER_NAME),
            "include_app_name": False,
            "llm_output_columns_used": False,
            "builder": "src/training/model_inputs.py:build_input_texts"
        },
        "actionability_features_list": ACTIONABILITY_FEATURE_NAMES,
        "all_rules_mode": "full (requirement field from taxonomy)",
        "all_rules_details": f"{n_rules} rule similarities + {all_rules_features.shape[1] - n_rules} category max similarities (categories in sorted order) = {all_rules_features.shape[1]} features"
    }

    feature_config_path = output_dir / "feature_config.json"
    with open(feature_config_path, 'w', encoding='utf-8') as f:
        json.dump(feature_config, f, indent=2)
    print(f"[SUCCESS] Saved feature config to {feature_config_path}")

    # Save model configuration
    model_config = {
        "model_type": "LogisticRegression",
        "hyperparameters": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "calibration": None,
        "probabilities": "raw LogisticRegression predict_proba (no probability calibration)",
        "classes": model.classes_.tolist(),
        "threshold_tuning": {
            "method": "inner stratified CV, classifier refit per inner fold (tune_thresholds_inner_cv)",
            "inner_cv_splits": INNER_CV_SPLITS,
            "objective": "macro-F1 subject to Compliant precision >= 0.70",
            "grid": {"tau_compliant": "0.30-0.65 step 0.05", "tau_partial": "0.20-0.40 step 0.05"}
        },
        "training_data": {
            "corpus": "labeled_clauses_bootstrap.csv",
            "taxonomy": "dpdp_taxonomy_final.json",
            "n_samples": int(len(df)),
            "n_classes": len(np.unique(y_all)),
            "class_distribution": {cls: int(np.sum(y_all == cls)) for cls in np.unique(y_all)}
        },
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit()
    }

    model_config_path = output_dir / "model_config.json"
    with open(model_config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)
    print(f"[SUCCESS] Saved model config to {model_config_path}")

    print("\n" + "=" * 120)
    print("FINAL MODEL SAVED SUCCESSFULLY")
    print("=" * 120)
    print(f"\nModel artifacts in: {output_dir}")
    print(f"  - model.joblib (LogisticRegression trained on all {len(df)} clauses)")
    print(f"  - scaler.joblib (RobustScaler)")
    print(f"  - thresholds.json (tau_compliant={tau_compliant:.3f}, tau_partial={tau_partial:.3f})")
    print(f"  - feature_config.json ({X_all.shape[1]} features)")
    print(f"  - model_config.json (hyperparameters, training data, git commit)")
    print(f"\nGit commit: {model_config['git_commit']}")
    print(f"Timestamp: {model_config['timestamp']}")


if __name__ == "__main__":
    main()
