"""
src/training/train_final_model.py
Train the final model (E5+8+all-rules, 825 features; ordinal classifier) on all 826 clauses.
Save the model, config, thresholds, feature list, and git commit hash.

Config = r1_ordinal from run_round2_variants.py (selected on training folds only,
confirmation macro-F1 0.6436 ± 0.0374):
    input text = "query: " + clause_text  (label-free, src/training/model_inputs.py)
    classifier = two cumulative LogRegs, P(y > NA) and P(y > Partial), on RobustScaled
                 features (src/training/ordinal.py), C in {0.1, 0.3, 1, 3} chosen by
                 inner CV on all 826 rows
    thresholds = tune_thresholds_inner_cv with real inner CV: the full model, including
                 its C selection, is refit on each inner training split
"""

import json
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from datetime import datetime
import joblib

# Import project modules
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.model_inputs import model_inputs, encoder_prefix
from src.training.ordinal import C_GRID, INNER_CV_SPLITS, OrdinalLRCV
from src.training.train_classifier import tune_thresholds_inner_cv
from src.evaluation.feature_variants import extract_encoder_swap_features
from src.evaluation.incremental_features import extract_all_rules_similarity, get_active_rules

ENCODER_NAME = "intfloat/e5-base-v2"

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

# Label-free evaluation of this exact config (reports/experiments/round2/)
EVALUATION = {
    "selected_by": "training-fold mean macro-F1 only (reports/experiments/round2/selection.json)",
    "training_folds": {"folds": "datasets/folds/fixed_folds_v1.json", "seeds": [42, 43, 44], "n_folds": 15,
                       "macro_f1_mean": 0.6311, "macro_f1_std": 0.0262,
                       "result": "reports/experiments/round2/train_r1_ordinal.json"},
    "confirmation_folds": {"folds": "datasets/folds/confirm_folds_v1.json", "seeds": [45, 46, 47, 48, 49], "n_folds": 25,
                           "macro_f1_mean": 0.6436, "macro_f1_std": 0.0374,
                           "accuracy_mean": 0.7421, "accuracy_std": 0.0305,
                           "per_class_f1": {"Not Addressed": 0.8437, "Partially Compliant": 0.4154, "Compliant": 0.6717},
                           "vs_previous_final_model": "+0.0304 macro-F1, 23/25 folds, Wilcoxon p = 1.8e-5",
                           "result": "reports/experiments/round2/confirm_r1_ordinal.json"},
}


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
    """Ordinal classifier with C selected by inner CV (also refit per threshold-tuning fold)."""
    return OrdinalLRCV(C_GRID, inner_cv_splits=INNER_CV_SPLITS)


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
    X_all = np.hstack([embeddings, action_features, all_rules_features]).astype(np.float64)
    y_all = df["verdict"].values

    print(f"\n[INFO] Final feature matrix shape: {X_all.shape}")
    print(f"[INFO] Feature breakdown: {embeddings.shape[1]} (E5) + {action_features.shape[1]} (actionability) + {all_rules_features.shape[1]} (all-rules) = {X_all.shape[1]} total")

    # Train ordinal model; RobustScaler is fit inside it
    print(f"\n[INFO] Training ordinal model, C in {list(C_GRID)} by {INNER_CV_SPLITS}-fold inner CV on all rows...")
    model = make_final_classifier().fit(X_all, y_all)
    print(f"[INFO] Inner-CV macro-F1 per C: {model.cv_scores_}")
    print(f"[INFO] Chosen C: {model.C_}")
    print(f"[INFO] Model classes: {model.classes_.tolist()}")

    # Tune thresholds with real inner CV (the whole model, incl. C selection, refit per inner fold)
    print(f"\n[INFO] Tuning decision thresholds with {INNER_CV_SPLITS}-fold inner CV (refit per inner fold)...")
    tau_compliant, tau_partial, inner_f1 = tune_thresholds_inner_cv(
        X_all, y_all, make_final_classifier, inner_cv_splits=INNER_CV_SPLITS
    )
    constraint_met = inner_f1 > 0.0
    if not constraint_met:
        print("[WARNING] No threshold pair met Compliant precision >= 0.70 on any inner fold; using defaults")

    print(f"[INFO] Tuned thresholds: tau_compliant={tau_compliant:.3f}, tau_partial={tau_partial:.3f} (best inner macro-F1={inner_f1:.4f})")

    # Save model and artifacts
    output_dir = Path("models/final_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving final model to {output_dir}...")

    # Save model (scaling is inside the model; there is no separate scaler artifact)
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Saved model to {model_path}")
    stale_scaler = output_dir / "scaler.joblib"
    if stale_scaler.exists():
        stale_scaler.unlink()
        print(f"[INFO] Removed {stale_scaler} (scaler now lives inside the ordinal model)")

    # Save thresholds
    thresholds_dict = {
        "tau_compliant": round(float(tau_compliant), 4),
        "tau_partial": round(float(tau_partial), 4),
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
        "feature_order": "[e5_embeddings | actionability_features | all_rules_similarity]",
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
        "all_rules_details": f"{n_rules} rule similarities + {all_rules_features.shape[1] - n_rules} category max similarities (categories in sorted order) = {all_rules_features.shape[1]} features",
        "scaling": "RobustScaler fit inside the model on the training rows"
    }

    feature_config_path = output_dir / "feature_config.json"
    with open(feature_config_path, 'w', encoding='utf-8') as f:
        json.dump(feature_config, f, indent=2)
    print(f"[SUCCESS] Saved feature config to {feature_config_path}")

    # Save model configuration
    model_config = {
        "model_type": "OrdinalLogisticRegression",
        "model_class": "src.training.ordinal.OrdinalLRCV",
        "description": "Two cumulative binary LogRegs on RobustScaled features: P(y > Not Addressed), "
                       "P(y > Partially Compliant); P(Partial) = P(>NA) - P(>Partial), P(>Partial) capped at P(>NA)",
        "class_order": ["Not Addressed", "Partially Compliant", "Compliant"],
        "hyperparameters": {
            "C": float(model.C_),
            "C_grid": list(C_GRID),
            "C_selection": f"{INNER_CV_SPLITS}-fold stratified inner CV on all {len(df)} rows, mean argmax macro-F1, ties to smaller C",
            "inner_cv_macro_f1_by_C": {str(k): v for k, v in model.cv_scores_.items()},
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "calibration": None,
        "probabilities": "from the two cumulative LogRegs (no probability calibration)",
        "classes": model.classes_.tolist(),
        "threshold_tuning": {
            "method": "inner stratified CV, full model (incl. C selection) refit per inner fold (tune_thresholds_inner_cv)",
            "inner_cv_splits": INNER_CV_SPLITS,
            "objective": "macro-F1 subject to Compliant precision >= 0.70",
            "grid": {"tau_compliant": "0.30-0.65 step 0.05", "tau_partial": "0.20-0.40 step 0.05"}
        },
        "evaluation": EVALUATION,
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
    print(f"  - model.joblib (ordinal LogReg, C={model.C_}, trained on all {len(df)} clauses)")
    print(f"  - thresholds.json (tau_compliant={tau_compliant:.3f}, tau_partial={tau_partial:.3f})")
    print(f"  - feature_config.json ({X_all.shape[1]} features)")
    print(f"  - model_config.json (hyperparameters, evaluation, training data, git commit)")
    print(f"\nGit commit: {model_config['git_commit']}")
    print(f"Timestamp: {model_config['timestamp']}")


if __name__ == "__main__":
    main()
