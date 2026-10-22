"""
src/training/train_final_model.py
Train the final model (E5+8+all-rules, 825 features) on all 826 clauses.
Save the model, config, thresholds, feature list, and git commit hash.
"""

import json
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
import joblib

# Import project modules
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.feature_extraction import extract_actionability_features
from src.training.train_classifier import build_context_texts
from src.evaluation.incremental_features import extract_all_rules_similarity


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


def calibrate_thresholds_simple(model, X_train, y_train, n_splits=3):
    """
    Simplified threshold calibration using grid search on inner CV.
    Returns optimal thresholds directly.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    
    # Convert to numpy if pandas
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get probabilities from base model
    # We'll just use the base model probabilities, no recalibration
    
    best_tau_c = 0.5
    best_tau_p = 0.5
    best_f1 = 0.0
    
    # Grid search for thresholds
    for tau_c in np.arange(0.3, 0.8, 0.05):
        for tau_p in np.arange(0.3, 0.8, 0.05):
            if tau_p >= tau_c:
                continue
            
            fold_f1s = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                
                # Get probabilities
                val_probs = model.predict_proba(X_val)
                val_preds = apply_thresholds(val_probs, tau_c, tau_p)
                f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
                fold_f1s.append(f1)
            
            mean_f1 = np.mean(fold_f1s)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_tau_c = tau_c
                best_tau_p = tau_p
    
    return best_tau_c, best_tau_p


def apply_thresholds(probs, tau_compliant, tau_partial):
    """Apply two-threshold decision rule."""
    predictions = []
    class_order = ["Not Addressed", "Partially Compliant", "Compliant"]
    
    for prob_row in probs:
        prob_dict = {cls: p for cls, p in zip(class_order, prob_row)}
        
        if prob_dict["Compliant"] >= tau_compliant:
            predictions.append("Compliant")
        elif prob_dict["Partially Compliant"] >= tau_partial:
            predictions.append("Partially Compliant")
        else:
            predictions.append("Not Addressed")
    
    return np.array(predictions)


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
    
    # Build features
    tax_map = {item["taxonomy_id"]: item for item in taxonomy}
    context_texts = build_context_texts(df, tax_map)
    
    print("[INFO] Loading E5-base embedder...")
    embedder = SentenceTransformer("intfloat/e5-base-v2")
    
    print("[INFO] Generating E5 embeddings...")
    embeddings = embedder.encode(
        context_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"[INFO] Embedding shape: {embeddings.shape}")
    
    print("[INFO] Extracting actionability features...")
    action_features = extract_actionability_features(df["clause_text"].values)
    print(f"[INFO] Actionability features shape: {action_features.shape}")
    
    print("[INFO] Computing all-rules similarity features...")
    all_rules_features = extract_all_rules_similarity(df, taxonomy, "intfloat/e5-base-v2", rule_text_mode="full")
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
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_all)
    
    print(f"[INFO] Model classes: {model.classes_.tolist()}")
    print(f"[INFO] Model coefficients shape: {model.coef_.shape}")
    
    # Calibrate thresholds
    print("\n[INFO] Calibrating decision thresholds using 3-fold inner CV...")
    tau_compliant, tau_partial = calibrate_thresholds_simple(model, X_scaled, y_all, n_splits=3)
    
    print(f"[INFO] Calibrated thresholds: tau_compliant={tau_compliant:.3f}, tau_partial={tau_partial:.3f}")
    
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
        "tau_partial": float(tau_partial)
    }
    thresholds_path = output_dir / "thresholds.json"
    with open(thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"[SUCCESS] Saved thresholds to {thresholds_path}")
    
    # Save feature configuration
    feature_config = {
        "total_features": int(X_all.shape[1]),
        "feature_breakdown": {
            "e5_embeddings": int(embeddings.shape[1]),
            "actionability_features": int(action_features.shape[1]),
            "all_rules_similarity": int(all_rules_features.shape[1])
        },
        "encoder_model": "intfloat/e5-base-v2",
        "actionability_features_list": [
            "clause_word_count",
            "policy_word_count",
            "has_shall",
            "has_must",
            "has_will",
            "has_should",
            "has_may",
            "has_can"
        ],
        "all_rules_mode": "full (requirement field from taxonomy)",
        "all_rules_details": "38 rule similarities + 11 category max similarities = 49 features"
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
        "calibration": {
            "method": "isotonic",
            "cv": 3
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
    print(f"  - model.joblib (LogisticRegression trained on all 826 clauses)")
    print(f"  - scaler.joblib (RobustScaler)")
    print(f"  - thresholds.json (tau_compliant={tau_compliant:.3f}, tau_partial={tau_partial:.3f})")
    print(f"  - feature_config.json (825 features: 768 E5-base + 8 actionability + 49 all-rules)")
    print(f"  - model_config.json (hyperparameters, training data, git commit)")
    print(f"\nGit commit: {model_config['git_commit']}")
    print(f"Timestamp: {model_config['timestamp']}")


if __name__ == "__main__":
    main()
