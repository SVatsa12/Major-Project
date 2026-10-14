"""
src/evaluation/evaluate_5fold_cv.py
Stratified 5-Fold Cross-Validation per Research Paper Specification.

Implements leak-free threshold calibration with:
- Context-formatted string-level conditioning
- all-MiniLM-L6-v2 embeddings (384 dims, mean pooling, L2-normalized)
- 8 actionability features, scaled with RobustScaler
- LogisticRegression with specified hyperparameters
- Inner stratified CV for threshold calibration on training fold only
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import joblib

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.feature_extraction import extract_actionability_features
from src.training.train_classifier import (
    build_context_texts,
    train_and_evaluate_fold,
)

CORPUS_PATH = Path("corpus/labeled_clauses_v2.csv")
TAXONOMY_PATH = Path("corpus/dpdp_taxonomy_v2.json")
OUTPUT_DIR = Path("reports")
MODELS_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CORPUS_PATH)
taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
tax_map = {item["taxonomy_id"]: item for item in taxonomy}

print(f"[INFO] Loaded {len(df)} clauses from {CORPUS_PATH}")
print("\nClass distribution:")
print(df["verdict"].value_counts().to_string())

# Verify expected corpus distribution
expected_dist = {"Not Addressed": 579, "Partially Compliant": 160, "Compliant": 87}
actual_dist = df["verdict"].value_counts().to_dict()
print(f"\n[CHECK] Expected distribution: {expected_dist}")
print(f"[CHECK] Actual distribution: {actual_dist}")
for cls, expected_count in expected_dist.items():
    actual_count = actual_dist.get(cls, 0)
    if actual_count != expected_count:
        print(f"[WARNING] Class '{cls}': expected {expected_count}, got {actual_count}")

# Build context texts with format: [app_name] Mandate: checkable_test | Policy: clause_text
print("\n[INFO] Building context-formatted input texts...")
context_texts = build_context_texts(df, tax_map)

# Extract 8 actionability features (early, before embeddings which are slow)
print("[INFO] Extracting 8 actionability features...")
action_features = extract_actionability_features(df["clause_text"].values)

# Get class labels
y_all = df["verdict"].values

# Generate embeddings: all-MiniLM-L6-v2, max_seq_length=256, mean pooling, L2-normalized
# This is done per-fold to avoid memory issues
print("[INFO] Will generate all-MiniLM-L6-v2 embeddings per-fold (384-dim, L2-normalized)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 5-Fold Stratified Cross-Validation
print("\n" + "=" * 80)
print("STRATIFIED 5-FOLD CROSS-VALIDATION WITH LEAK-FREE THRESHOLD CALIBRATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
all_true_labels = []
all_pred_labels = []
fold_f1s = []
fold_accs = []

# Pre-compute all embeddings to avoid doing it per-fold
print("\n[INFO] Pre-computing embeddings for all clauses...")
embeddings = embedder.encode(
    context_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # L2 normalization
)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, y_all), 1):
    print(f"\n--- Fold {fold_idx}/5 ---")
    
    # Split data
    X_train_embedding = embeddings[train_idx]
    X_test_embedding = embeddings[test_idx]
    X_train_features = action_features[train_idx]
    X_test_features = action_features[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    
    # Prepare data dict for train function
    X_train_dict = {"embedding": X_train_embedding, "features": X_train_features}
    X_test_dict = {"embedding": X_test_embedding, "features": X_test_features}
    
    # Train and calibrate thresholds
    result = train_and_evaluate_fold(X_train_dict, y_train, X_test_dict, y_test, inner_cv_splits=3)
    
    # Get predictions
    test_preds = result["predictions"]
    tau_c = result["tau_compliant"]
    tau_p = result["tau_partial"]
    
    # Compute metrics
    f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    acc = accuracy_score(y_test, test_preds)
    
    fold_f1s.append(f1)
    fold_accs.append(acc)
    all_true_labels.extend(y_test)
    all_pred_labels.extend(test_preds)
    
    fold_results.append({
        "fold": fold_idx,
        "macro_f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tau_compliant": round(tau_c, 4),
        "tau_partial": round(tau_p, 4),
    })
    
    print(f"  Macro-F1: {f1:.4f} | Accuracy: {acc:.4f}")
    print(f"  tau_compliant: {tau_c:.4f} | tau_partial: {tau_p:.4f}")

# Compute aggregate metrics
mean_f1 = float(np.mean(fold_f1s))
std_f1 = float(np.std(fold_f1s))
mean_acc = float(np.mean(fold_accs))

print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY")
print("=" * 80)
print(f"Mean Macro-F1: {mean_f1:.4f} (std: {std_f1:.4f})")
print(f"Mean Accuracy: {mean_acc:.4f}")

# Per-fold thresholds
thresholds_per_fold = [
    {"fold": r["fold"], "tau_compliant": r["tau_compliant"], "tau_partial": r["tau_partial"]}
    for r in fold_results
]
tau_compliant_median = float(np.median([r["tau_compliant"] for r in fold_results]))
tau_partial_median = float(np.median([r["tau_partial"] for r in fold_results]))

print(f"\nPer-fold tau_compliant: {[r['tau_compliant'] for r in fold_results]}")
print(f"  Median: {tau_compliant_median:.4f}")
print(f"Per-fold tau_partial: {[r['tau_partial'] for r in fold_results]}")
print(f"  Median: {tau_partial_median:.4f}")

# Classification report
print("\n" + "-" * 80)
print("POOLED CLASSIFICATION REPORT (All folds)")
print("-" * 80)
class_report_str = classification_report(
    all_true_labels, all_pred_labels,
    labels=["Not Addressed", "Partially Compliant", "Compliant"],
    digits=4,
    zero_division=0
)
print(class_report_str)

# Parse classification report for JSON
class_report_dict = classification_report(
    all_true_labels, all_pred_labels,
    labels=["Not Addressed", "Partially Compliant", "Compliant"],
    output_dict=True,
    digits=4,
    zero_division=0
)

# Prepare output report
report_data = {
    "evaluation_method": "Stratified 5-Fold Cross-Validation (Leak-Free Threshold Calibration)",
    "total_clauses_evaluated": len(df),
    "expected_corpus_distribution": expected_dist,
    "actual_corpus_distribution": actual_dist,
    "mean_macro_f1": round(mean_f1, 4),
    "std_macro_f1": round(std_f1, 4),
    "mean_accuracy": round(mean_acc, 4),
    "per_fold_results": fold_results,
    "per_fold_thresholds": thresholds_per_fold,
    "tau_compliant_median": round(tau_compliant_median, 4),
    "tau_partial_median": round(tau_partial_median, 4),
    "embedding_model": "all-MiniLM-L6-v2 (384-dim, mean pooling, L2-normalized)",
    "feature_count": 8,
    "feature_names": [
        "hedge_density",
        "action_verb_count",
        "has_ui_path",
        "has_contact_mechanism",
        "has_timeline_sla",
        "commitment_ratio",
        "clause_length_norm",
        "parental_verification_marker"
    ],
    "classifier_config": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "multinomial",
        "class_weight": "balanced"
    },
    "scaler": "RobustScaler (fit on training fold only per-fold)",
    "threshold_calibration": "Inner stratified 3-fold CV on training partition, maximize macro-F1 subject to minority class (Compliant) precision >= 0.70",
    "classification_report": class_report_str,
    "classification_report_dict": class_report_dict,
}

# Write report
report_path = OUTPUT_DIR / "cross_validation_report_v2.json"
report_path.write_text(json.dumps(report_data, indent=2))
print(f"\n[SUCCESS] Saved cross_validation_report_v2.json to {report_path}")

# Save the trained classifier (using last fold's classifier and scaler for final model)
# Note: In production, would retrain on full dataset with median thresholds
# For now, save fold 1's model as context_classifier.joblib
print(f"\n[INFO] Model artifacts would be saved here: {MODELS_DIR / 'context_classifier.joblib'}")