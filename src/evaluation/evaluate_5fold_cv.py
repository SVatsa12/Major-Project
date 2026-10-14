"""
src/evaluation/evaluate_5fold_cv.py
Stratified 5-Fold Cross-Validation - Phase B: Feature Union.

Implements leak-free threshold calibration with:
- Context-formatted string-level conditioning
- all-MiniLM-L6-v2 embeddings (384 dims, mean pooling, L2-normalized)
- 15 actionability features (8 baseline + 7 count-based union), scaled with RobustScaler
- LogisticRegression with specified hyperparameters
- Inner stratified CV for threshold calibration on training fold only
- Baseline objective only: maximize macro-F1 subject to Compliant precision >= 0.70
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.feature_extraction import extract_actionability_features
from src.training.train_classifier import build_context_texts

CORPUS_PATH = Path("corpus/labeled_clauses_bootstrap.csv")
TAXONOMY_PATH = Path("corpus/dpdp_taxonomy_final.json")
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
expected_dist = {"Not Addressed": 586, "Partially Compliant": 143, "Compliant": 97}
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

# Extract 15 actionability features (Phase B: Feature Union)
print("[INFO] Extracting 15 actionability features (Phase B: Feature Union)...")
action_features = extract_actionability_features(df["clause_text"].values)
print(f"[INFO] Feature shape: {action_features.shape}")

# Get class labels
y_all = df["verdict"].values

# Generate embeddings: all-MiniLM-L6-v2, max_seq_length=256, mean pooling, L2-normalized
print("[INFO] Will generate all-MiniLM-L6-v2 embeddings (384-dim, L2-normalized)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 5-Fold Stratified Cross-Validation
print("\n" + "=" * 80)
print("STRATIFIED 5-FOLD CROSS-VALIDATION - PHASE B: FEATURE UNION")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
all_true_labels = []
all_pred_labels = []
fold_f1s = []
fold_accs = []

# Pre-compute all embeddings
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
    
    # Fit RobustScaler ONLY on training partition of this fold
    scaler = RobustScaler()
    X_train_features_scaled = scaler.fit_transform(X_train_features)
    X_test_features_scaled = scaler.transform(X_test_features)
    
    # Combine embedding and scaled features (384 + 15 = 399 dimensions)
    X_train_combined = np.hstack([X_train_embedding, X_train_features_scaled])
    X_test_combined = np.hstack([X_test_embedding, X_test_features_scaled])
    
    # Train classifier
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_combined, y_train)
    
    # Perform inner stratified CV for threshold calibration
    # Use training fold to search for best thresholds
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    best_tau_compliant = 0.5
    best_tau_partial = 0.35
    best_inner_f1 = 0.0
    
    for inner_train_idx, inner_val_idx in inner_skf.split(X_train_combined, y_train):
        X_inner_train = X_train_combined[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = X_train_combined[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]
        
        # Train classifier on inner training
        inner_clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=42
        )
        inner_clf.fit(X_inner_train, y_inner_train)
        
        # Get probabilities on inner validation fold
        inner_probs = inner_clf.predict_proba(X_inner_val)
        classes = list(inner_clf.classes_)
        
        if "Compliant" not in classes:
            continue
        
        c_idx = classes.index("Compliant")
        p_idx = classes.index("Partially Compliant")
        
        # Grid search over thresholds
        for tau_c in np.arange(0.30, 0.70, 0.05):
            for tau_p in np.arange(0.20, 0.45, 0.05):
                preds = []
                for i in range(len(inner_probs)):
                    p_compliant = inner_probs[i, c_idx]
                    p_partial = inner_probs[i, p_idx]
                    
                    if p_compliant >= tau_c:
                        preds.append("Compliant")
                    elif p_partial >= tau_p:
                        preds.append("Partially Compliant")
                    else:
                        preds.append("Not Addressed")
                
                # Compute precision for Compliant class
                try:
                    compliant_precision = precision_score(
                        y_inner_val, preds, labels=["Compliant"], average=None, zero_division=0
                    )[0]
                except:
                    compliant_precision = 0.0
                
                # Maximize inner macro-F1 subject to precision constraint (baseline objective)
                if compliant_precision >= 0.70:
                    inner_f1 = f1_score(y_inner_val, preds, average="macro", zero_division=0)
                    if inner_f1 > best_inner_f1:
                        best_inner_f1 = inner_f1
                        best_tau_compliant = tau_c
                        best_tau_partial = tau_p
    
    # Apply best thresholds to test fold
    test_probs = clf.predict_proba(X_test_combined)
    classes = list(clf.classes_)
    test_preds = []
    
    if "Compliant" in classes:
        c_idx = classes.index("Compliant")
        p_idx = classes.index("Partially Compliant")
        
        for i in range(len(test_probs)):
            p_compliant = test_probs[i, c_idx]
            p_partial = test_probs[i, p_idx]
            
            if p_compliant >= best_tau_compliant:
                test_preds.append("Compliant")
            elif p_partial >= best_tau_partial:
                test_preds.append("Partially Compliant")
            else:
                test_preds.append("Not Addressed")
    else:
        test_preds = ["Not Addressed"] * len(test_probs)
    
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
        "tau_compliant": round(best_tau_compliant, 4),
        "tau_partial": round(best_tau_partial, 4),
    })
    
    print(f"  Macro-F1: {f1:.4f} | Accuracy: {acc:.4f}")
    print(f"  tau_compliant: {best_tau_compliant:.4f} | tau_partial: {best_tau_partial:.4f}")

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
    "phase": "Phase B - Feature Union",
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
    "feature_count": 15,
    "feature_names": [
        "hedge_density (baseline)",
        "action_verb_count (baseline)",
        "has_ui_path (baseline)",
        "has_contact_mechanism (baseline)",
        "has_timeline_sla (baseline)",
        "commitment_ratio (baseline)",
        "clause_length_norm (baseline)",
        "parental_verification_marker (baseline)",
        "hedge_phrase_count (union)",
        "action_phrase_count (union)",
        "requirement_count (union)",
        "commitment_phrase_count (union)",
        "mechanism_count (union)",
        "actionability_ratio (union)",
        "has_nav (union)"
    ],
    "duplicate_analysis": "No duplicates detected. All 15 features retained: hedge_density and hedge_phrase_count differ in computation (density vs count); action_verb_count and action_phrase_count use different lexicons; commitment_ratio and commitment_phrase_count differ (ratio vs count); has_ui_path and has_nav use different regex patterns.",
    "classifier_config": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "multinomial",
        "class_weight": "balanced"
    },
    "scaler": "RobustScaler (fit on training fold only, applied to 15-dim handcrafted features)",
    "threshold_calibration": "Inner stratified 3-fold CV on training partition, maximize macro-F1 subject to Compliant precision >= 0.70 (baseline objective only)",
    "classification_report": class_report_str,
    "classification_report_dict": class_report_dict,
}

# Write report
report_path = OUTPUT_DIR / "cross_validation_report.json"
report_path.write_text(json.dumps(report_data, indent=2))
print(f"\n[SUCCESS] Saved cross_validation_report.json to {report_path}")
