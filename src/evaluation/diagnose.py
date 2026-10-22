"""
Comprehensive Model Diagnostics
================================
Analyzes the trained classifier across all splits to detect issues:
- Branch fingerprint (git, model, taxonomy, labeling prompt, test hash)
- Label distributions
- Baseline performance
- Current model performance with confidence intervals
- Data leakage detection (duplicates and near-duplicates)
- Per-taxonomy-rule label concentration
- Top misclassifications

Output: reports/diagnostics/
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import joblib
import warnings
import subprocess
import hashlib

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.feature_extraction import extract_actionability_features

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TRAIN_PATH = Path("datasets/splits/train.csv")
VAL_PATH = Path("datasets/splits/val.csv")
TEST_PATH = Path("datasets/splits/test.csv")
TAXONOMY_PATH = Path("corpus/dpdp_taxonomy_final.json")
RUN_STATE_PATH = Path("corpus/labeled_clauses_run_state.json")
REFERENCE_TEST_PATH = Path("datasets/splits/test.csv")  # For comparison across branches
MODEL_PATH = Path("models/context_classifier.joblib")
OUTPUT_DIR = Path("reports/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDER_NAME = "all-MiniLM-L6-v2"
N_BOOTSTRAP = 1000
SIMILARITY_THRESHOLD = 0.90
LABEL_CONCENTRATION_THRESHOLD = 0.15

CLASS_LABELS = ["Not Addressed", "Partially Compliant", "Compliant"]

print("=" * 80)
print("COMPREHENSIVE MODEL DIAGNOSTICS")
print("=" * 80)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Output Directory: {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 0. Branch Fingerprint
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[0/8] Generating branch fingerprint...")

# Git information
try:
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                                         text=True, stderr=subprocess.DEVNULL).strip()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                         text=True, stderr=subprocess.DEVNULL).strip()
except:
    git_branch = "unknown"
    git_commit = "unknown"

# Load taxonomy
taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
tax_map = {item["taxonomy_id"]: item for item in taxonomy}
n_taxonomy_rules = len(taxonomy)

# Labeling prompt version
labeling_prompt_version = "unknown"
if RUN_STATE_PATH.exists():
    run_state = json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
    labeling_prompt_version = run_state.get("status", "unknown")

# Test set hash (clause_id + verdict columns)
df_test_temp = pd.read_csv(TEST_PATH)
test_fingerprint_str = df_test_temp[["clause_id", "verdict"]].to_csv(index=False)
test_hash = hashlib.sha256(test_fingerprint_str.encode()).hexdigest()

# Count shared test clause_ids with reference test.csv
reference_test_ids = set()
if REFERENCE_TEST_PATH.exists():
    df_ref = pd.read_csv(REFERENCE_TEST_PATH)
    reference_test_ids = set(df_ref["clause_id"].values)

current_test_ids = set(df_test_temp["clause_id"].values)
shared_test_count = len(current_test_ids & reference_test_ids)

# Model information (will be populated after loading)
model_fingerprint = {
    "git": {
        "branch": git_branch,
        "commit_hash": git_commit,
    },
    "taxonomy": {
        "file": str(TAXONOMY_PATH),
        "n_rules": n_taxonomy_rules,
    },
    "labeling_prompt_version": labeling_prompt_version,
    "test_set": {
        "file": str(TEST_PATH),
        "sha256_hash": test_hash,
        "n_samples": len(df_test_temp),
        "shared_clause_ids_with_reference": shared_test_count,
    },
    "model": {
        "file": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "type": "unknown",
        "hyperparameters": {},
        "thresholds": {},
        "thresholds_tuned_on": "unknown",
    }
}

print(f"  Branch: {git_branch}")
print(f"  Commit: {git_commit[:8]}")
print(f"  Taxonomy rules: {n_taxonomy_rules}")
print(f"  Test hash: {test_hash[:16]}...")
print(f"  Shared test samples with reference: {shared_test_count}/{len(df_test_temp)}")

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[1/8] Loading data splits...")
df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)
df_test = pd.read_csv(TEST_PATH)
df_all = pd.concat([df_train, df_val, df_test], keys=["train", "val", "test"])

print(f"  Train: {len(df_train)} samples")
print(f"  Val:   {len(df_val)} samples")
print(f"  Test:  {len(df_test)} samples")
print(f"  Total: {len(df_all)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Label Distribution Analysis
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2/8] Analyzing label distributions...")

label_stats = {}
for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test), ("overall", df_all)]:
    counts = df_split["verdict"].value_counts()
    total = len(df_split)
    
    label_stats[split_name] = {
        label: {
            "count": int(counts.get(label, 0)),
            "percentage": round(counts.get(label, 0) / total * 100, 2)
        }
        for label in CLASS_LABELS
    }
    label_stats[split_name]["total"] = total

# Save label distribution
with open(OUTPUT_DIR / "label_distribution.json", "w") as f:
    json.dump(label_stats, f, indent=2)

print("\n  Label Distribution:")
print(f"  {'Split':<10} {'Not Addressed':<20} {'Partially Compliant':<25} {'Compliant':<20}")
print("  " + "-" * 80)
for split_name in ["train", "val", "test", "overall"]:
    stats = label_stats[split_name]
    na = f"{stats['Not Addressed']['count']} ({stats['Not Addressed']['percentage']}%)"
    pc = f"{stats['Partially Compliant']['count']} ({stats['Partially Compliant']['percentage']}%)"
    c = f"{stats['Compliant']['count']} ({stats['Compliant']['percentage']}%)"
    print(f"  {split_name:<10} {na:<20} {pc:<25} {c:<20}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Majority-Class Baseline
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3/8] Computing majority-class baseline...")

majority_class = df_train["verdict"].mode()[0]
y_test = df_test["verdict"].values
baseline_preds = np.array([majority_class] * len(y_test))

baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_macro_f1 = f1_score(y_test, baseline_preds, labels=CLASS_LABELS, average="macro", zero_division=0)

baseline_stats = {
    "majority_class": majority_class,
    "test_accuracy": round(baseline_acc, 4),
    "test_macro_f1": round(baseline_macro_f1, 4)
}

with open(OUTPUT_DIR / "baseline_performance.json", "w") as f:
    json.dump(baseline_stats, f, indent=2)

print(f"  Majority Class: {majority_class}")
print(f"  Test Accuracy:  {baseline_acc:.4f}")
print(f"  Test Macro-F1:  {baseline_macro_f1:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Current Model Performance
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/8] Evaluating current model...")

# Helper function to build context texts
def build_context_texts(df, tax_map):
    contexts = []
    for _, row in df.iterrows():
        app_name = str(row.get("app_name", "Platform")).strip()
        clause_text = str(row["clause_text"]).strip()
        tax_id = str(row.get("best_match_taxonomy_id", "NONE"))
        
        if tax_id != "NONE" and tax_id in tax_map:
            statutory_test = tax_map[tax_id].get("checkable_test", "General Policy")
        else:
            statutory_test = "General Policy"
        
        context = f"[{app_name}] Mandate: {statutory_test} | Policy: {clause_text}"
        contexts.append(context)
    
    return contexts

# Load model
if not MODEL_PATH.exists():
    print(f"  [WARNING] Model not found at {MODEL_PATH}")
    print(f"  [INFO] Skipping model evaluation. Please train a model first.")
    model_available = False
else:
    model_artifact = joblib.load(MODEL_PATH)
    model_available = True
    
    # Extract components
    clf = model_artifact["classifier"]
    scaler = model_artifact["scaler"]
    tau_compliant = model_artifact.get("threshold_compliant", model_artifact.get("tau_compliant", 0.5))
    tau_partial = model_artifact.get("threshold_partial", model_artifact.get("tau_partial", 0.35))
    
    # Update model fingerprint
    model_fingerprint["model"]["type"] = type(clf).__name__
    model_fingerprint["model"]["exists"] = True
    
    # Extract hyperparameters if available
    if hasattr(clf, "get_params"):
        clf_params = clf.get_params()
        model_fingerprint["model"]["hyperparameters"] = {
            k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
            for k, v in clf_params.items()
        }
    
    model_fingerprint["model"]["thresholds"] = {
        "tau_compliant": round(float(tau_compliant), 4),
        "tau_partial": round(float(tau_partial), 4),
    }
    
    # Infer tuning split from model artifact
    tuning_info = model_artifact.get("training_data", {})
    if "threshold_tuned_on" in model_artifact:
        model_fingerprint["model"]["thresholds_tuned_on"] = model_artifact["threshold_tuned_on"]
    elif "validation" in str(tuning_info).lower():
        model_fingerprint["model"]["thresholds_tuned_on"] = "validation"
    else:
        model_fingerprint["model"]["thresholds_tuned_on"] = "inferred: train or validation"
    
    print(f"  Model loaded from: {MODEL_PATH}")
    print(f"  Model type: {model_fingerprint['model']['type']}")
    print(f"  Thresholds: tau_compliant={tau_compliant:.3f}, tau_partial={tau_partial:.3f}")
    
    # Generate embeddings and features
    print(f"  Generating embeddings with {EMBEDDER_NAME}...")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    
    val_texts = build_context_texts(df_val, tax_map)
    test_texts = build_context_texts(df_test, tax_map)
    
    val_embeddings = embedder.encode(val_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    test_embeddings = embedder.encode(test_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    
    print(f"  Extracting actionability features...")
    val_features = extract_actionability_features(df_val["clause_text"].values)
    test_features = extract_actionability_features(df_test["clause_text"].values)
    
    # Scale and combine
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    
    X_val = np.hstack([val_embeddings, val_features_scaled])
    X_test = np.hstack([test_embeddings, test_features_scaled])
    
    y_val = df_val["verdict"].values
    y_test = df_test["verdict"].values
    
    # Get predictions
    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)
    
    classes = list(clf.classes_)
    idx_compliant = classes.index("Compliant")
    idx_partial = classes.index("Partially Compliant")
    
    def apply_thresholds(probs, tau_c, tau_p):
        preds = []
        for i in range(len(probs)):
            p_c = probs[i, idx_compliant]
            p_p = probs[i, idx_partial]
            
            if p_c >= tau_c:
                preds.append("Compliant")
            elif p_p >= tau_p:
                preds.append("Partially Compliant")
            else:
                preds.append("Not Addressed")
        return np.array(preds)
    
    val_preds = apply_thresholds(val_probs, tau_compliant, tau_partial)
    test_preds = apply_thresholds(test_probs, tau_compliant, tau_partial)
    
    # Compute metrics
    def compute_metrics(y_true, y_pred, probs=None):
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0), 4),
        }
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
        
        metrics["per_class"] = {
            CLASS_LABELS[i]: {
                "precision": round(precision[i], 4),
                "recall": round(recall[i], 4),
                "f1_score": round(f1[i], 4),
            }
            for i in range(len(CLASS_LABELS))
        }
        
        return metrics
    
    val_metrics = compute_metrics(y_val, val_preds)
    test_metrics = compute_metrics(y_test, test_preds)
    
    model_performance = {
        "model_path": str(MODEL_PATH),
        "thresholds": {
            "tau_compliant": round(tau_compliant, 4),
            "tau_partial": round(tau_partial, 4),
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    
    with open(OUTPUT_DIR / "model_performance.json", "w") as f:
        json.dump(model_performance, f, indent=2)
    
    print(f"\n  Validation Performance:")
    print(f"    Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"    Macro-F1:    {val_metrics['macro_f1']:.4f}")
    print(f"    Weighted-F1: {val_metrics['weighted_f1']:.4f}")
    
    print(f"\n  Test Performance:")
    print(f"    Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"    Macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"    Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    
    # Confusion matrices
    for split_name, y_true, y_pred in [("val", y_val, val_preds), ("test", y_test, test_preds)]:
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
        
        # Save as CSV
        cm_df = pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS)
        cm_df.to_csv(OUTPUT_DIR / f"confusion_matrix_{split_name}.csv")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {split_name.upper()}')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"confusion_matrix_{split_name}.png", dpi=150)
        plt.close()
    
    print(f"  Confusion matrices saved to {OUTPUT_DIR}")

# Save branch fingerprint
with open(OUTPUT_DIR / "branch_fingerprint.json", "w") as f:
    json.dump(model_fingerprint, f, indent=2)

print(f"  Branch fingerprint saved to {OUTPUT_DIR / 'branch_fingerprint.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────
if model_available:
    print(f"\n[5/8] Computing bootstrap confidence intervals (n={N_BOOTSTRAP})...")
    
    n_test = len(y_test)
    bootstrap_accs = []
    bootstrap_f1s = []
    
    for i in range(N_BOOTSTRAP):
        # Resample with replacement
        indices = np.random.choice(n_test, size=n_test, replace=True)
        y_boot = y_test[indices]
        preds_boot = test_preds[indices]
        
        acc_boot = accuracy_score(y_boot, preds_boot)
        f1_boot = f1_score(y_boot, preds_boot, labels=CLASS_LABELS, average="macro", zero_division=0)
        
        bootstrap_accs.append(acc_boot)
        bootstrap_f1s.append(f1_boot)
    
    # 95% CI
    acc_ci = (np.percentile(bootstrap_accs, 2.5), np.percentile(bootstrap_accs, 97.5))
    f1_ci = (np.percentile(bootstrap_f1s, 2.5), np.percentile(bootstrap_f1s, 97.5))
    
    bootstrap_results = {
        "n_bootstrap": N_BOOTSTRAP,
        "test_accuracy": {
            "point_estimate": round(test_metrics["accuracy"], 4),
            "ci_95_lower": round(acc_ci[0], 4),
            "ci_95_upper": round(acc_ci[1], 4),
        },
        "test_macro_f1": {
            "point_estimate": round(test_metrics["macro_f1"], 4),
            "ci_95_lower": round(f1_ci[0], 4),
            "ci_95_upper": round(f1_ci[1], 4),
        },
    }
    
    with open(OUTPUT_DIR / "bootstrap_confidence_intervals.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}  95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"  Test Macro-F1:  {test_metrics['macro_f1']:.4f}  95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
else:
    print(f"\n[5/8] Skipping bootstrap (model not available)")
    bootstrap_results = None

# ─────────────────────────────────────────────────────────────────────────────
# 5. Leakage Detection
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/8] Checking for data leakage (duplicates and near-duplicates)...")

# Normalize text for exact duplicate detection
def normalize_text(text):
    return " ".join(str(text).lower().split())

df_train["clause_normalized"] = df_train["clause_text"].apply(normalize_text)
df_val["clause_normalized"] = df_val["clause_text"].apply(normalize_text)
df_test["clause_normalized"] = df_test["clause_text"].apply(normalize_text)

# Exact duplicates
exact_duplicates = []

for split_a, split_b, df_a, df_b in [
    ("train", "val", df_train, df_val),
    ("train", "test", df_train, df_test),
    ("val", "test", df_val, df_test),
]:
    duplicates = pd.merge(
        df_a[["clause_id", "clause_normalized"]], 
        df_b[["clause_id", "clause_normalized"]], 
        on="clause_normalized",
        suffixes=(f"_{split_a}", f"_{split_b}")
    )
    
    for _, row in duplicates.iterrows():
        exact_duplicates.append({
            "split_a": split_a,
            "split_b": split_b,
            f"clause_id_{split_a}": row[f"clause_id_{split_a}"],
            f"clause_id_{split_b}": row[f"clause_id_{split_b}"],
            "text": row["clause_normalized"][:100] + "..."
        })

print(f"  Exact duplicates found: {len(exact_duplicates)}")

# Near-duplicates via embeddings
print(f"  Computing embeddings for near-duplicate detection...")
embedder_leak = SentenceTransformer(EMBEDDER_NAME)

train_leak_emb = embedder_leak.encode(df_train["clause_text"].tolist(), show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
val_leak_emb = embedder_leak.encode(df_val["clause_text"].tolist(), show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
test_leak_emb = embedder_leak.encode(df_test["clause_text"].tolist(), show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

near_duplicates = []

for split_a, split_b, emb_a, emb_b, df_a, df_b in [
    ("train", "val", train_leak_emb, val_leak_emb, df_train, df_val),
    ("train", "test", train_leak_emb, test_leak_emb, df_train, df_test),
    ("val", "test", val_leak_emb, test_leak_emb, df_val, df_test),
]:
    # Compute cosine similarities
    sims = cosine_similarity(emb_a, emb_b)
    
    # Find pairs above threshold
    high_sim_pairs = np.argwhere(sims >= SIMILARITY_THRESHOLD)
    
    for i, j in high_sim_pairs:
        near_duplicates.append({
            "split_a": split_a,
            "split_b": split_b,
            f"clause_id_{split_a}": df_a.iloc[i]["clause_id"],
            f"clause_id_{split_b}": df_b.iloc[j]["clause_id"],
            "cosine_similarity": round(float(sims[i, j]), 4),
            f"text_{split_a}": df_a.iloc[i]["clause_text"][:100] + "...",
            f"text_{split_b}": df_b.iloc[j]["clause_text"][:100] + "...",
        })

print(f"  Near-duplicates (cosine >= {SIMILARITY_THRESHOLD}): {len(near_duplicates)}")

# Save leakage report
leakage_report = {
    "exact_duplicates": {
        "count": len(exact_duplicates),
        "pairs": exact_duplicates,
    },
    "near_duplicates": {
        "threshold": SIMILARITY_THRESHOLD,
        "count": len(near_duplicates),
        "pairs": near_duplicates[:100],  # Limit to first 100 for file size
    },
    "summary": {
        "train_val_exact": sum(1 for d in exact_duplicates if d["split_a"] == "train" and d["split_b"] == "val"),
        "train_test_exact": sum(1 for d in exact_duplicates if d["split_a"] == "train" and d["split_b"] == "test"),
        "val_test_exact": sum(1 for d in exact_duplicates if d["split_a"] == "val" and d["split_b"] == "test"),
        "train_val_near": sum(1 for d in near_duplicates if d["split_a"] == "train" and d["split_b"] == "val"),
        "train_test_near": sum(1 for d in near_duplicates if d["split_a"] == "train" and d["split_b"] == "test"),
        "val_test_near": sum(1 for d in near_duplicates if d["split_a"] == "val" and d["split_b"] == "test"),
    }
}

with open(OUTPUT_DIR / "leakage_check.json", "w") as f:
    json.dump(leakage_report, f, indent=2)

# Save near-duplicates to CSV
if near_duplicates:
    pd.DataFrame(near_duplicates).to_csv(OUTPUT_DIR / "near_duplicates.csv", index=False)

print(f"  Leakage report saved to {OUTPUT_DIR / 'leakage_check.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-Taxonomy-Rule Label Distribution
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[7/8] Analyzing per-taxonomy-rule label distribution...")

# Get positive labels only (Partially Compliant + Compliant)
df_positive = df_all[df_all["verdict"].isin(["Partially Compliant", "Compliant"])]
total_positive = len(df_positive)

# Count by taxonomy rule
rule_counts = df_positive["best_match_taxonomy_id"].value_counts()

flagged_rules = []
for rule_id, count in rule_counts.items():
    if rule_id == "NONE":
        continue
    
    percentage = count / total_positive
    
    if percentage > LABEL_CONCENTRATION_THRESHOLD:
        flagged_rules.append({
            "taxonomy_id": rule_id,
            "count": int(count),
            "percentage": round(percentage * 100, 2),
            "description": tax_map.get(rule_id, {}).get("checkable_test", "N/A")[:100],
        })

flagged_rules = sorted(flagged_rules, key=lambda x: x["percentage"], reverse=True)

taxonomy_analysis = {
    "total_positive_labels": int(total_positive),
    "concentration_threshold": LABEL_CONCENTRATION_THRESHOLD,
    "flagged_rules": flagged_rules,
    "flagged_count": len(flagged_rules),
}

with open(OUTPUT_DIR / "taxonomy_rule_concentration.json", "w") as f:
    json.dump(taxonomy_analysis, f, indent=2)

print(f"  Total positive labels: {total_positive}")
print(f"  Rules holding >{LABEL_CONCENTRATION_THRESHOLD*100}% of positive labels: {len(flagged_rules)}")

if flagged_rules:
    print(f"\n  Top concentrated rules:")
    for rule in flagged_rules[:5]:
        print(f"    {rule['taxonomy_id']}: {rule['count']} ({rule['percentage']}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Top Misclassifications
# ─────────────────────────────────────────────────────────────────────────────
if model_available:
    print(f"\n[8/8] Extracting top 30 test misclassifications...")
    
    # Get misclassifications
    misclass_mask = y_test != test_preds
    misclass_indices = np.where(misclass_mask)[0]
    
    misclassifications = []
    
    for idx in misclass_indices:
        true_label = y_test[idx]
        pred_label = test_preds[idx]
        
        # Get confidence (max probability)
        confidence = float(test_probs[idx].max())
        
        misclassifications.append({
            "clause_id": df_test.iloc[idx]["clause_id"],
            "app_name": df_test.iloc[idx]["app_name"],
            "clause_text": df_test.iloc[idx]["clause_text"],
            "taxonomy_id": df_test.iloc[idx]["best_match_taxonomy_id"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
        })
    
    # Sort by confidence (descending) - most confident mistakes first
    misclassifications = sorted(misclassifications, key=lambda x: x["confidence"], reverse=True)
    
    # Take top 30
    top_30_misclass = misclassifications[:30]
    
    # Save to CSV
    pd.DataFrame(top_30_misclass).to_csv(OUTPUT_DIR / "top_30_misclassifications.csv", index=False)
    
    print(f"  Total misclassifications: {len(misclassifications)}")
    print(f"  Top 30 saved to {OUTPUT_DIR / 'top_30_misclassifications.csv'}")
    
    if len(top_30_misclass) > 0:
        print(f"\n  Sample misclassification (most confident error):")
        sample = top_30_misclass[0]
        print(f"    Clause: {sample['clause_text'][:150]}...")
        print(f"    True: {sample['true_label']}, Predicted: {sample['predicted_label']}, Confidence: {sample['confidence']:.4f}")
else:
    print(f"\n[8/8] Skipping misclassification analysis (model not available)")
    misclassifications = []

# ─────────────────────────────────────────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\n[OK] Label Distribution:")
print(f"  - Train: {len(df_train)} samples")
print(f"  - Val:   {len(df_val)} samples")
print(f"  - Test:  {len(df_test)} samples")
print(f"  - Majority class: {majority_class} ({label_stats['train']['Not Addressed']['percentage']}% of train)")

print(f"\n[OK] Majority-Class Baseline (Test):")
print(f"  - Accuracy:  {baseline_acc:.4f}")
print(f"  - Macro-F1:  {baseline_macro_f1:.4f}")

if model_available:
    print(f"\n[OK] Current Model Performance:")
    print(f"  - Val Accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"  - Val Macro-F1:   {val_metrics['macro_f1']:.4f}")
    print(f"  - Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  - Test Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  - 95% CI (Test Accuracy):  [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"  - 95% CI (Test Macro-F1):  [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
else:
    print(f"\n[WARNING] Model not available - skipped performance evaluation")

print(f"\n[OK] Data Leakage Check:")
print(f"  - Exact duplicates:  {len(exact_duplicates)}")
print(f"  - Near-duplicates (cosine >= {SIMILARITY_THRESHOLD}): {len(near_duplicates)}")
if len(exact_duplicates) > 0 or len(near_duplicates) > 10:
    print(f"  [WARNING] Potential data leakage detected!")

print(f"\n[OK] Taxonomy Rule Concentration:")
print(f"  - Rules holding >{LABEL_CONCENTRATION_THRESHOLD*100}% of positive labels: {len(flagged_rules)}")
if len(flagged_rules) > 0:
    print(f"  [WARNING] Some rules are over-represented in positive labels")

if model_available:
    print(f"\n[OK] Misclassifications:")
    print(f"  - Total test misclassifications: {len(misclassifications)}")
    print(f"  - Top 30 saved for review")

print(f"\n[OK] All diagnostic reports saved to: {OUTPUT_DIR}")
print(f"\nFiles generated:")
print(f"  - branch_fingerprint.json")
print(f"  - label_distribution.json")
print(f"  - baseline_performance.json")
if model_available:
    print(f"  - model_performance.json")
    print(f"  - confusion_matrix_val.csv / .png")
    print(f"  - confusion_matrix_test.csv / .png")
    print(f"  - bootstrap_confidence_intervals.json")
    print(f"  - top_30_misclassifications.csv")
print(f"  - leakage_check.json")
print(f"  - near_duplicates.csv")
print(f"  - taxonomy_rule_concentration.json")

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
