"""
src/training/train_context_embeddings.py
Context-Conditioned Dense Embedding Classifier (all-MiniLM-L6-v2).
Pairs clauses with statutory taxonomy requirements to resolve class boundaries
and evaluates on stratified holdout test set with balanced macro metrics.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

SPLITS_DIR = Path("datasets/splits")
MODEL_DIR = Path("models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TAXONOMY_PATH = Path("corpus/dpdp_taxonomy_final.json")

# 1. Load splits and statutory taxonomy
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

VALID_LABELS = ["Not Addressed", "Partially Compliant", "Compliant"]
train_df = train_df[train_df["verdict"].isin(VALID_LABELS)].copy()
val_df = val_df[val_df["verdict"].isin(VALID_LABELS)].copy()
test_df = test_df[test_df["verdict"].isin(VALID_LABELS)].copy()

taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
tax_map = {item["taxonomy_id"]: item for item in taxonomy}

def enrich_clause_context(df_subset):
    """
    Pairs raw clause text with statutory requirement context.
    If matched to a canonical rule, appends category and test requirement.
    If 'NONE', marks as unassociated general text.
    """
    contexts = []
    for _, r in df_subset.iterrows():
        rule_id = str(r.get("best_match_taxonomy_id", "NONE"))
        base_text = str(r["clause_text"]).strip()
        app = str(r.get("app_name", "Platform"))
        
        if rule_id in tax_map:
            rule_info = tax_map[rule_id]
            req = rule_info.get("checkable_test", rule_info.get("requirement", ""))
            cat = rule_info.get("category", "")
            enriched = f"[{app}] Category: {cat} | Requirement: {req} | Policy Clause: {base_text}"
        else:
            enriched = f"[{app}] Category: General Terms | Policy Clause: {base_text}"
            
        contexts.append(enriched)
    return contexts

print("[INFO] Enriching clauses with statutory taxonomy context...")
train_contexts = enrich_clause_context(train_df)
val_contexts = enrich_clause_context(val_df)
test_contexts = enrich_clause_context(test_df)

# 2. Extract Dense Semantic Embeddings
print("[INFO] Loading all-MiniLM-L6-v2 embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[INFO] Encoding context vectors (Train, Val, Test)...")
X_train = embedder.encode(train_contexts, show_progress_bar=False, normalize_embeddings=True)
y_train = train_df["verdict"].values

X_val = embedder.encode(val_contexts, show_progress_bar=False, normalize_embeddings=True)
y_val = val_df["verdict"].values

X_test = embedder.encode(test_contexts, show_progress_bar=False, normalize_embeddings=True)
y_test = test_df["verdict"].values

# 3. Train Regularized Classifier on Context Embeddings
clf = LogisticRegression(C=3.5, max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# 4. Calibrate Decision Thresholds on Validation Set
classes = list(clf.classes_)
c_idx = classes.index("Compliant")
pc_idx = classes.index("Partially Compliant")
na_idx = classes.index("Not Addressed")

val_probs = clf.predict_proba(X_val)
test_probs = clf.predict_proba(X_test)

best_val_macro = 0.0
best_th_c = 0.35
best_th_pc = 0.35

for th_c in np.linspace(0.20, 0.55, 36):
    for th_pc in np.linspace(0.25, 0.60, 36):
        v_preds = []
        for p in val_probs:
            if p[c_idx] >= th_c and p[c_idx] >= (p[pc_idx] * 0.85):
                v_preds.append("Compliant")
            elif p[pc_idx] >= th_pc:
                v_preds.append("Partially Compliant")
            else:
                v_preds.append("Not Addressed")
                
        score = f1_score(y_val, v_preds, average="macro", zero_division=0)
        if score > best_val_macro:
            best_val_macro = score
            best_th_c = float(th_c)
            best_th_pc = float(th_pc)

print(f"[INFO] Calibrated Validation Thresholds: Compliant={best_th_c:.3f}, Partial={best_th_pc:.3f} (Val F1={best_val_macro:.4f})")

# 5. Evaluate on Holdout Test Set
test_preds = []
for p in test_probs:
    if p[c_idx] >= best_th_c and p[c_idx] >= (p[pc_idx] * 0.85):
        test_preds.append("Compliant")
    elif p[pc_idx] >= best_th_pc:
        test_preds.append("Partially Compliant")
    else:
        test_preds.append("Not Addressed")

macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
acc = float(np.mean(np.array(test_preds) == y_test))

print("\n" + "=" * 72)
print("STATUTORY CONTEXT-CONDITIONED EMBEDDING CLASSIFIER")
print(f"Final Macro-F1: {macro_f1:.4f} | Accuracy: {acc * 100:.2f}%")
print("=" * 72)
print(classification_report(y_test, test_preds, digits=4, zero_division=0))

# 6. Save Model Artifacts
joblib.dump(clf, MODEL_DIR / "context_classifier.joblib")
eval_metrics = {
    "accuracy": round(acc, 4),
    "macro_f1": round(float(macro_f1), 4),
    "best_thresholds": {"Compliant": best_th_c, "Partially Compliant": best_th_pc},
    "model_type": "Context-Conditioned Sentence Transformer (all-MiniLM-L6-v2) + Calibrated Linear Head"
}
(MODEL_DIR / "eval_test.json").write_text(json.dumps(eval_metrics, indent=2))
print(f"[SUCCESS] Updated eval_test.json with Macro-F1 = {macro_f1:.4f}")