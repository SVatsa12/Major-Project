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

# 1. Load splits directly from train.csv (natural real-world distribution)
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

VALID_LABELS = ["Not Addressed", "Partially Compliant", "Compliant"]
train_df = train_df[train_df["verdict"].isin(VALID_LABELS)].copy()
val_df = val_df[val_df["verdict"].isin(VALID_LABELS)].copy()
test_df = test_df[test_df["verdict"].isin(VALID_LABELS)].copy()

print(f"[INFO] Train clauses: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 2. Extract MiniLM Dense Embeddings
print("[INFO] Loading all-MiniLM-L6-v2 embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[INFO] Encoding train, validation, and test clauses...")
X_train = embedder.encode(train_df["clause_text"].tolist(), show_progress_bar=False, normalize_embeddings=True)
y_train = train_df["verdict"].values

X_val = embedder.encode(val_df["clause_text"].tolist(), show_progress_bar=False, normalize_embeddings=True)
y_val = val_df["verdict"].values

X_test = embedder.encode(test_df["clause_text"].tolist(), show_progress_bar=False, normalize_embeddings=True)
y_test = test_df["verdict"].values

# 3. Train Regularized Logistic Classifier on Natural Class Distribution
clf = LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# 4. Calibrate Decision Thresholds on Validation Set (Zero Test Leakage)
classes = list(clf.classes_)
c_idx = classes.index("Compliant")
pc_idx = classes.index("Partially Compliant")
na_idx = classes.index("Not Addressed")

val_probs = clf.predict_proba(X_val)
test_probs = clf.predict_proba(X_test)

best_val_macro = 0.0
best_th_c = 0.35
best_th_pc = 0.40

for th_c in np.linspace(0.20, 0.60, 41):
    for th_pc in np.linspace(0.25, 0.65, 41):
        v_preds = []
        for p in val_probs:
            if p[c_idx] >= th_c and p[c_idx] >= p[pc_idx]:
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

# 5. Apply Calibrated Thresholds to Test Set
test_preds = []
for p in test_probs:
    if p[c_idx] >= best_th_c and p[c_idx] >= p[pc_idx]:
        test_preds.append("Compliant")
    elif p[pc_idx] >= best_th_pc:
        test_preds.append("Partially Compliant")
    else:
        test_preds.append("Not Addressed")

macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
acc = float(np.mean(np.array(test_preds) == y_test))

print("\n" + "=" * 70)
print(f"CALIBRATED SENTENCE TRANSFORMER (all-MiniLM-L6-v2) TEST RESULTS")
print(f"Macro-F1: {macro_f1:.4f} | Accuracy: {acc * 100:.2f}%")
print("=" * 70)
print(classification_report(y_test, test_preds, digits=4, zero_division=0))

# 6. Save Model Artifacts
joblib.dump(clf, MODEL_DIR / "embedding_classifier.joblib")
eval_metrics = {
    "accuracy": round(acc, 4),
    "macro_f1": round(float(macro_f1), 4),
    "calibrated_thresholds": {"Compliant": best_th_c, "Partially Compliant": best_th_pc},
    "model_type": "Dense Semantic Embeddings (all-MiniLM-L6-v2) + Calibrated Linear Classifier"
}
(MODEL_DIR / "eval_test.json").write_text(json.dumps(eval_metrics, indent=2))
print(f"[SUCCESS] eval_test.json updated with Macro-F1 = {macro_f1:.4f}")