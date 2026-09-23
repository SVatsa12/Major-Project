"""
src/training/train_baseline.py
Standard Baseline Compliance Classifier:
TF-IDF (1-2 ngrams) + Logistic Regression with standard class_weight='balanced'.
No artificial multipliers or non-standard heuristics.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

SPLITS_DIR = Path("datasets/splits")
MODEL_DIR = Path("models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Splits
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

X_train, y_train = train_df["clause_text"], train_df["verdict"]
X_val, y_val = val_df["clause_text"], val_df["verdict"]
X_test, y_test = test_df["clause_text"], test_df["verdict"]

# 2. Fit Standard TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=25000,
    sublinear_tf=True,
    stop_words="english"
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# 3. Standard Logistic Regression with balanced loss weighting
clf = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    C=1.0,
    random_state=42
)
clf.fit(X_train_vec, y_train)

# Active evaluation classes present in evaluation data
eval_labels = ["Compliant", "Partially Compliant", "Not Addressed"]

# 4. Standard Prediction (argmax over model probabilities)
y_val_pred = clf.predict(X_val_vec)
y_test_pred = clf.predict(X_test_vec)

# 5. Evaluate on Validation Set
print("=" * 60)
print(f"EVALUATION: VALIDATION SET ({len(y_val)} samples)")
print("=" * 60)
print(classification_report(y_val, y_val_pred, labels=eval_labels, zero_division=0))
val_macro_f1 = f1_score(y_val, y_val_pred, labels=eval_labels, average="macro", zero_division=0)
print(f"Validation Macro-F1: {val_macro_f1:.4f}")

# 6. Evaluate on Test Set
print("\n" + "=" * 60)
print(f"EVALUATION: TEST SET ({len(y_test)} samples)")
print("=" * 60)
print(classification_report(y_test, y_test_pred, labels=eval_labels, zero_division=0))
test_macro_f1 = f1_score(y_test, y_test_pred, labels=eval_labels, average="macro", zero_division=0)
print(f"Test Set Macro-F1: {test_macro_f1:.4f}")

print("\nConfusion Matrix (Test Set):")
cm = pd.DataFrame(
    confusion_matrix(y_test, y_test_pred, labels=eval_labels),
    index=[f"True: {c}" for c in eval_labels],
    columns=[f"Pred: {c}" for c in eval_labels]
)
print(cm)

# 7. Save Pipeline & Evaluation Output
bundle = {
    "vectorizer": vectorizer,
    "classifier": clf,
    "eval_labels": eval_labels
}
joblib.dump(bundle, MODEL_DIR / "baseline_model.joblib")

report_dict = classification_report(y_test, y_test_pred, labels=eval_labels, zero_division=0, output_dict=True)
report_dict["macro_f1"] = test_macro_f1
(MODEL_DIR / "eval_test.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
print(f"\nArtifacts saved to: {MODEL_DIR / 'baseline_model.joblib'}")