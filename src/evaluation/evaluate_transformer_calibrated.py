"""
src/evaluation/evaluate_transformer_calibrated.py
Evaluates the saved InLegalBERT checkpoint using validation-calibrated
decision thresholds to accurately capture the Compliant minority class.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SPLITS_DIR = Path("datasets/splits")
MODEL_DIR = Path("models/inlegalbert_classifier")

# 1. Load Data
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

LABEL2ID = {"Not Addressed": 0, "Partially Compliant": 1, "Compliant": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

val_df["label"] = val_df["verdict"].map(LABEL2ID)
test_df["label"] = test_df["verdict"].map(LABEL2ID)

# 2. Load Trained Model & Tokenizer
print("Loading trained InLegalBERT checkpoint from disk...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_probabilities(df_split):
    encoded = tokenizer(
        df_split["clause_text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=384,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

print("Extracting prediction probabilities on Validation and Test sets...")
val_probs = get_probabilities(val_df)
test_probs = get_probabilities(test_df)

# 3. Decision Function with Calibrated Compliant Threshold
def predict_with_threshold(prob_matrix, compliant_thresh=0.25):
    preds = []
    comp_idx = LABEL2ID["Compliant"]
    part_idx = LABEL2ID["Partially Compliant"]
    not_idx = LABEL2ID["Not Addressed"]

    for p in prob_matrix:
        if p[comp_idx] >= compliant_thresh:
            preds.append(comp_idx)
        else:
            preds.append(part_idx if p[part_idx] >= p[not_idx] else not_idx)
    return np.array(preds)

# 4. Tune Threshold on Validation Set
best_thresh = 0.25
best_val_macro_f1 = 0.0
y_val = val_df["label"].values

for thresh in np.arange(0.15, 0.45, 0.02):
    preds = predict_with_threshold(val_probs, compliant_thresh=thresh)
    score = f1_score(y_val, preds, average="macro", zero_division=0)
    if score > best_val_macro_f1:
        best_val_macro_f1 = score
        best_thresh = thresh

print(f"\nOptimal Compliant Threshold calibrated on Validation set: {best_thresh:.2f}")
print(f"Validation Macro-F1 at tuned threshold: {best_val_macro_f1:.4f}")

# 5. Evaluate on Test Set
y_test = test_df["label"].values
test_preds = predict_with_threshold(test_probs, compliant_thresh=best_thresh)

target_names = [ID2LABEL[i] for i in range(len(LABEL2ID))]
print("\n" + "=" * 60)
print("INLEGALBERT TEST SET EVALUATION (CALIBRATED THRESHOLD)")
print("=" * 60)
print(classification_report(y_test, test_preds, target_names=target_names, zero_division=0))

macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
print(f"Calibrated Test Set Macro-F1: {macro_f1:.4f}")

cm = pd.DataFrame(
    confusion_matrix(y_test, test_preds),
    index=[f"True: {t}" for t in target_names],
    columns=[f"Pred: {t}" for t in target_names]
)
print("\nConfusion Matrix:")
print(cm)

# Save Calibrated Evaluation Report
report = classification_report(y_test, test_preds, target_names=target_names, zero_division=0, output_dict=True)
report["calibrated_macro_f1"] = macro_f1
report["calibrated_threshold"] = float(best_thresh)
(MODEL_DIR / "eval_calibrated_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nSaved calibrated evaluation report to {MODEL_DIR / 'eval_calibrated_test_report.json'}")