"""
Fine-tuning law-ai/InLegalBERT for Policy Clause Compliance Classification.

Input format (context-conditioned):
    clause_text + " [SEP] Category: " + dpdp_category + " | Rule: " + best_match_taxonomy_id

Training data: datasets/splits/train_augmented.csv  (rebalanced via augment_minority.py)
Eval/Test data: datasets/splits/val.csv, test.csv   (original, untouched)

Loss: Standard Cross-Entropy (no class weights, no label smoothing).
"""

import sys
import io
# Force UTF-8 stdout on Windows to avoid UnicodeEncodeError from special chars in clause text
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
SPLITS_DIR = Path("datasets/splits")
MODEL_OUTPUT_DIR = Path("models/inlegalbert_classifier")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "law-ai/InLegalBERT"
MAX_LENGTH = 256   # All NLI inputs are <=359 tokens (mean=89); 256 covers 99%+ while saving CPU memory
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.15

# ---------------------------------------------------------------------------
# 1. Label Mappings (3-class compliance taxonomy)
# ---------------------------------------------------------------------------
LABEL2ID = {"Not Addressed": 0, "Partially Compliant": 1, "Compliant": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# 2. Load Splits
#    Train: augmented/rebalanced split
#    Val / Test: original untouched splits
# ---------------------------------------------------------------------------
train_df = pd.read_csv(SPLITS_DIR / "train_augmented.csv")
val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
test_df  = pd.read_csv(SPLITS_DIR / "test.csv")

# Filter to only the 3 target classes and cast labels to int
for df in [train_df, val_df, test_df]:
    df.drop(df[~df["verdict"].isin(LABEL2ID)].index, inplace=True)
    df["label"] = df["verdict"].map(LABEL2ID).astype(int)
    df.reset_index(drop=True, inplace=True)

print(f"Train samples : {len(train_df)}  |  distribution: {dict(train_df['verdict'].value_counts())}")
print(f"Val samples   : {len(val_df)}  |  distribution: {dict(val_df['verdict'].value_counts())}")
print(f"Test samples  : {len(test_df)}  |  distribution: {dict(test_df['verdict'].value_counts())}")

# ---------------------------------------------------------------------------
# 3. Context-Conditioned Input Construction
#    Format: clause_text [SEP] Category: <cat> | Rule: <rule_id>
#    Missing values default to "General" / "None" (no leakage — these are
#    metadata fields from upstream retrieval, not from the labeler rationale).
# ---------------------------------------------------------------------------
def build_nli_input(df: pd.DataFrame) -> pd.Series:
    cat  = df["dpdp_category"].fillna("General").replace("", "General")
    rule = df["best_match_taxonomy_id"].fillna("None").replace("", "None").replace("NONE", "None")
    return df["clause_text"] + " [SEP] Category: " + cat + " | Rule: " + rule

for df in [train_df, val_df, test_df]:
    df["nli_input"] = build_nli_input(df)

# ---------------------------------------------------------------------------
# 4. Tokenization
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch):
    return tokenizer(
        batch["nli_input"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

train_ds = Dataset.from_pandas(train_df[["nli_input", "label"]]).map(tokenize_function, batched=True)
val_ds   = Dataset.from_pandas(val_df[["nli_input", "label"]]).map(tokenize_function, batched=True)
test_ds  = Dataset.from_pandas(test_df[["nli_input", "label"]]).map(tokenize_function, batched=True)

# ---------------------------------------------------------------------------
# 5. Metrics — Macro-F1 + per-class F1 scores
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1":              macro_f1,
        "f1_not_addressed":      per_class[0],
        "f1_partially_compliant": per_class[1],
        "f1_compliant":          per_class[2],
    }

# ---------------------------------------------------------------------------
# 6. Model
# ---------------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# ---------------------------------------------------------------------------
# 7. Training Arguments
#    fp16 only when CUDA is available (CPU training uses fp32)
#    warmup_ratio not accepted in this transformers version — compute warmup_steps manually
# ---------------------------------------------------------------------------
use_fp16 = torch.cuda.is_available()

steps_per_epoch = max(1, len(train_df) // BATCH_SIZE_TRAIN)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

training_args = TrainingArguments(
    output_dir="./models/inlegalbert_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,           # 15% of total steps (warmup_ratio not in this transformers build)
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=20,
    fp16=use_fp16,
    seed=42,
    report_to="none",
)

# ---------------------------------------------------------------------------
# 8. Standard Trainer (no class weights, no custom loss)
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ---------------------------------------------------------------------------
# 9. Train
# ---------------------------------------------------------------------------
print(f"\nStarting InLegalBERT fine-tuning...")
print(f"  Train: {len(train_df)} samples | Val: {len(val_df)} samples")
print(f"  Epochs: {EPOCHS} | LR: {LEARNING_RATE} | fp16: {use_fp16}\n")
trainer.train()

# ---------------------------------------------------------------------------
# 10. Test Set Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

test_predictions = trainer.predict(test_ds)
logits_np = test_predictions.predictions          # shape (N, 3)
y_pred_raw = np.argmax(logits_np, axis=-1)
y_true = test_df["label"].values

target_names = [ID2LABEL[i] for i in range(len(LABEL2ID))]
print(classification_report(y_true, y_pred_raw, target_names=target_names, zero_division=0))
macro_f1 = f1_score(y_true, y_pred_raw, average="macro", zero_division=0)
print(f"Test Set Macro-F1: {macro_f1:.4f}")

cm = pd.DataFrame(
    confusion_matrix(y_true, y_pred_raw),
    index=[f"True: {t}" for t in target_names],
    columns=[f"Pred: {t}" for t in target_names],
)
print("\nConfusion Matrix:")
print(cm)

# ---------------------------------------------------------------------------
# 11. Threshold Calibration on Validation Set
#     Grid-search probability thresholds for Compliant and Partially Compliant
#     that maximise Macro-F1, without modifying model weights.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("THRESHOLD CALIBRATION ON VALIDATION SET")
print("=" * 60)

val_predictions = trainer.predict(val_ds)
val_logits = val_predictions.predictions
val_probs = torch.softmax(torch.tensor(val_logits, dtype=torch.float32), dim=-1).numpy()
y_val_true = val_df["label"].values

best_val_f1 = -1.0
best_thresh = {"compliant": 0.5, "partially_compliant": 0.5}

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

for t_c in thresholds:
    for t_pc in thresholds:
        y_cal = np.full(len(val_probs), 0)  # default = Not Addressed
        for i, p in enumerate(val_probs):
            if p[2] >= t_c:
                y_cal[i] = 2  # Compliant
            elif p[1] >= t_pc:
                y_cal[i] = 1  # Partially Compliant
        f1 = f1_score(y_val_true, y_cal, average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = {"compliant": t_c, "partially_compliant": t_pc}

print(f"Best validation Macro-F1 with calibration: {best_val_f1:.4f}")
print(f"Optimal thresholds: Compliant >= {best_thresh['compliant']}, "
      f"Partially Compliant >= {best_thresh['partially_compliant']}")

# Apply best thresholds to test set
probs_test = torch.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=-1).numpy()
y_pred_cal = np.full(len(probs_test), 0)
for i, p in enumerate(probs_test):
    if p[2] >= best_thresh["compliant"]:
        y_pred_cal[i] = 2
    elif p[1] >= best_thresh["partially_compliant"]:
        y_pred_cal[i] = 1

print("\n--- Test Set Results WITH Threshold Calibration ---")
print(classification_report(y_true, y_pred_cal, target_names=target_names, zero_division=0))
macro_f1_cal = f1_score(y_true, y_pred_cal, average="macro", zero_division=0)
print(f"Test Set Macro-F1 (calibrated): {macro_f1_cal:.4f}")

# ---------------------------------------------------------------------------
# 12. Save Final Artifacts
# ---------------------------------------------------------------------------
trainer.save_model(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

# Full classification report (using calibrated predictions for final output)
report = classification_report(
    y_true, y_pred_cal, target_names=target_names, zero_division=0, output_dict=True
)
report["macro_f1"] = macro_f1_cal
report["macro_f1_argmax"] = macro_f1          # also record raw argmax score
report["calibration_thresholds"] = best_thresh

(MODEL_OUTPUT_DIR / "eval_test_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

# Save thresholds separately for inference pipeline
(MODEL_OUTPUT_DIR / "thresholds.json").write_text(
    json.dumps(best_thresh, indent=2), encoding="utf-8"
)

print(f"\nModel and tokenizer saved to : {MODEL_OUTPUT_DIR}")
print(f"Evaluation report saved to   : {MODEL_OUTPUT_DIR / 'eval_test_report.json'}")
print(f"Calibration thresholds saved : {MODEL_OUTPUT_DIR / 'thresholds.json'}")