"""
Fine-tuning law-ai/InLegalBERT for Policy Clause Compliance Classification.
Uses square-root smoothed Cross-Entropy Loss to handle class imbalance across
Not Addressed, Partially Compliant, and Compliant.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

SPLITS_DIR = Path("datasets/splits")
MODEL_OUTPUT_DIR = Path("models/inlegalbert_classifier")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "law-ai/InLegalBERT"
MAX_LENGTH = 384
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1.5e-5

# 1. Label Mappings (3-class compliance taxonomy)
LABEL2ID = {"Not Addressed": 0, "Partially Compliant": 1, "Compliant": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 2. Load Splits
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

# Filter out non-target classes (e.g. solitary Non-Compliant) and cast to integer
train_df = train_df[train_df["verdict"].isin(LABEL2ID)].copy()
val_df = val_df[val_df["verdict"].isin(LABEL2ID)].copy()
test_df = test_df[test_df["verdict"].isin(LABEL2ID)].copy()

for df in [train_df, val_df, test_df]:
    df["label"] = df["verdict"].map(LABEL2ID).astype(int)

# Compute Smoothed (Square-Root) Class Weights for balanced loss
class_counts = train_df["label"].value_counts().sort_index().values
total_samples = len(train_df)
raw_weights = total_samples / (len(LABEL2ID) * class_counts)
smoothed_weights = np.sqrt(raw_weights)
weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float)

print("Calculated Smoothed Class Weights:")
for label_name, idx in LABEL2ID.items():
    print(f"  {label_name}: {smoothed_weights[idx]:.3f}")

# 3. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch):
    return tokenizer(
        batch["clause_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

train_ds = Dataset.from_pandas(train_df[["clause_text", "label"]]).map(tokenize_function, batched=True)
val_ds = Dataset.from_pandas(val_df[["clause_text", "label"]]).map(tokenize_function, batched=True)
test_ds = Dataset.from_pandas(test_df[["clause_text", "label"]]).map(tokenize_function, batched=True)

# 4. Custom Trainer with Smoothed Weighted Cross-Entropy Loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 5. Metrics Computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"macro_f1": macro_f1}

# 6. Training Configuration
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

total_steps = (len(train_df) // BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)

training_args = TrainingArguments(
    output_dir="./models/inlegalbert_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.02,
    warmup_steps=warmup_steps,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=15,
    seed=42,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 7. Train Model
print(f"\nStarting InLegalBERT fine-tuning ({len(train_df)} train samples, {len(val_df)} val samples)...")
trainer.train()

# 8. Test Set Evaluation
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)
test_predictions = trainer.predict(test_ds)
y_pred = np.argmax(test_predictions.predictions, axis=-1)
y_true = test_df["label"].values

target_names = [ID2LABEL[i] for i in range(len(LABEL2ID))]
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"Test Set Macro-F1: {macro_f1:.4f}")

cm = pd.DataFrame(
    confusion_matrix(y_true, y_pred),
    index=[f"True: {t}" for t in target_names],
    columns=[f"Pred: {t}" for t in target_names]
)
print("\nConfusion Matrix:")
print(cm)

# 9. Save Final Artifacts
trainer.save_model(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
report["macro_f1"] = macro_f1
(MODEL_OUTPUT_DIR / "eval_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nModel and tokenizer successfully saved to: {MODEL_OUTPUT_DIR}")