"""
src/evaluation/diagnose_test_errors.py
Inspects raw probability outputs and text for True Compliant and borderline test samples.
"""

from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SPLITS_DIR = Path("datasets/splits")
MODEL_DIR = Path("models/inlegalbert_classifier")

test_df = pd.read_csv(SPLITS_DIR / "test.csv")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

LABEL2ID = {"Not Addressed": 0, "Partially Compliant": 1, "Compliant": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Isolate the 3 True Compliant samples
comp_samples = test_df[test_df["verdict"] == "Compliant"]

print("=" * 80)
print("INSPECTION OF TRUE COMPLIANT TEST SAMPLES & MODEL PREDICTIONS")
print("=" * 80)

for idx, (_, row) in enumerate(comp_samples.iterrows(), 1):
    text = row["clause_text"]
    encoded = tokenizer(text, truncation=True, max_length=384, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    prob_dict = {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(3)}
    pred_label = ID2LABEL[int(probs.argmax())]
    
    print(f"\n[Sample {idx}] App: {row['app_name']} | ID: {row['clause_id']}")
    print(f"Clause Text: {text[:180]}...")
    print(f"Probabilities: {prob_dict}")
    print(f"Argmax Prediction: {pred_label}")
    print("-" * 80)