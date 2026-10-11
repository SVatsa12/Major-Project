#!/usr/bin/env python
"""Quick test of feature extraction and training logic."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.training.feature_extraction import extract_actionability_features
from src.training.train_classifier import build_context_texts

# Load minimal data
df = pd.read_csv("corpus/labeled_clauses_bootstrap.csv").head(50)  # Just 50 for testing
taxonomy = json.loads(Path("corpus/dpdp_taxonomy_final.json").read_text())
tax_map = {item["taxonomy_id"]: item for item in taxonomy}

print(f"[TEST] Loaded {len(df)} clauses")
print(f"[TEST] Class distribution:\n{df['verdict'].value_counts()}")

# Test feature extraction
print("\n[TEST] Testing feature extraction...")
features = extract_actionability_features(df["clause_text"].values)
print(f"[TEST] Extracted features shape: {features.shape}")
print(f"[TEST] Feature 0 (hedge_density) sample: {features[0, 0]}")
print(f"[TEST] Feature 7 (parental_marker) sample: {features[0, 7]}")

# Test context building
print("\n[TEST] Testing context text building...")
contexts = build_context_texts(df, tax_map)
print(f"[TEST] Built {len(contexts)} context texts")
print(f"[TEST] Sample context: {contexts[0][:100]}...")

# Test RobustScaler on training fold
print("\n[TEST] Testing RobustScaler (fit on training only)...")
train_idx = np.random.choice(len(df), size=int(0.8 * len(df)), replace=False)
test_idx = np.array([i for i in range(len(df)) if i not in train_idx])
X_train_features = features[train_idx]
X_test_features = features[test_idx]
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)
print(f"[TEST] Scaler fit on {len(train_idx)} samples, transform test {len(test_idx)} samples")
print(f"[TEST] Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

# Test dummy embedding + classifier
print("\n[TEST] Testing classifier pipeline...")
X_train_combined = np.hstack([np.random.randn(len(train_idx), 384), X_train_scaled])
X_test_combined = np.hstack([np.random.randn(len(test_idx), 384), X_test_scaled])
y_train = np.array([df.iloc[i]["verdict"] for i in train_idx])
y_test = np.array([df.iloc[i]["verdict"] for i in test_idx])

clf = LogisticRegression(
    C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
    multi_class="multinomial", class_weight="balanced", random_state=42
)
clf.fit(X_train_combined, y_train)
probs = clf.predict_proba(X_test_combined)
print(f"[TEST] Classifier trained on {X_train_combined.shape}, probabilities shape: {probs.shape}")
print(f"[TEST] Probability sample: {probs[0]}")

# Test threshold calibration logic
print("\n[TEST] Testing threshold calibration logic...")
classes = list(clf.classes_)
print(f"[TEST] Classes: {classes}")
c_idx = classes.index("Compliant")
p_idx = classes.index("Partially Compliant")
n_idx = classes.index("Not Addressed")
print(f"[TEST] Class indices: Compliant={c_idx}, Partially Compliant={p_idx}, Not Addressed={n_idx}")

tau_c, tau_p = 0.5, 0.35
preds = []
for i in range(len(probs)):
    p_c = probs[i, c_idx]
    p_p = probs[i, p_idx]
    if p_c >= tau_c:
        preds.append("Compliant")
    elif p_p >= tau_p:
        preds.append("Partially Compliant")
    else:
        preds.append("Not Addressed")
print(f"[TEST] Made {len(preds)} predictions with thresholds tau_c={tau_c}, tau_p={tau_p}")
print(f"[TEST] Prediction distribution: {pd.Series(preds).value_counts().to_dict()}")

print("\n[SUCCESS] All quick tests passed!")
