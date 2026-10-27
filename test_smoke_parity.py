"""
Smoke Test: 20-clause parity between ClauseClassifier and API.

Tests:
0. Training/inference feature parity (predictor features == train_final_model features)
1. Direct ClauseClassifier predictions
2. API predictions via HTTP
3. Compare results for consistency
4. Report average latency for both

Selects 20 clauses:
- 10 Compliant
- 5 Partially Compliant
- 5 Not Addressed
"""
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import ClauseClassifier
from src.training.model_inputs import model_inputs
from src.training.train_final_model import build_feature_matrix

print("=" * 80)
print("20-CLAUSE SMOKE TEST: ClauseClassifier vs API")
print("=" * 80)

# Load corpus
df = pd.read_csv("corpus/labeled_clauses_bootstrap.csv")

# Select 20 clauses stratified by verdict
compliant = df[df["verdict"] == "Compliant"].sample(n=10, random_state=42)
partial = df[df["verdict"] == "Partially Compliant"].sample(n=5, random_state=42)
not_addressed = df[df["verdict"] == "Not Addressed"].sample(n=5, random_state=42)

test_df = pd.concat([compliant, partial, not_addressed]).reset_index(drop=True)

print(f"\n[INFO] Selected 20 test clauses:")
print(f"  - 10 Compliant")
print(f"  - 5 Partially Compliant")
print(f"  - 5 Not Addressed")

# Initialize ClauseClassifier
print(f"\n[INFO] Initializing ClauseClassifier...")
classifier = ClauseClassifier()

# Test 0: training/inference feature parity
print(f"\n{'='*80}")
print("TEST 0: Training vs Inference Feature Parity")
print('='*80)

train_parts = build_feature_matrix(model_inputs(test_df), classifier.taxonomy)
X_train_features = np.hstack(train_parts)
X_infer_features = np.vstack([
    classifier._build_features(row["clause_text"], row.get("app_name", "Platform"))
    for _, row in test_df.iterrows()
])
feature_max_diff = float(np.max(np.abs(X_train_features - X_infer_features)))
train_probs = classifier.model.predict_proba(X_train_features)
infer_probs = classifier.model.predict_proba(X_infer_features)
prob_max_diff = float(np.max(np.abs(train_probs - infer_probs)))
feature_parity = feature_max_diff < 1e-4 and prob_max_diff < 1e-3
print(f"  Max |feature diff|:     {feature_max_diff:.2e}")
print(f"  Max |probability diff|: {prob_max_diff:.2e}")
print(f"  {'✓' if feature_parity else '✗'} Feature parity {'OK' if feature_parity else 'FAILED'}")

# Test 1: Direct classifier predictions
print(f"\n{'='*80}")
print("TEST 1: Direct ClauseClassifier Predictions")
print('='*80)

classifier_results = []
classifier_times = []

for idx, row in test_df.iterrows():
    start = time.time()
    result = classifier.predict(row["clause_text"], row.get("app_name", "Platform"))
    elapsed = time.time() - start
    
    classifier_times.append(elapsed)
    classifier_results.append(result)
    
    print(f"  [{idx+1}/20] {row['verdict']:20s} -> {result['verdict']:20s} ({result['confidence']:.3f}) [{elapsed*1000:.1f}ms]")

avg_classifier_latency = np.mean(classifier_times) * 1000
print(f"\n[Classifier] Average latency: {avg_classifier_latency:.1f}ms")

# Test 2: API predictions
print(f"\n{'='*80}")
print("TEST 2: API Predictions (ensure server is running on localhost:8000)")
print('='*80)

API_URL = "http://localhost:8000/api/predict"

print(f"\n[INFO] Checking API health...")
try:
    health_response = requests.get("http://localhost:8000/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("model_loaded"):
            print(f"  ✓ API is healthy, model loaded")
        else:
            print(f"  ⚠ API is starting, model not yet loaded")
            time.sleep(5)  # Wait for model to load
    else:
        print(f"  ✗ API health check failed: {health_response.status_code}")
        print(f"\n[ERROR] Please start the API server first:")
        print(f"  python src/api/server.py")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print(f"  ✗ Could not connect to API at http://localhost:8000")
    print(f"\n[ERROR] Please start the API server first:")
    print(f"  python src/api/server.py")
    sys.exit(1)

api_results = []
api_times = []

for idx, row in test_df.iterrows():
    payload = {
        "clause_text": row["clause_text"],
        "app_name": row.get("app_name", "Platform")
    }
    
    start = time.time()
    response = requests.post(API_URL, json=payload, timeout=30)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        api_times.append(elapsed)
        api_results.append(result)
        
        print(f"  [{idx+1}/20] {row['verdict']:20s} -> {result['verdict']:20s} ({result['confidence']:.3f}) [{elapsed*1000:.1f}ms]")
    else:
        print(f"  [{idx+1}/20] ✗ API error: {response.status_code} {response.text[:100]}")
        api_results.append(None)
        api_times.append(elapsed)

avg_api_latency = np.mean(api_times) * 1000
print(f"\n[API] Average latency: {avg_api_latency:.1f}ms")

# Test 3: Compare results
print(f"\n{'='*80}")
print("TEST 3: Parity Check")
print('='*80)

matches = 0
mismatches = []

for idx in range(20):
    cls_result = classifier_results[idx]
    api_result = api_results[idx]
    
    if api_result is None:
        mismatches.append({
            "idx": idx,
            "reason": "API error",
            "ground_truth": test_df.iloc[idx]["verdict"]
        })
        continue
    
    verdict_match = cls_result["verdict"] == api_result["verdict"]
    conf_diff = abs(cls_result["confidence"] - api_result["confidence"])
    
    # Check if related_rule matches
    cls_rule_id = cls_result["related_rule"]["rule_id"]
    api_rule_id = api_result["related_rule"]["rule_id"]
    rule_match = cls_rule_id == api_rule_id
    
    if verdict_match and conf_diff < 0.01 and rule_match:
        matches += 1
    else:
        mismatches.append({
            "idx": idx,
            "verdict_match": verdict_match,
            "conf_diff": conf_diff,
            "rule_match": rule_match,
            "classifier_verdict": cls_result["verdict"],
            "api_verdict": api_result["verdict"],
            "classifier_conf": cls_result["confidence"],
            "api_conf": api_result["confidence"],
            "classifier_rule": cls_rule_id,
            "api_rule": api_rule_id,
            "ground_truth": test_df.iloc[idx]["verdict"]
        })

print(f"\n[Parity] {matches}/20 predictions match exactly")

if mismatches:
    print(f"\n[Mismatches] Found {len(mismatches)} discrepancies:")
    for mm in mismatches[:5]:  # Show first 5
        print(f"  Row {mm['idx']}:")
        if 'reason' in mm:
            print(f"    Reason: {mm['reason']}")
        else:
            if not mm['verdict_match']:
                print(f"    Verdict: Classifier={mm['classifier_verdict']}, API={mm['api_verdict']}")
            if mm['conf_diff'] >= 0.01:
                print(f"    Confidence diff: {mm['conf_diff']:.4f}")
            if not mm['rule_match']:
                print(f"    Rule: Classifier={mm['classifier_rule']}, API={mm['api_rule']}")

# Final summary
print(f"\n{'='*80}")
print("SMOKE TEST SUMMARY")
print('='*80)
print(f"Test clauses: 20")
print(f"  - 10 Compliant")
print(f"  - 5 Partially Compliant")
print(f"  - 5 Not Addressed")
print(f"\nLatency:")
print(f"  Classifier: {avg_classifier_latency:.1f}ms avg")
print(f"  API:        {avg_api_latency:.1f}ms avg")
print(f"  Overhead:   {avg_api_latency - avg_classifier_latency:.1f}ms")
print(f"\nTraining/inference feature parity: {'OK' if feature_parity else 'FAILED'} (max feature diff {feature_max_diff:.2e})")
print(f"\nParity:")
print(f"  Exact matches: {matches}/20 ({matches/20*100:.1f}%)")
print(f"  Discrepancies: {len(mismatches)}/20 ({len(mismatches)/20*100:.1f}%)")

if not feature_parity:
    print(f"\n✗ SMOKE TEST FAILED - Training and inference features differ")
    exit_code = 1
elif matches == 20:
    print(f"\n✓ SMOKE TEST PASSED - Perfect parity!")
    exit_code = 0
elif matches >= 18:
    print(f"\n⚠ SMOKE TEST PASSED WITH WARNINGS - Minor discrepancies")
    exit_code = 0
else:
    print(f"\n✗ SMOKE TEST FAILED - Significant discrepancies")
    exit_code = 1

print('='*80)
sys.exit(exit_code)
