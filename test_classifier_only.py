"""
Quick test: ClauseClassifier on 20 clauses without API.
Reports latency and accuracy.
"""
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import ClauseClassifier

print("=" * 80)
print("CLASSIFIER-ONLY SMOKE TEST (20 clauses)")
print("=" * 80)

# Load corpus
df = pd.read_csv("corpus/labeled_clauses_bootstrap.csv")

# Select 20 clauses stratified by verdict
compliant = df[df["verdict"] == "Compliant"].sample(n=10, random_state=42)
partial = df[df["verdict"] == "Partially Compliant"].sample(n=5, random_state=42)
not_addressed = df[df["verdict"] == "Not Addressed"].sample(n=5, random_state=42)

test_df = pd.concat([compliant, partial, not_addressed]).reset_index(drop=True)

print(f"\n[INFO] Test set: 10 Compliant, 5 Partial, 5 Not Addressed")

# Initialize classifier
print(f"\n[INFO] Loading ClauseClassifier...")
init_start = time.time()
classifier = ClauseClassifier()
init_time = time.time() - init_start
print(f"  Initialization time: {init_time:.2f}s")

# Run predictions
print(f"\n{'='*80}")
print("PREDICTIONS")
print('='*80)

results = []
times = []
correct = 0

for idx, row in test_df.iterrows():
    ground_truth = row["verdict"]
    clause_text = row["clause_text"][:80] + "..." if len(row["clause_text"]) > 80 else row["clause_text"]
    
    start = time.time()
    result = classifier.predict(row["clause_text"], row.get("app_name", "Platform"))
    elapsed = time.time() - start
    
    times.append(elapsed)
    results.append(result)
    
    is_correct = result["verdict"] == ground_truth
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    print(f"{status} [{idx+1:2d}/20] GT:{ground_truth:20s} Pred:{result['verdict']:20s} "
          f"Conf:{result['confidence']:.3f} Time:{elapsed*1000:5.1f}ms")
    print(f"    Rule: {result['related_rule']['rule_id']} ({result['related_rule']['category']}, "
          f"sim={result['related_rule']['similarity']:.3f})")
    if idx < 19:  # Add separator except for last
        print()

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)
print(f"Total clauses:        20")
print(f"Correct predictions:  {correct}/20 ({correct/20*100:.1f}%)")
print(f"Average latency:      {np.mean(times)*1000:.1f}ms")
print(f"Min latency:          {np.min(times)*1000:.1f}ms")
print(f"Max latency:          {np.max(times)*1000:.1f}ms")
print(f"Std deviation:        {np.std(times)*1000:.1f}ms")

print(f"\nBreakdown by ground truth:")
for verdict in ["Compliant", "Partially Compliant", "Not Addressed"]:
    mask = test_df["verdict"] == verdict
    pred_correct = sum(1 for i, m in enumerate(mask) if m and results[i]["verdict"] == verdict)
    total = mask.sum()
    print(f"  {verdict:20s}: {pred_correct}/{total} correct")

print('='*80)

if correct >= 15:
    print("\n✓ TEST PASSED - Classifier working well")
    sys.exit(0)
else:
    print("\n⚠ TEST WARNING - Accuracy lower than expected")
    sys.exit(1)
