"""
Quick test: Verify ClauseClassifier loads and predicts correctly.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import ClauseClassifier

print("=" * 80)
print("QUICK CLASSIFIER TEST")
print("=" * 80)

# Load classifier
print("\n1. Loading ClauseClassifier...")
classifier = ClauseClassifier()

# Test prediction
test_clause = "We will delete your data within 30 days of your request."
print(f"\n2. Testing prediction on sample clause:")
print(f"   Clause: {test_clause}")

result = classifier.predict(test_clause)

print(f"\n3. Prediction result:")
print(f"   Verdict: {result['verdict']}")
print(f"   Confidence: {result['confidence']:.4f}")
print(f"   Probabilities:")
for cls, prob in result['probabilities'].items():
    print(f"     - {cls}: {prob:.4f}")
print(f"   Related Rule:")
print(f"     - ID: {result['related_rule']['rule_id']}")
print(f"     - Category: {result['related_rule']['category']}")
print(f"     - Similarity: {result['related_rule']['similarity']:.4f}")
print(f"     - Text: {result['related_rule']['statutory_text'][:100]}...")

print("\n" + "=" * 80)
print("TEST PASSED - Classifier loaded and predicted successfully")
print("=" * 80)
