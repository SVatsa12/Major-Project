"""
Train final model on full dataset and save as context_classifier.joblib.
This script trains the classifier on the entire bootstrap corpus using median thresholds.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.training.feature_extraction import extract_actionability_features
from src.training.train_classifier import build_context_texts

CORPUS_PATH = Path("corpus/labeled_clauses_bootstrap.csv")
TAXONOMY_PATH = Path("corpus/dpdp_taxonomy_final.json")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CORPUS_PATH)
taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
tax_map = {item["taxonomy_id"]: item for item in taxonomy}

print(f"[INFO] Training final model on {len(df)} clauses...")

# Build context texts
context_texts = build_context_texts(df, tax_map)

# Generate embeddings
print("[INFO] Generating embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    context_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Extract features
print("[INFO] Extracting actionability features...")
features = extract_actionability_features(df["clause_text"].values)

# Scale features
print("[INFO] Fitting RobustScaler...")
scaler = RobustScaler()
features_scaled = scaler.fit_transform(features)

# Combine features
X_combined = np.hstack([embeddings, features_scaled])
y = df["verdict"].values

# Train final classifier
print("[INFO] Training LogisticRegression on full dataset...")
clf = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    multi_class="multinomial",
    class_weight="balanced",
    random_state=42
)
clf.fit(X_combined, y)

# Create model artifact
model_artifact = {
    "classifier": clf,
    "scaler": scaler,
    "embedder_model_name": "all-MiniLM-L6-v2",
    "feature_names": [
        "hedge_density",
        "action_verb_count",
        "has_ui_path",
        "has_contact_mechanism",
        "has_timeline_sla",
        "commitment_ratio",
        "clause_length_norm",
        "parental_verification_marker"
    ],
    "context_format": "[{app_name}] Mandate: {checkable_test} | Policy: {clause_text}",
    "threshold_compliant": 0.55,  # Median from CV
    "threshold_partial": 0.4,     # Median from CV
    "classes": list(clf.classes_),
    "training_data": {
        "corpus_path": str(CORPUS_PATH),
        "total_clauses": len(df),
        "distribution": {
            "Not Addressed": int((y == "Not Addressed").sum()),
            "Partially Compliant": int((y == "Partially Compliant").sum()),
            "Compliant": int((y == "Compliant").sum()),
        }
    }
}

# Save model
model_path = MODELS_DIR / "context_classifier.joblib"
joblib.dump(model_artifact, model_path)
print(f"\n[SUCCESS] Model saved to {model_path}")

# Verify
loaded = joblib.load(model_path)
print(f"[VERIFY] Loaded model has classifier: {loaded['classifier'] is not None}")
print(f"[VERIFY] Loaded model has scaler: {loaded['scaler'] is not None}")
print(f"[VERIFY] Model is ready for deployment!")
