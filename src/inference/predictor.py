"""
src/inference/predictor.py

ClauseClassifier: Single source of truth for DPDP compliance prediction.

Loads the final model (E5+825 features, ordinal LogReg) and provides prediction
interface by importing the EXACT feature extraction functions used in training.

Usage:
    classifier = ClauseClassifier()
    result = classifier.predict("Your policy clause text here")
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# Import the EXACT functions used in training (single source of truth)
from src.training.model_inputs import build_input_texts, encode_texts, encoder_prefix, model_inputs
from src.training.feature_extraction import extract_actionability_features
from src.evaluation.incremental_features import all_rules_similarity_from_embeddings, get_active_rules
from src.training.ordinal import OrdinalLRCV
from src.training.train_classifier import apply_thresholds


class ClauseClassifier:
    """
    DPDP Compliance Classifier using the final model (E5-base + 8 actionability + 49 all-rules similarity,
    ordinal LogReg: cumulative P(y > Not Addressed), P(y > Partially Compliant); scaling inside the model).
    
    Loads models/final_classifier/ artifacts and provides prediction interface.
    Feature extraction imports the exact functions used in src/training/ (single source of truth).
    """
    
    def __init__(self, model_dir: str = "models/final_classifier", taxonomy_path: str = "corpus/dpdp_taxonomy_final.json"):
        """
        Initialize classifier by loading model artifacts and caching embedder + taxonomy.
        
        Args:
            model_dir: Path to final_classifier directory
            taxonomy_path: Path to taxonomy JSON
        """
        self.model_dir = Path(model_dir)
        self.taxonomy_path = Path(taxonomy_path)
        
        print(f"[ClauseClassifier] Loading model from {self.model_dir}...")
        start_time = time.time()
        
        # Load model artifacts (RobustScaler is fit inside the ordinal model)
        self.model = joblib.load(self.model_dir / "model.joblib")

        with open(self.model_dir / "thresholds.json", 'r') as f:
            thresholds = json.load(f)
            self.tau_compliant = thresholds["tau_compliant"]
            self.tau_partial = thresholds["tau_partial"]
        
        with open(self.model_dir / "feature_config.json", 'r') as f:
            self.feature_config = json.load(f)
        
        with open(self.model_dir / "model_config.json", 'r') as f:
            self.model_config = json.load(f)
        
        # Verify model configuration
        assert self.feature_config["total_features"] == 825, \
            f"Expected 825 features, got {self.feature_config['total_features']}"
        assert self.feature_config["encoder_model"] == "intfloat/e5-base-v2", \
            f"Expected intfloat/e5-base-v2, got {self.feature_config['encoder_model']}"
        self.encoder_name = self.feature_config["encoder_model"]
        assert self.model_config["model_type"] == "OrdinalLogisticRegression" and isinstance(self.model, OrdinalLRCV), \
            f"Expected the ordinal final model, got {self.model_config['model_type']} / {type(self.model).__name__}; " \
            f"retrain with src/training/train_final_model.py"

        # The model must have been trained on label-free "query: " + clause_text inputs
        input_text = self.feature_config.get("input_text", {})
        assert input_text.get("llm_output_columns_used") is False and \
            input_text.get("encoder_prefix") == encoder_prefix(self.encoder_name) and \
            input_text.get("include_app_name") is False, \
            f"Model was not trained on label-free inputs (input_text={input_text}); retrain with src/training/train_final_model.py"

        # Load taxonomy
        print(f"[ClauseClassifier] Loading taxonomy from {self.taxonomy_path}...")
        with open(self.taxonomy_path, 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)
        
        # Filter out NONE taxonomy items for rule similarity
        self.active_rules = get_active_rules(self.taxonomy)
        print(f"[ClauseClassifier] Loaded {len(self.active_rules)} active taxonomy rules")
        
        # Load and cache E5-base embedder
        print(f"[ClauseClassifier] Loading E5-base embedder (first load may take ~30s)...")
        embedder_start = time.time()
        self.embedder = SentenceTransformer(self.encoder_name)
        embedder_time = time.time() - embedder_start
        print(f"[ClauseClassifier] E5-base loaded in {embedder_time:.2f}s")
        
        # Pre-compute and cache taxonomy rule embeddings (38 rules)
        print(f"[ClauseClassifier] Pre-computing taxonomy rule embeddings...")
        rule_texts = [item["requirement"] for item in self.active_rules]
        self.rule_embeddings = encode_texts(self.embedder, rule_texts, self.encoder_name)
        
        self.rule_ids = [item["taxonomy_id"] for item in self.active_rules]
        self.rule_categories = [item.get("category", "Unknown") for item in self.active_rules]
        
        total_time = time.time() - start_time
        print(f"[ClauseClassifier] Initialization complete in {total_time:.2f}s")
        print(f"[ClauseClassifier] Model: {self.model_config['model_type']}, C={self.model_config['hyperparameters']['C']}")
        print(f"[ClauseClassifier] Features: {self.feature_config['feature_breakdown']}")
        print(f"[ClauseClassifier] Thresholds: tau_compliant={self.tau_compliant:.3f}, tau_partial={self.tau_partial:.3f}")
        print(f"[ClauseClassifier] Classes: {self.model.classes_.tolist()}")
    
    def _build_features(self, clause_text: str, app_name: str = "Platform") -> np.ndarray:
        """
        Build 825-dimensional feature vector by importing training functions.
        
        Args:
            clause_text: Policy clause text
            app_name: Platform name (accepted for API compatibility; not a model input)
        
        Returns:
            np.ndarray of shape (825,) with features in training order
        """
        # Only clause_text/app_name reach the feature builders (same as training)
        inputs = model_inputs(pd.DataFrame([{"clause_text": clause_text, "app_name": app_name}]))
        
        # 1. E5-base embeddings (768 dims): "query: " + clause_text, as in training
        embeddings = encode_texts(self.embedder, build_input_texts(inputs), self.encoder_name)
        
        # 2. Actionability features (8 dims) - using training function
        action_features = extract_actionability_features(inputs["clause_text"].values)
        
        # 3. All-rules similarity features (49 dims) - training function on cached rule embeddings
        all_rules_features = all_rules_similarity_from_embeddings(embeddings, self.rule_embeddings, self.rule_categories)
        
        # Combine in training order: embeddings (768) + actionability (8) + all-rules (49)
        features = np.hstack([embeddings, action_features, all_rules_features])
        
        # Assert feature count matches training
        assert features.shape == (1, 825), \
            f"Feature mismatch: expected (1, 825), got {features.shape}"
        
        return features[0]  # Return 1D array
    
    def predict(self, clause_text: str, app_name: str = "Platform") -> Dict[str, Any]:
        """
        Predict compliance verdict for a policy clause.
        
        Args:
            clause_text: Policy clause text
            app_name: Platform name (for context)
        
        Returns:
            dict with keys:
                - verdict: str (Compliant | Partially Compliant | Not Addressed)
                - confidence: float (probability of predicted class)
                - probabilities: dict mapping class -> probability
                - related_rule: dict with most similar taxonomy rule info
                    - rule_id: taxonomy ID
                    - statutory_text: requirement text
                    - category: rule category
                    - similarity: cosine similarity score
        """
        # Build features
        features = self._build_features(clause_text, app_name)
        
        # Get probabilities (the ordinal model scales features internally)
        probs = self.model.predict_proba(features.reshape(1, -1))

        # Build probability dict
        prob_dict = {cls: float(probs[0, i]) for i, cls in enumerate(self.model.classes_)}

        # Apply inner-CV-tuned thresholds (same decision rule as evaluation)
        verdict = apply_thresholds(probs, self.model.classes_, self.tau_compliant, self.tau_partial)[0]

        confidence = prob_dict[verdict]
        
        # Find most related taxonomy rule (highest similarity)
        # Similarities are in features[768+8 : 768+8+38]
        rule_similarities = features[768 + 8 : 768 + 8 + 38]
        best_rule_idx = int(np.argmax(rule_similarities))
        best_similarity = float(rule_similarities[best_rule_idx])
        
        best_rule = self.active_rules[best_rule_idx]
        related_rule = {
            "rule_id": best_rule["taxonomy_id"],
            "statutory_text": best_rule["requirement"],
            "category": best_rule.get("category", "Unknown"),
            "similarity": best_similarity
        }
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "probabilities": prob_dict,
            "related_rule": related_rule
        }
    
    def predict_batch(self, clauses: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Predict compliance for multiple clauses.
        
        Args:
            clauses: List of dicts with keys 'clause_text' and optionally 'app_name'
        
        Returns:
            List of prediction dicts (same format as predict())
        """
        results = []
        for clause_dict in clauses:
            clause_text = clause_dict["clause_text"]
            app_name = clause_dict.get("app_name", "Platform")
            result = self.predict(clause_text, app_name)
            results.append(result)
        return results
