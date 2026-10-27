"""
src/training/train_classifier.py
Training and Cross-Validation Pipeline with Leak-Free Threshold Calibration.

Implements the full training pipeline per research paper specifications:
- Label-free clause embeddings (see src/training/model_inputs.py)
- Feature extraction and scaling (RobustScaler, fit only on training fold)
- LogisticRegression with specified hyperparameters
- Leak-free inner CV threshold calibration
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score
from src.training.feature_extraction import extract_actionability_features


def get_taxonomy_embeddings(taxonomy_list, embedder):
    """
    Create embeddings for all 38 taxonomy items using their checkable_test fields.
    These are used for runtime pairing of unannotated clauses.
    
    Args:
        taxonomy_list: list of taxonomy items
        embedder: SentenceTransformer model
    
    Returns:
        dict mapping taxonomy_id -> embedding (normalized)
        dict mapping taxonomy_id -> checkable_test text
    """
    tax_embeddings = {}
    tax_texts = {}
    
    for item in taxonomy_list:
        tax_id = item["taxonomy_id"]
        test_text = item.get("checkable_test", "")
        if test_text:
            embedding = embedder.encode(test_text, convert_to_numpy=True, normalize_embeddings=True)
            tax_embeddings[tax_id] = embedding
            tax_texts[tax_id] = test_text
    
    return tax_embeddings, tax_texts


def pair_with_taxonomy_runtime(clause_embedding, tax_embeddings, similarity_threshold=0.35):
    """
    At runtime: cosine similarity against all 38 taxonomy test-prompt embeddings.
    Keep pairs with similarity >= threshold.
    
    Args:
        clause_embedding: normalized 384-dim embedding of clause (with context)
        tax_embeddings: dict mapping taxonomy_id -> normalized embedding
        similarity_threshold: float, minimum cosine similarity to retain pair
    
    Returns:
        taxonomy_id of best match (highest similarity)
        or "UNKNOWN" if no match above threshold
    """
    if not tax_embeddings:
        return "UNKNOWN"
    
    best_tax_id = "UNKNOWN"
    best_sim = similarity_threshold
    
    for tax_id, tax_emb in tax_embeddings.items():
        # Cosine similarity between normalized vectors = dot product
        similarity = np.dot(clause_embedding, tax_emb)
        if similarity > best_sim:
            best_sim = similarity
            best_tax_id = tax_id
    
    return best_tax_id


def make_fold_classifier():
    """LogisticRegression configuration used by train_and_evaluate_fold."""
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",
        random_state=42
    )


def apply_thresholds(probs, classes, tau_compliant, tau_partial):
    """
    Two-threshold decision rule. Probability columns are looked up by name in
    `classes` (the classifier's classes_), never by an assumed order.
    """
    classes = list(classes)
    compliant_idx = classes.index("Compliant") if "Compliant" in classes else None
    partial_idx = classes.index("Partially Compliant") if "Partially Compliant" in classes else None
    
    preds = []
    for i in range(len(probs)):
        p_compliant = probs[i, compliant_idx] if compliant_idx is not None else 0.0
        p_partial = probs[i, partial_idx] if partial_idx is not None else 0.0
        
        if p_compliant >= tau_compliant:
            preds.append("Compliant")
        elif p_partial >= tau_partial:
            preds.append("Partially Compliant")
        else:
            preds.append("Not Addressed")
    return preds


def tune_thresholds_inner_cv(X_train, y_train, make_classifier, inner_cv_splits=3):
    """
    Leak-free threshold calibration: stratified inner CV on the training data only.
    A fresh classifier is refit on each inner training split and thresholds are
    scored on the held-out inner split. Grid: tau_c in [0.30, 0.70), tau_p in
    [0.20, 0.45), step 0.05. Objective: inner macro-F1 subject to Compliant
    precision >= 0.70. Falls back to (0.5, 0.35) if no setting meets the constraint.
    
    Returns:
        (tau_compliant, tau_partial, best_inner_f1)
    """
    y_train = np.asarray(y_train)
    inner_skf = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=42)
    
    best_tau_compliant = 0.5
    best_tau_partial = 0.35
    best_inner_f1 = 0.0
    
    for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = X_train[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]
        
        # Skip if training data doesn't have all classes
        if len(np.unique(y_inner_train)) < 2:
            continue
        
        # Train classifier on inner training
        inner_clf = make_classifier()
        try:
            inner_clf.fit(X_inner_train, y_inner_train)
        except ValueError:
            # Skip if fit fails (e.g., only one class)
            continue
        
        # Get probabilities on inner validation fold
        inner_probs = inner_clf.predict_proba(X_inner_val)
        classes = list(inner_clf.classes_)
        
        # Skip this inner fold if not all classes present
        if len(classes) < 3:
            continue
        
        # Grid search over thresholds
        for tau_c in np.arange(0.30, 0.70, 0.05):
            for tau_p in np.arange(0.20, 0.45, 0.05):
                preds = apply_thresholds(inner_probs, classes, tau_c, tau_p)
                
                # Check constraint: minority class precision >= 0.70
                compliant_precision = precision_score(
                    y_inner_val, preds, labels=["Compliant"], average=None, zero_division=0
                )[0]
                
                # Maximize inner macro-F1 subject to precision constraint
                if compliant_precision >= 0.70:
                    inner_f1 = f1_score(y_inner_val, preds, average="macro", zero_division=0)
                    if inner_f1 > best_inner_f1:
                        best_inner_f1 = inner_f1
                        best_tau_compliant = tau_c
                        best_tau_partial = tau_p
    
    return float(best_tau_compliant), float(best_tau_partial), float(best_inner_f1)


def train_and_evaluate_fold(
    X_train, y_train, X_test, y_test,
    inner_cv_splits=3
):
    """
    Train classifier on fold and perform leak-free threshold calibration.
    
    The scaler is fit ONLY on the training partition of this fold.
    Inner CV for threshold tuning uses only the training partition.
    Thresholds are applied to the test fold.
    
    Args:
        X_train: dict with keys "embedding" (N_tr, 384) and "features" (N_tr, 8)
        y_train: class labels for training fold
        X_test: dict with keys "embedding" (N_te, 384) and "features" (N_te, 8)
        y_test: class labels for test fold
        inner_cv_splits: number of inner CV folds for threshold calibration
    
    Returns:
        dict with keys: predictions, probs, thresholds_chosen, inner_f1
    """
    # Fit RobustScaler ONLY on training partition of this fold
    scaler = RobustScaler()
    features_train_scaled = scaler.fit_transform(X_train["features"])
    features_test_scaled = scaler.transform(X_test["features"])
    
    # Combine embedding and scaled features (392 dimensions)
    X_train_combined = np.hstack([X_train["embedding"], features_train_scaled])
    X_test_combined = np.hstack([X_test["embedding"], features_test_scaled])
    
    # Train classifier
    clf = make_fold_classifier()
    clf.fit(X_train_combined, y_train)
    
    # Inner stratified CV on the training partition only (refit per inner fold)
    best_tau_compliant, best_tau_partial, best_inner_f1 = tune_thresholds_inner_cv(
        X_train_combined, y_train, make_fold_classifier, inner_cv_splits=inner_cv_splits
    )
    
    # Apply best thresholds to test fold
    test_probs = clf.predict_proba(X_test_combined)
    classes = list(clf.classes_)
    test_preds = apply_thresholds(test_probs, classes, best_tau_compliant, best_tau_partial)
    
    return {
        "predictions": test_preds,
        "probs": test_probs,
        "classes": classes,
        "tau_compliant": best_tau_compliant,
        "tau_partial": best_tau_partial,
        "inner_f1": best_inner_f1,
        "scaler": scaler,
        "classifier": clf,
    }
