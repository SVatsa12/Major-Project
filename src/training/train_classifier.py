"""
src/training/train_classifier.py
Training and Cross-Validation Pipeline with Leak-Free Threshold Calibration.

Implements the full training pipeline per research paper specifications:
- Context-conditioned embedding generation
- Feature extraction and scaling (RobustScaler, fit only on training fold)
- LogisticRegression with specified hyperparameters
- Leak-free inner CV threshold calibration
- Phase A: Multiple threshold tuning objectives
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


def build_context_texts(df, taxonomy_map):
    """
    Build context-formatted input texts for encoding.
    
    Format: f"[{app_name}] Mandate: {statutory_test} | Policy: {clause_text}"
    where statutory_test is from the taxonomy checkable_test field.
    
    Args:
        df: DataFrame with columns [app_name, clause_text, best_match_taxonomy_id]
        taxonomy_map: dict mapping taxonomy_id -> taxonomy_item
    
    Returns:
        list of context-formatted strings
    """
    contexts = []
    for _, row in df.iterrows():
        app_name = str(row.get("app_name", "Platform")).strip()
        clause_text = str(row["clause_text"]).strip()
        tax_id = str(row.get("best_match_taxonomy_id", "NONE"))
        
        if tax_id != "NONE" and tax_id in taxonomy_map:
            tax_item = taxonomy_map[tax_id]
            statutory_test = tax_item.get("checkable_test", "")
        else:
            statutory_test = "General Policy"
        
        context = f"[{app_name}] Mandate: {statutory_test} | Policy: {clause_text}"
        contexts.append(context)
    
    return contexts


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


def train_and_evaluate_fold(
    X_train, y_train, X_test, y_test,
    inner_cv_splits=3,
    test_all_objectives=False
):
    """
    Train classifier on fold and perform leak-free threshold calibration.
    
    The scaler is fit ONLY on the training partition of this fold.
    Inner CV for threshold tuning uses only the training partition.
    Thresholds are applied to the test fold.
    
    Phase A: Tests three threshold tuning objectives and selects best per inner fold.
    
    Args:
        X_train: dict with keys "embedding" (N_tr, 384) and "features" (N_tr, 8)
        y_train: class labels for training fold
        X_test: dict with keys "embedding" (N_te, 384) and "features" (N_te, 8)
        y_test: class labels for test fold
        inner_cv_splits: number of inner CV folds for threshold calibration
        test_all_objectives: if True, evaluate all three objectives and return results
    
    Returns:
        dict with predictions, thresholds, and metrics
        if test_all_objectives=True: also includes per-objective results
    """
    # Fit RobustScaler ONLY on training partition of this fold
    scaler = RobustScaler()
    features_train_scaled = scaler.fit_transform(X_train["features"])
    features_test_scaled = scaler.transform(X_test["features"])
    
    # Combine embedding and scaled features (768 + 8 = 776 dimensions)
    X_train_combined = np.hstack([X_train["embedding"], features_train_scaled])
    X_test_combined = np.hstack([X_test["embedding"], features_test_scaled])
    
    # Train classifier
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_combined, y_train)
    
    # Perform inner stratified CV for threshold calibration
    inner_skf = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=42)
    
    # Track results for all three objectives (Phase A)
    objectives = {
        "constraint_0.70": {"best_tau_c": 0.5, "best_tau_p": 0.35, "best_f1": 0.0},
        "constraint_0.60": {"best_tau_c": 0.5, "best_tau_p": 0.35, "best_f1": 0.0},
        "no_constraint": {"best_tau_c": 0.5, "best_tau_p": 0.35, "best_f1": 0.0},
    }
    
    for inner_train_idx, inner_val_idx in inner_skf.split(X_train_combined, y_train):
        X_inner_train = X_train_combined[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = X_train_combined[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]
        
        # Train classifier on inner training
        inner_clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=42
        )
        inner_clf.fit(X_inner_train, y_inner_train)
        
        # Get probabilities on inner validation fold
        inner_probs = inner_clf.predict_proba(X_inner_val)
        classes = list(inner_clf.classes_)
        
        if "Compliant" not in classes:
            continue
        
        c_idx = classes.index("Compliant")
        p_idx = classes.index("Partially Compliant")
        
        # Grid search over thresholds
        for tau_c in np.arange(0.30, 0.70, 0.05):
            for tau_p in np.arange(0.20, 0.45, 0.05):
                preds = []
                for i in range(len(inner_probs)):
                    p_compliant = inner_probs[i, c_idx]
                    p_partial = inner_probs[i, p_idx]
                    
                    if p_compliant >= tau_c:
                        preds.append("Compliant")
                    elif p_partial >= tau_p:
                        preds.append("Partially Compliant")
                    else:
                        preds.append("Not Addressed")
                
                # Compute precision for Compliant class
                try:
                    compliant_precision = precision_score(
                        y_inner_val, preds, labels=["Compliant"], average=None, zero_division=0
                    )[0]
                except:
                    compliant_precision = 0.0
                
                # Compute macro-F1
                inner_f1 = f1_score(y_inner_val, preds, average="macro", zero_division=0)
                
                # (a) Maximize macro-F1 subject to Compliant precision >= 0.70
                if compliant_precision >= 0.70:
                    if inner_f1 > objectives["constraint_0.70"]["best_f1"]:
                        objectives["constraint_0.70"]["best_f1"] = inner_f1
                        objectives["constraint_0.70"]["best_tau_c"] = tau_c
                        objectives["constraint_0.70"]["best_tau_p"] = tau_p
                
                # (b) Maximize macro-F1 subject to Compliant precision >= 0.60
                if compliant_precision >= 0.60:
                    if inner_f1 > objectives["constraint_0.60"]["best_f1"]:
                        objectives["constraint_0.60"]["best_f1"] = inner_f1
                        objectives["constraint_0.60"]["best_tau_c"] = tau_c
                        objectives["constraint_0.60"]["best_tau_p"] = tau_p
                
                # (c) Maximize macro-F1 with no constraint
                if inner_f1 > objectives["no_constraint"]["best_f1"]:
                    objectives["no_constraint"]["best_f1"] = inner_f1
                    objectives["no_constraint"]["best_tau_c"] = tau_c
                    objectives["no_constraint"]["best_tau_p"] = tau_p
    
    # Select best objective for this fold
    # Priority: select the variant with best inner F1
    best_variant = max(
        objectives.keys(),
        key=lambda k: objectives[k]["best_f1"]
    )
    best_tau_compliant = objectives[best_variant]["best_tau_c"]
    best_tau_partial = objectives[best_variant]["best_tau_p"]
    best_inner_f1 = objectives[best_variant]["best_f1"]
    
    # Apply best thresholds to test fold
    test_probs = clf.predict_proba(X_test_combined)
    classes = list(clf.classes_)
    test_preds = []
    
    if "Compliant" in classes:
        c_idx = classes.index("Compliant")
        p_idx = classes.index("Partially Compliant")
        
        for i in range(len(test_probs)):
            p_compliant = test_probs[i, c_idx]
            p_partial = test_probs[i, p_idx]
            
            if p_compliant >= best_tau_compliant:
                test_preds.append("Compliant")
            elif p_partial >= best_tau_partial:
                test_preds.append("Partially Compliant")
            else:
                test_preds.append("Not Addressed")
    else:
        test_preds = ["Not Addressed"] * len(test_probs)
    
    result = {
        "predictions": test_preds,
        "probs": test_probs,
        "classes": classes,
        "tau_compliant": best_tau_compliant,
        "tau_partial": best_tau_partial,
        "inner_f1": best_inner_f1,
        "scaler": scaler,
        "classifier": clf,
    }
    
    # If test_all_objectives, include all objective scores for this fold
    if test_all_objectives:
        result["inner_scores_all"] = objectives
        result["selected_objective"] = best_variant
    
    return result
