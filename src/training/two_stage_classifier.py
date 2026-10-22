"""
src/training/two_stage_classifier.py
Two-stage classification: Stage 1 (Not Addressed vs Addressed), Stage 2 (Partial vs Compliant)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score


def train_two_stage_fold(X_train, y_train, X_test, y_test, inner_cv_splits=3):
    """
    Two-stage classifier with inner CV threshold tuning for each stage.
    
    Stage 1: Not Addressed vs Addressed (Partially Compliant + Compliant)
    Stage 2: Partially Compliant vs Compliant (trained only on Addressed rows)
    
    Returns:
        dict with predictions, thresholds, and metrics
    """
    # Fit scaler on training partition
    scaler = RobustScaler()
    features_train_scaled = scaler.fit_transform(X_train["features"])
    features_test_scaled = scaler.transform(X_test["features"])
    
    # Combine embedding and scaled features
    X_train_combined = np.hstack([X_train["embedding"], features_train_scaled])
    X_test_combined = np.hstack([X_test["embedding"], features_test_scaled])
    
    # ========================================================================
    # STAGE 1: Not Addressed vs Addressed
    # ========================================================================
    y_train_stage1 = np.array(["Addressed" if y != "Not Addressed" else "Not Addressed" for y in y_train])
    
    clf_stage1 = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    clf_stage1.fit(X_train_combined, y_train_stage1)
    
    # Inner CV for stage 1 threshold tuning
    inner_skf = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=42)
    
    best_tau_stage1 = 0.5
    best_inner_f1_stage1 = 0.0
    
    for inner_train_idx, inner_val_idx in inner_skf.split(X_train_combined, y_train_stage1):
        X_inner_train = X_train_combined[inner_train_idx]
        y_inner_train = y_train_stage1[inner_train_idx]
        X_inner_val = X_train_combined[inner_val_idx]
        y_inner_val = y_train_stage1[inner_val_idx]
        
        # Skip if not enough classes
        if len(np.unique(y_inner_train)) < 2:
            continue
        
        inner_clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        try:
            inner_clf.fit(X_inner_train, y_inner_train)
        except ValueError:
            continue
        inner_probs = inner_clf.predict_proba(X_inner_val)
        
        classes = list(inner_clf.classes_)
        addressed_idx = classes.index("Addressed")
        
        # Grid search over thresholds
        for tau in np.arange(0.30, 0.70, 0.05):
            preds = ["Addressed" if inner_probs[i, addressed_idx] >= tau else "Not Addressed" 
                     for i in range(len(inner_probs))]
            
            inner_f1 = f1_score(y_inner_val, preds, average="binary", pos_label="Addressed", zero_division=0)
            if inner_f1 > best_inner_f1_stage1:
                best_inner_f1_stage1 = inner_f1
                best_tau_stage1 = tau
    
    # Apply stage 1 to test fold
    test_probs_stage1 = clf_stage1.predict_proba(X_test_combined)
    classes_stage1 = list(clf_stage1.classes_)
    addressed_idx = classes_stage1.index("Addressed")
    
    stage1_preds = ["Addressed" if test_probs_stage1[i, addressed_idx] >= best_tau_stage1 else "Not Addressed"
                    for i in range(len(test_probs_stage1))]
    
    # ========================================================================
    # STAGE 2: Partially Compliant vs Compliant (on Addressed rows only)
    # ========================================================================
    # Filter training data to only Addressed samples
    addressed_train_mask = y_train != "Not Addressed"
    X_train_addressed = X_train_combined[addressed_train_mask]
    y_train_addressed = y_train[addressed_train_mask]
    
    clf_stage2 = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    clf_stage2.fit(X_train_addressed, y_train_addressed)
    
    # Inner CV for stage 2 threshold tuning
    best_tau_compliant = 0.5
    best_tau_partial = 0.35
    best_inner_f1_stage2 = 0.0
    
    for inner_train_idx, inner_val_idx in inner_skf.split(X_train_addressed, y_train_addressed):
        X_inner_train = X_train_addressed[inner_train_idx]
        y_inner_train = y_train_addressed[inner_train_idx]
        X_inner_val = X_train_addressed[inner_val_idx]
        y_inner_val = y_train_addressed[inner_val_idx]
        
        # Skip if not enough classes
        if len(np.unique(y_inner_train)) < 2:
            continue
        
        inner_clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
        try:
            inner_clf.fit(X_inner_train, y_inner_train)
        except ValueError:
            continue
        inner_probs = inner_clf.predict_proba(X_inner_val)
        
        classes = list(inner_clf.classes_)
        
        # Grid search over thresholds
        for tau_c in np.arange(0.30, 0.70, 0.05):
            for tau_p in np.arange(0.20, 0.45, 0.05):
                preds = []
                for i in range(len(inner_probs)):
                    p_compliant = inner_probs[i, classes.index("Compliant")]
                    p_partial = inner_probs[i, classes.index("Partially Compliant")]
                    
                    if p_compliant >= tau_c:
                        preds.append("Compliant")
                    elif p_partial >= tau_p:
                        preds.append("Partially Compliant")
                    else:
                        preds.append("Not Addressed")  # Fallback (shouldn't happen in stage 2)
                
                # Check constraint: minority class precision >= 0.70
                compliant_precision = precision_score(
                    y_inner_val, preds, labels=["Compliant"], average=None, zero_division=0
                )[0] if "Compliant" in preds else 0.0
                
                if compliant_precision >= 0.70:
                    inner_f1 = f1_score(y_inner_val, preds, average="macro", zero_division=0)
                    if inner_f1 > best_inner_f1_stage2:
                        best_inner_f1_stage2 = inner_f1
                        best_tau_compliant = tau_c
                        best_tau_partial = tau_p
    
    # Apply stage 2 to Addressed test samples
    test_probs_stage2 = clf_stage2.predict_proba(X_test_combined)
    classes_stage2 = list(clf_stage2.classes_)
    
    # Combine stage 1 and stage 2 predictions
    final_preds = []
    for i in range(len(stage1_preds)):
        if stage1_preds[i] == "Not Addressed":
            final_preds.append("Not Addressed")
        else:
            # Apply stage 2
            p_compliant = test_probs_stage2[i, classes_stage2.index("Compliant")]
            p_partial = test_probs_stage2[i, classes_stage2.index("Partially Compliant")]
            
            if p_compliant >= best_tau_compliant:
                final_preds.append("Compliant")
            elif p_partial >= best_tau_partial:
                final_preds.append("Partially Compliant")
            else:
                final_preds.append("Not Addressed")  # Fallback
    
    return {
        "predictions": final_preds,
        "tau_stage1": best_tau_stage1,
        "tau_compliant": best_tau_compliant,
        "tau_partial": best_tau_partial,
        "inner_f1_stage1": best_inner_f1_stage1,
        "inner_f1_stage2": best_inner_f1_stage2,
        "scaler": scaler,
        "clf_stage1": clf_stage1,
        "clf_stage2": clf_stage2,
    }
