"""
src/training/ordinal.py
Ordinal classifier for Not Addressed < Partially Compliant < Compliant.

OrdinalLR: two cumulative binary LogRegs on RobustScaled features,
    P(y > Not Addressed) and P(y > Partially Compliant), with
    P(Partial) = P(> NA) - P(> Partial) and P(> Partial) capped at P(> NA).
OrdinalLRCV: OrdinalLR with C chosen by inner stratified CV on the fit rows
    (mean inner macro-F1 of argmax predictions; ties go to the smaller C).
    This is the r1_ordinal config evaluated in run_round2_variants.py.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

CLASSES = ("Compliant", "Not Addressed", "Partially Compliant")  # sorted, as sklearn orders classes_
C_GRID = (0.1, 0.3, 1.0, 3.0)
INNER_CV_SPLITS = 3


def make_binary_lr(C):
    return LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=42
    )


class OrdinalLR:
    """Two cumulative binary LogRegs on RobustScaled features."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        y = np.asarray(y)
        self.scaler_ = RobustScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.gt_na_ = make_binary_lr(self.C).fit(Z, y != "Not Addressed")
        self.gt_partial_ = make_binary_lr(self.C).fit(Z, y == "Compliant")
        self.classes_ = np.array(CLASSES)
        return self

    def predict_proba(self, X):
        Z = self.scaler_.transform(X)
        p_gt_na = self.gt_na_.predict_proba(Z)[:, list(self.gt_na_.classes_).index(True)]
        p_gt_partial = np.minimum(
            self.gt_partial_.predict_proba(Z)[:, list(self.gt_partial_.classes_).index(True)], p_gt_na)
        by_class = {"Not Addressed": 1 - p_gt_na, "Partially Compliant": p_gt_na - p_gt_partial,
                    "Compliant": p_gt_partial}
        return np.column_stack([by_class[c] for c in self.classes_])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class OrdinalLRCV:
    """OrdinalLR with C chosen by inner stratified CV on the rows it is fit on."""

    def __init__(self, Cs=C_GRID, inner_cv_splits=INNER_CV_SPLITS):
        self.Cs = tuple(Cs)
        self.inner_cv_splits = inner_cv_splits

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.cv_scores_ = {}
        if len(self.Cs) > 1:
            inner = StratifiedKFold(n_splits=self.inner_cv_splits, shuffle=True, random_state=42)
            for C in self.Cs:
                f1s = [f1_score(y[va], OrdinalLR(C).fit(X[tr], y[tr]).predict(X[va]), average="macro", zero_division=0)
                       for tr, va in inner.split(np.zeros(len(y)), y)]
                self.cv_scores_[C] = float(np.mean(f1s))
            best = max(self.cv_scores_.values())
            self.C_ = min(C for C, s in self.cv_scores_.items() if s == best)
        else:
            self.C_ = self.Cs[0]
        self.model_ = OrdinalLR(self.C_).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)
