"""
src/evaluation/fold_metrics.py
Per-fold metrics shared by all experiment runners, so confirmation and training
folds report the same fields and can be pooled.
"""

from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

CLASS_KEYS = [
    ("Not Addressed", "not_addressed"),
    ("Partially Compliant", "partial"),
    ("Compliant", "compliant"),
]


def fold_metrics(y_true, y_pred):
    """Accuracy, macro P/R/F1 and per-class P/R/F1 for one test fold."""
    report = classification_report(
        y_true, y_pred,
        labels=[cls for cls, _ in CLASS_KEYS],
        output_dict=True,
        zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for cls, key in CLASS_KEYS:
        metrics[f"{key}_precision"] = float(report[cls]["precision"])
        metrics[f"{key}_recall"] = float(report[cls]["recall"])
        metrics[f"{key}_f1"] = float(report[cls]["f1-score"])
    return metrics
