"""
src/evaluation/export_confusion_matrix.py

Evaluates the baseline compliance classifier on the test split and
generates publication-ready outputs:

  models/baseline/confusion_matrix.png  — heatmap figure (300 dpi)
  models/baseline/confusion_matrix.csv  — raw tabular matrix
  models/baseline/calibrated_test_report.json — per-class metrics
"""

import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for headless/script execution
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Label constants (formerly in src/inference/predict.py)
LABEL_NAMES = ["Not Addressed", "Partially Compliant", "Compliant"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}

TEST_FILE  = Path("datasets/splits/test.csv")
MODEL_FILE = Path("models/baseline/baseline_model.joblib")
OUTPUT_DIR = Path("models/baseline")


def generate_evaluation_artifacts() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test split not found at {TEST_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load test split — original untouched split, filter to 3 classes  #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(TEST_FILE)
    df = df[df["verdict"].isin(LABEL_NAMES)].copy().reset_index(drop=True)
    print(f"Test samples loaded: {len(df)}")
    print(df["verdict"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    # 2. Load baseline model and run inference                            #
    # ------------------------------------------------------------------ #
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {MODEL_FILE}. "
            "Run: python src/training/train_baseline.py"
        )

    bundle = joblib.load(MODEL_FILE)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    print(f"\nRunning prediction on {len(df)} test samples...")

    X_test = vectorizer.transform(df["clause_text"])
    raw_preds = classifier.predict(X_test)

    y_true = df["verdict"].tolist()
    y_pred  = list(raw_preds)

    # ------------------------------------------------------------------ #
    # 3. Confusion Matrix — CSV                                           #
    # ------------------------------------------------------------------ #
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {l}"  for l in LABEL_NAMES],
        columns=[f"Pred: {l}" for l in LABEL_NAMES],
    )
    csv_path = OUTPUT_DIR / "confusion_matrix.csv"
    cm_df.to_csv(csv_path)
    print(f"\nConfusion matrix CSV saved to {csv_path}")
    print(cm_df.to_string())

    # ------------------------------------------------------------------ #
    # 4. Confusion Matrix — PNG heatmap (publication-ready, 300 dpi)     #
    # ------------------------------------------------------------------ #
    # Shorten labels for the axis tick display
    short_labels = ["Not\nAddressed", "Partially\nCompliant", "Compliant"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="lightgrey",
        ax=ax,
    )
    ax.set_title(
        "Baseline Compliance Classifier — Test Set Confusion Matrix\n"
        "(TF-IDF + Logistic Regression)",
        fontsize=11,
        pad=14,
    )
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("Ground Truth Label", fontsize=10, labelpad=8)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()

    img_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix PNG saved to {img_path}")

    # ------------------------------------------------------------------ #
    # 5. Calibrated classification report — JSON                          #
    # ------------------------------------------------------------------ #
    report = classification_report(
        y_true, y_pred,
        labels=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Test report saved to {report_path}")

    # ------------------------------------------------------------------ #
    # 6. Human-readable summary                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("FINAL TEST METRICS (Baseline Model)")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        labels=LABEL_NAMES,
        zero_division=0,
    ))


if __name__ == "__main__":
    generate_evaluation_artifacts()
