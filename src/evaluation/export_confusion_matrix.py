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
EMBEDDING_MODEL_FILE = Path("models/baseline/embedding_classifier.joblib")
CONTEXT_MODEL_FILE = Path("models/baseline/context_classifier.joblib")
OUTPUT_DIR = Path("models/baseline")


def generate_evaluation_artifacts() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================ #
    # Check if we have 5-fold CV results (preferred over test split)   #
    # ================================================================ #
    eval_json_path = OUTPUT_DIR / "eval_test.json"
    cv_report_path = OUTPUT_DIR.parent / "reports" / "cross_validation_report.json"
    
    if eval_json_path.exists():
        with open(eval_json_path) as f:
            eval_data = json.load(f)
            if eval_data.get("evaluation_method") == "Stratified 5-Fold Cross-Validation (Bayes-Optimal Prior-Adjusted Calibration)":
                print("\n" + "=" * 70)
                print("DISPLAYING 5-FOLD CROSS-VALIDATION RESULTS (Full Corpus Evaluation)")
                print("=" * 70)
                print(f"\nEvaluation Method: {eval_data.get('evaluation_method')}")
                print(f"Total Clauses Evaluated: {eval_data.get('total_clauses_evaluated')}")
                print(f"Mean Macro-F1: {eval_data.get('mean_macro_f1'):.4f}")
                print(f"  (+/- {eval_data.get('std_macro_f1'):.4f})")
                print(f"Mean Accuracy: {eval_data.get('mean_accuracy')*100:.2f}%")
                print(f"\nIndividual Fold Macro-F1 Scores:")
                for i, f1 in enumerate(eval_data.get('fold_macro_f1s', []), 1):
                    print(f"  Fold {i}: {f1:.4f}")
                print(f"\nClass Distribution:")
                for cls, count in eval_data.get('class_distribution', {}).items():
                    print(f"  {cls}: {count}")
                print(f"\nArchitecture: {eval_data.get('architecture')}")
                print(f"Calibration: {eval_data.get('calibration')}")
                print(f"\n{eval_data.get('classification_report')}")
                print("=" * 70)
                return
    
    # ================================================================ #
    # Fallback to test split evaluation if 5-fold CV not available     #
    # ================================================================ #
    df = pd.read_csv(TEST_FILE)
    df = df[df["verdict"].isin(LABEL_NAMES)].copy().reset_index(drop=True)
    print(f"Test samples loaded: {len(df)}")
    print(df["verdict"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    # 2. Load latest trained model and run inference                      #
    # ------------------------------------------------------------------ #
    # Try to use embedding classifier (latest trained model)
    if EMBEDDING_MODEL_FILE.exists():
        print(f"\n[INFO] Loading embedding classifier from {EMBEDDING_MODEL_FILE}")
        from sentence_transformers import SentenceTransformer
        
        classifier = joblib.load(EMBEDDING_MODEL_FILE)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"Running prediction on {len(df)} test samples using SentenceTransformer embeddings...")
        X_test = embedder.encode(df["clause_text"].tolist(), show_progress_bar=False, normalize_embeddings=True)
        
        # Load calibrated thresholds from eval_test.json if available
        eval_json = OUTPUT_DIR / "eval_test.json"
        if eval_json.exists():
            with open(eval_json) as f:
                eval_data = json.load(f)
                if "calibrated_thresholds" in eval_data:
                    thresholds = eval_data["calibrated_thresholds"]
                    th_c = thresholds.get("Compliant", 0.35)
                    th_pc = thresholds.get("Partially Compliant", 0.40)
                    print(f"[INFO] Using calibrated thresholds: Compliant={th_c:.3f}, PartiallyCompliant={th_pc:.3f}")
                else:
                    th_c = 0.35
                    th_pc = 0.40
        else:
            th_c = 0.35
            th_pc = 0.40
        
        # Apply calibrated threshold logic
        test_probs = classifier.predict_proba(X_test)
        classes = list(classifier.classes_)
        c_idx = classes.index("Compliant")
        pc_idx = classes.index("Partially Compliant")
        
        raw_preds = []
        for p in test_probs:
            if p[c_idx] >= th_c and p[c_idx] >= p[pc_idx]:
                raw_preds.append("Compliant")
            elif p[pc_idx] >= th_pc:
                raw_preds.append("Partially Compliant")
            else:
                raw_preds.append("Not Addressed")
    else:
        raise FileNotFoundError(
            f"No trained model found. Expected {EMBEDDING_MODEL_FILE}\n"
            "Run: python src/training/train_embeddings_classifier.py"
        )

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
