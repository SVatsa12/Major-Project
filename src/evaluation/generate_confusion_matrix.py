"""
src/evaluation/generate_confusion_matrix.py
Generate confusion matrix for the final model (E5+8+all-rules) on confirmation folds.
Outputs both PNG visualization and CSV data (counts + row-normalized).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


def main():
    print("=" * 120)
    print("GENERATING CONFUSION MATRIX FOR FINAL MODEL")
    print("=" * 120)
    
    # Load confirmation results for E5+8+all-rules
    result_path = Path("reports/experiments/confirmation/confirm_all_rules_labelfree.json")
    print(f"\n[INFO] Loading results from {result_path}")
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if we have predictions stored
    if "per_fold_results" not in data:
        print("[ERROR] No per_fold_results found")
        return
    
    # We need to get y_true and y_pred for all folds
    # These should be stored in the results JSON
    # If not, we need to re-run the experiment with predictions saved
    
    # Check if predictions are available
    first_fold = data["per_fold_results"][0]
    if "y_true" not in first_fold or "y_pred" not in first_fold:
        print("[ERROR] Predictions not stored in results JSON")
        print("[INFO] The confusion matrix requires y_true and y_pred from each fold")
        print("[INFO] These are not currently saved. We need to:")
        print("  1. Modify the CV evaluation to save predictions")
        print("  2. Re-run the confirmation experiment")
        print("\n[WORKAROUND] For now, we'll need to re-compute the final model predictions")
        print("  or modify evaluate_5fold_nested_cv.py to save predictions")
        return
    
    # Aggregate predictions from all confirmation folds
    print(f"[INFO] Aggregating predictions from {len(data['per_fold_results'])} folds")
    
    all_y_true = []
    all_y_pred = []
    
    for fold_result in data["per_fold_results"]:
        all_y_true.extend(fold_result["y_true"])
        all_y_pred.extend(fold_result["y_pred"])
    
    # Class names
    classes = ["Not Addressed", "Partially Compliant", "Compliant"]
    
    # Compute confusion matrix (counts)
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n[INFO] Confusion matrix shape: {cm.shape}")
    print(f"[INFO] Total predictions: {cm.sum()}")
    
    # Save counts as CSV
    output_dir = Path("reports/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_counts_path = output_dir / "confusion_matrix_counts.csv"
    cm_df.to_csv(cm_counts_path)
    print(f"[SUCCESS] Saved confusion matrix counts to {cm_counts_path}")
    
    # Compute row-normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=classes, columns=classes)
    cm_norm_path = output_dir / "confusion_matrix_normalized.csv"
    cm_norm_df.to_csv(cm_norm_path)
    print(f"[SUCCESS] Saved normalized confusion matrix to {cm_norm_path}")
    
    # Generate PNG visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot counts
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)\nE5+8+all-rules on Confirmation Folds')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot normalized
    sns.heatmap(cm_norm_df, annot=True, fmt='.3f', cmap='Blues', ax=ax2, 
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Row-Normalized)\nE5+8+all-rules on Confirmation Folds')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    png_path = output_dir / "confusion_matrix.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved confusion matrix visualization to {png_path}")
    
    # Print summary statistics
    print("\n" + "=" * 120)
    print("CONFUSION MATRIX SUMMARY")
    print("=" * 120)
    print("\nCounts:")
    print(cm_df.to_string())
    print("\nRow-Normalized (Recall per class):")
    print(cm_norm_df.to_string())
    print("\nDiagonal (correct predictions):")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {cm[i, i]} / {cm[i].sum()} = {cm_normalized[i, i]:.3f}")
    
    print("\n[SUCCESS] Confusion matrix generation complete")


if __name__ == "__main__":
    main()
