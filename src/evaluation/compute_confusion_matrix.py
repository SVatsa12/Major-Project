"""
src/evaluation/compute_confusion_matrix.py
Compute confusion matrix for the final model (E5+8+all-rules) on confirmation folds
by re-running predictions on those folds.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# Import project modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.feature_extraction import extract_actionability_features
from src.training.train_classifier import build_context_texts, train_and_evaluate_fold
from src.evaluation.incremental_features import extract_all_rules_similarity


def main():
    print("=" * 120)
    print("COMPUTING CONFUSION MATRIX FOR FINAL MODEL (E5+8+all-rules)")
    print("=" * 120)
    
    # Load data
    print("\n[INFO] Loading data...")
    data_path = Path("corpus/labeled_clauses_bootstrap.csv")
    taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")
    
    df = pd.read_csv(data_path)
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    print(f"[INFO] Loaded {len(df)} clauses and {len(taxonomy)} taxonomy items")
    
    # Load confirmation folds
    folds_path = Path("datasets/folds/confirm_folds_v1.json")
    print(f"[INFO] Loading confirmation folds from {folds_path}")
    
    with open(folds_path, 'r', encoding='utf-8') as f:
        confirm_folds_data = json.load(f)
    
    # Extract folds from the nested structure
    confirm_folds = []
    for seed_str, folds_list in confirm_folds_data["seeds"].items():
        seed = int(seed_str)
        for fold_info in folds_list:
            confirm_folds.append({
                "seed": seed,
                "fold": fold_info["fold"],
                "train_idx": fold_info["train"],
                "test_idx": fold_info["test"]
            })
    
    print(f"[INFO] Loaded {len(confirm_folds)} confirmation folds")
    
    # Load embedder
    print("[INFO] Loading E5-small embedder...")
    embedder = SentenceTransformer("intfloat/e5-small-v2")
    
    # Build features
    tax_map = {item["taxonomy_id"]: item for item in taxonomy}
    context_texts = build_context_texts(df, tax_map)
    
    print("[INFO] Generating embeddings...")
    embeddings = embedder.encode(
        context_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print("[INFO] Extracting actionability features...")
    action_features = extract_actionability_features(df["clause_text"].values)
    
    print("[INFO] Computing all-rules similarity features...")
    all_rules_features = extract_all_rules_similarity(df, taxonomy, "intfloat/e5-small-v2", rule_text_mode="full")
    
    y_all = df["verdict"].values
    
    # Aggregate predictions across all confirmation folds
    all_y_true = []
    all_y_pred = []
    
    print(f"\n[INFO] Running predictions on {len(confirm_folds)} confirmation folds...")
    
    for fold_idx, fold_info in enumerate(confirm_folds, 1):
        seed = fold_info["seed"]
        train_idx = np.array(fold_info["train_idx"])
        test_idx = np.array(fold_info["test_idx"])
        
        # Split data
        X_train_embedding = embeddings[train_idx]
        X_test_embedding = embeddings[test_idx]
        X_train_action = action_features[train_idx]
        X_test_action = action_features[test_idx]
        X_train_rules = all_rules_features[train_idx]
        X_test_rules = all_rules_features[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        
        # Combine features: embedding + action + all_rules
        X_train = np.hstack([X_train_embedding, X_train_action, X_train_rules])
        X_test = np.hstack([X_test_embedding, X_test_action, X_test_rules])
        
        # Prepare data dict for training
        X_train_dict = {"embedding": X_train_embedding, "features": X_train_action, "all_rules": X_train_rules}
        X_test_dict = {"embedding": X_test_embedding, "features": X_test_action, "all_rules": X_test_rules}
        
        # Train and get predictions
        result = train_and_evaluate_fold(X_train_dict, y_train, X_test_dict, y_test, inner_cv_splits=3)
        test_preds = result["predictions"]
        
        all_y_true.extend(y_test)
        all_y_pred.extend(test_preds)
        
        print(f"  Fold {fold_idx}/25 (seed {seed}): {len(test_preds)} predictions")
    
    print(f"\n[INFO] Total predictions: {len(all_y_pred)}")
    
    # Class names
    classes = ["Not Addressed", "Partially Compliant", "Compliant"]
    
    # Compute confusion matrix (counts)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)
    print(f"[INFO] Confusion matrix shape: {cm.shape}")
    
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
    plt.close()
    
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
    
    print("\n[SUCCESS] Confusion matrix computation complete")


if __name__ == "__main__":
    main()
