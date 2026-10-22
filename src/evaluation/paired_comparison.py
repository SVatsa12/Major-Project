"""
src/evaluation/paired_comparison.py
Paired statistical comparison of final model (E5+8+all-rules) vs baseline
on confirmation folds using Wilcoxon signed-rank test.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
import pandas as pd


def main():
    print("=" * 120)
    print("PAIRED COMPARISON: E5+8+all-rules vs Baseline on Confirmation Folds")
    print("=" * 120)
    
    # Load confirmation results
    print("\n[INFO] Loading confirmation results...")
    
    baseline_path = Path("reports/experiments/confirmation/confirm_baseline_v1_corrected.json")
    final_path = Path("reports/experiments/confirmation/confirm_all_rules_corrected.json")
    
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    with open(final_path, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    # Extract per-fold macro F1 scores
    baseline_scores = []
    final_scores = []
    
    for fold_result in baseline_data["per_fold_results"]:
        baseline_scores.append(fold_result["macro_f1"])
    
    for fold_result in final_data["per_fold_results"]:
        final_scores.append(fold_result["macro_f1"])
    
    baseline_scores = np.array(baseline_scores)
    final_scores = np.array(final_scores)
    
    print(f"[INFO] Baseline: {len(baseline_scores)} folds")
    print(f"[INFO] Final model: {len(final_scores)} folds")
    
    # Compute differences
    differences = final_scores - baseline_scores
    
    # Count wins
    final_wins = np.sum(differences > 0)
    baseline_wins = np.sum(differences < 0)
    ties = np.sum(differences == 0)
    
    # Compute mean difference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    # Wilcoxon signed-rank test
    # Note: wilcoxon automatically excludes zeros (ties)
    if np.sum(differences != 0) < 10:
        print("\n[WARNING] Too few non-zero differences for reliable Wilcoxon test")
        statistic, p_value = None, None
    else:
        statistic, p_value = wilcoxon(differences, alternative='greater')
    
    # Print results
    print("\n" + "=" * 120)
    print("PAIRED COMPARISON RESULTS")
    print("=" * 120)
    
    print(f"\nBaseline (MiniLM+8):")
    print(f"  Mean macro-F1: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
    print(f"  Range: [{np.min(baseline_scores):.4f}, {np.max(baseline_scores):.4f}]")
    
    print(f"\nFinal Model (E5+8+all-rules):")
    print(f"  Mean macro-F1: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
    print(f"  Range: [{np.min(final_scores):.4f}, {np.max(final_scores):.4f}]")
    
    print(f"\nPaired Differences (Final - Baseline):")
    print(f"  Mean difference: {mean_diff:.4f} ± {std_diff:.4f}")
    print(f"  Final model wins: {final_wins}/{len(differences)} folds ({100*final_wins/len(differences):.1f}%)")
    print(f"  Baseline wins: {baseline_wins}/{len(differences)} folds ({100*baseline_wins/len(differences):.1f}%)")
    print(f"  Ties: {ties}/{len(differences)} folds ({100*ties/len(differences):.1f}%)")
    
    if statistic is not None:
        print(f"\nWilcoxon Signed-Rank Test (one-sided, alternative='greater'):")
        print(f"  Statistic: {statistic:.2f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.001:
            print(f"  Result: *** Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"  Result: ** Very significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"  Result: * Significant (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
    
    # Save results to CSV
    output_dir = Path("reports/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create detailed fold-by-fold comparison table
    comparison_df = pd.DataFrame({
        "Fold": range(len(baseline_scores)),
        "Baseline_F1": baseline_scores,
        "Final_F1": final_scores,
        "Difference": differences,
        "Winner": ["Final" if d > 0 else "Baseline" if d < 0 else "Tie" for d in differences]
    })
    
    comparison_path = output_dir / "paired_comparison_folds.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[SUCCESS] Saved fold-by-fold comparison to {comparison_path}")
    
    # Create summary table
    summary_df = pd.DataFrame({
        "Model": ["Baseline (MiniLM+8)", "E5+8+all-rules", "Difference"],
        "Mean_F1": [np.mean(baseline_scores), np.mean(final_scores), mean_diff],
        "Std_F1": [np.std(baseline_scores), np.std(final_scores), std_diff],
        "Min_F1": [np.min(baseline_scores), np.min(final_scores), np.min(differences)],
        "Max_F1": [np.max(baseline_scores), np.max(final_scores), np.max(differences)],
    })
    
    summary_path = output_dir / "paired_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[SUCCESS] Saved comparison summary to {summary_path}")
    
    # Create statistical test results file
    stats_dict = {
        "test": "Wilcoxon signed-rank test",
        "alternative": "greater (final > baseline)",
        "n_folds": len(differences),
        "final_wins": int(final_wins),
        "baseline_wins": int(baseline_wins),
        "ties": int(ties),
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "statistic": float(statistic) if statistic is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant_at_0.05": bool(p_value < 0.05) if p_value is not None else None
    }
    
    stats_path = output_dir / "paired_comparison_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[SUCCESS] Saved statistical test results to {stats_path}")
    
    print("\n[SUCCESS] Paired comparison complete")


if __name__ == "__main__":
    main()
