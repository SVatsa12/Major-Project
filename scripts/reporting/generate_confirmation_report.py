"""
src/evaluation/generate_confirmation_report.py
Generate comprehensive confirmation report with paired comparisons.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_confirmation_results(tag):
    """Load confirmation experiment results."""
    results_path = Path(f"reports/experiments/confirmation/{tag}.json")
    return json.loads(results_path.read_text())


def compute_paired_comparison(winner_results, baseline_results):
    """Compute per-fold paired comparison."""
    winner_folds = sorted(winner_results["per_fold_results"], key=lambda x: (x["seed"], x["fold"]))
    baseline_folds = sorted(baseline_results["per_fold_results"], key=lambda x: (x["seed"], x["fold"]))
    
    assert len(winner_folds) == len(baseline_folds), "Fold count mismatch"
    
    differences = []
    winner_wins = 0
    
    for w, b in zip(winner_folds, baseline_folds):
        assert w["seed"] == b["seed"] and w["fold"] == b["fold"], "Fold mismatch"
        
        diff = w["macro_f1"] - b["macro_f1"]
        differences.append({
            "seed": w["seed"],
            "fold": w["fold"],
            "winner_f1": w["macro_f1"],
            "baseline_f1": b["macro_f1"],
            "difference": diff
        })
        
        if diff > 0:
            winner_wins += 1
    
    mean_diff = np.mean([d["difference"] for d in differences])
    std_diff = np.std([d["difference"] for d in differences])
    
    return {
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "winner_wins": winner_wins,
        "total_folds": len(differences),
        "win_rate": winner_wins / len(differences),
        "per_fold_differences": differences
    }


def generate_report():
    """Generate confirmation report."""
    
    # Load all results
    print("[INFO] Loading confirmation results...")
    confirm_baseline = load_confirmation_results("confirm_baseline_v1")
    confirm_A_e5 = load_confirmation_results("confirm_fixed_A_e5")
    confirm_all_rules = load_confirmation_results("confirm_all_rules")
    confirm_desc_only = load_confirmation_results("confirm_allrules_desc_only")
    
    # Create summary table
    variants = [
        ("confirm_baseline_v1", "MiniLM + 8", confirm_baseline),
        ("confirm_fixed_A_e5", "E5 + 8", confirm_A_e5),
        ("confirm_all_rules", "E5 + 8 + all-rules", confirm_all_rules),
        ("confirm_allrules_desc_only", "E5 + 8 + all-rules (desc_only)", confirm_desc_only),
    ]
    
    print("\n" + "=" * 120)
    print("CONFIRMATION EVALUATION SUMMARY")
    print("=" * 120)
    print(f"{'Variant':<35} {'Folds':<7} {'Acc':<17} {'Macro-F1':<17} {'Macro-Prec':<17} {'Macro-Rec':<17}")
    print("-" * 120)
    
    summary_rows = []
    for tag, desc, results in variants:
        agg = results["aggregates"]
        row = {
            "variant": desc,
            "tag": tag,
            "folds": "45-49",
            "n_folds": results["config"]["total_folds"],
            "accuracy": f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}",
            "macro_f1": f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}",
            "macro_precision": f"{agg['macro_precision_mean']:.4f} ± {agg['macro_precision_std']:.4f}",
            "macro_recall": f"{agg['macro_recall_mean']:.4f} ± {agg['macro_recall_std']:.4f}",
            "na_f1": f"{agg['not_addressed_f1_mean']:.4f} ± {agg['not_addressed_f1_std']:.4f}",
            "partial_f1": f"{agg['partial_f1_mean']:.4f} ± {agg['partial_f1_std']:.4f}",
            "compliant_prec": f"{agg['compliant_precision_mean']:.4f} ± {agg['compliant_precision_std']:.4f}",
            "compliant_rec": f"{agg['compliant_recall_mean']:.4f} ± {agg['compliant_recall_std']:.4f}",
            "compliant_f1": f"{agg['compliant_f1_mean']:.4f} ± {agg['compliant_f1_std']:.4f}",
        }
        summary_rows.append(row)
        
        print(f"{desc:<35} {row['folds']:<7} {row['accuracy']:<17} {row['macro_f1']:<17} {row['macro_precision']:<17} {row['macro_recall']:<17}")
    
    print("\n" + "=" * 120)
    print("PER-CLASS METRICS")
    print("=" * 120)
    print(f"{'Variant':<35} {'NA F1':<17} {'Partial F1':<17} {'Compliant P':<17} {'Compliant R':<17} {'Compliant F1':<17}")
    print("-" * 120)
    
    for row in summary_rows:
        print(f"{row['variant']:<35} {row['na_f1']:<17} {row['partial_f1']:<17} {row['compliant_prec']:<17} {row['compliant_rec']:<17} {row['compliant_f1']:<17}")
    
    # Paired comparison: winner vs baseline
    print("\n" + "=" * 120)
    print("PAIRED COMPARISON: all_rules vs baseline")
    print("=" * 120)
    
    paired_comp = compute_paired_comparison(confirm_all_rules, confirm_baseline)
    
    print(f"Mean F1 difference:  {paired_comp['mean_difference']:+.4f} ± {paired_comp['std_difference']:.4f}")
    print(f"Winner wins:         {paired_comp['winner_wins']}/{paired_comp['total_folds']} folds ({paired_comp['win_rate']:.1%})")
    
    print("\nPer-fold breakdown:")
    print(f"{'Seed':<6} {'Fold':<6} {'Baseline F1':<12} {'Winner F1':<12} {'Difference':<12} {'Winner?':<8}")
    print("-" * 60)
    for diff in paired_comp["per_fold_differences"]:
        winner_mark = "✓" if diff["difference"] > 0 else "✗"
        print(f"{diff['seed']:<6} {diff['fold']:<6} {diff['baseline_f1']:<12.4f} {diff['winner_f1']:<12.4f} {diff['difference']:+12.4f} {winner_mark:<8}")
    
    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    output_csv = Path("reports/experiments/confirmation/summary.csv")
    df_summary.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Saved summary to {output_csv}")
    
    # Save paired comparison
    paired_output = Path("reports/experiments/confirmation/paired_comparison.json")
    paired_output.write_text(json.dumps(paired_comp, indent=2))
    print(f"[SUCCESS] Saved paired comparison to {paired_output}")
    
    return summary_rows, paired_comp


if __name__ == "__main__":
    generate_report()
