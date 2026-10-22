"""
src/evaluation/generate_final_comparison_report.py
Generate final comparison report with corrected folds and pooled 8-seed results.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_results(tag):
    """Load experiment results."""
    path = Path(f"reports/experiments/confirmation/{tag}.json")
    if not path.exists():
        # Try training results
        path = Path(f"reports/experiments/{tag}.json")
    
    data = json.loads(path.read_text())
    
    # Normalize: training uses "aggregate_metrics", confirmation uses "aggregates"
    if "aggregate_metrics" in data and "aggregates" not in data:
        data["aggregates"] = data["aggregate_metrics"]
    
    return data


def pool_results(training_results, confirmation_results):
    """Pool results across all 8 seeds (42-44 training + 45-49 confirmation)."""
    
    all_folds = (training_results["per_fold_results"] + 
                confirmation_results["per_fold_results"])
    
    # Only pool metrics that exist in per_fold_results
    metrics = ["accuracy", "macro_f1"]
    
    # Check what's available in first fold
    if "macro_precision" in all_folds[0]:
        metrics.extend(["macro_precision", "macro_recall"])
    if "not_addressed_f1" in all_folds[0]:
        metrics.extend(["not_addressed_f1", "partial_f1",
                       "compliant_precision", "compliant_recall", "compliant_f1"])
    
    pooled = {}
    for metric in metrics:
        values = [r[metric] for r in all_folds]
        pooled[f"{metric}_mean"] = float(np.mean(values))
        pooled[f"{metric}_std"] = float(np.std(values))
    
    return pooled, all_folds


def main():
    print("=" * 120)
    print("FINAL COMPARISON REPORT (CORRECTED FOLDS)")
    print("=" * 120)
    
    # Load all results
    print("\n[INFO] Loading results...")
    
    # Training results (seeds 42-44)
    train_baseline = load_results("fixed_baseline_v1")
    train_A_e5 = load_results("fixed_A_e5")
    train_all_rules = load_results("incr_1_all_rules")
    
    # Confirmation results (seeds 45-49, CORRECTED)
    confirm_baseline = load_results("confirm_baseline_v1_corrected")
    confirm_A_e5 = load_results("confirm_fixed_A_e5_corrected")
    confirm_all_rules = load_results("confirm_all_rules_corrected")
    
    # Table 1: Training vs Confirmation
    print("\n" + "=" * 120)
    print("TABLE 1: TRAINING VS CONFIRMATION PERFORMANCE")
    print("=" * 120)
    print(f"{'Variant':<30} {'Folds':<10} {'Acc':<17} {'Macro-F1':<17} {'Macro-Prec':<17}")
    print("-" * 120)
    
    variants = [
        ("Baseline (MiniLM+8)", "42-44", train_baseline),
        ("", "45-49", confirm_baseline),
        ("E5+8", "42-44", train_A_e5),
        ("", "45-49", confirm_A_e5),
        ("E5+8+all-rules", "42-44", train_all_rules),
        ("", "45-49", confirm_all_rules),
    ]
    
    for name, folds, results in variants:
        agg = results["aggregates"]
        acc_str = f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}"
        f1_str = f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}"
        prec_str = f"{agg['macro_precision_mean']:.4f} ± {agg['macro_precision_std']:.4f}"
        
        print(f"{name:<30} {folds:<10} {acc_str:<17} {f1_str:<17} {prec_str:<17}")
    
    # Table 2: Per-Class Metrics
    print("\n" + "=" * 120)
    print("TABLE 2: PER-CLASS METRICS (CONFIRMATION FOLDS 45-49)")
    print("=" * 120)
    print(f"{'Variant':<30} {'NA F1':<17} {'Partial F1':<17} {'Compliant Prec':<17} {'Compliant Rec':<17} {'Compliant F1':<17}")
    print("-" * 120)
    
    confirm_variants = [
        ("Baseline (MiniLM+8)", confirm_baseline),
        ("E5+8", confirm_A_e5),
        ("E5+8+all-rules", confirm_all_rules),
    ]
    
    for name, results in confirm_variants:
        agg = results["aggregates"]
        print(f"{name:<30} "
              f"{agg['not_addressed_f1_mean']:.4f} ± {agg['not_addressed_f1_std']:.4f}   "
              f"{agg['partial_f1_mean']:.4f} ± {agg['partial_f1_std']:.4f}   "
              f"{agg['compliant_precision_mean']:.4f} ± {agg['compliant_precision_std']:.4f}   "
              f"{agg['compliant_recall_mean']:.4f} ± {agg['compliant_recall_std']:.4f}   "
              f"{agg['compliant_f1_mean']:.4f} ± {agg['compliant_f1_std']:.4f}")
    
    # Table 3: Pooled Results (All 8 Seeds = 40 Folds)
    print("\n" + "=" * 120)
    print("TABLE 3: POOLED RESULTS ACROSS ALL 8 SEEDS (42-49 = 40 FOLDS)")
    print("=" * 120)
    print(f"{'Variant':<30} {'N Folds':<10} {'Acc':<17} {'Macro-F1':<17}")
    print("-" * 120)
    
    # Pool baseline
    pooled_baseline, _ = pool_results(train_baseline, confirm_baseline)
    print(f"{'Baseline (MiniLM+8)':<30} {'40':<10} "
          f"{pooled_baseline['accuracy_mean']:.4f} ± {pooled_baseline['accuracy_std']:.4f}   "
          f"{pooled_baseline['macro_f1_mean']:.4f} ± {pooled_baseline['macro_f1_std']:.4f}")
    
    # Pool E5+8
    pooled_A_e5, _ = pool_results(train_A_e5, confirm_A_e5)
    print(f"{'E5+8':<30} {'40':<10} "
          f"{pooled_A_e5['accuracy_mean']:.4f} ± {pooled_A_e5['accuracy_std']:.4f}   "
          f"{pooled_A_e5['macro_f1_mean']:.4f} ± {pooled_A_e5['macro_f1_std']:.4f}")
    
    # Pool all-rules
    pooled_all_rules, _ = pool_results(train_all_rules, confirm_all_rules)
    print(f"{'E5+8+all-rules':<30} {'40':<10} "
          f"{pooled_all_rules['accuracy_mean']:.4f} ± {pooled_all_rules['accuracy_std']:.4f}   "
          f"{pooled_all_rules['macro_f1_mean']:.4f} ± {pooled_all_rules['macro_f1_std']:.4f}")
    
    # Table 4: Pooled Per-Class (only if available)
    if "compliant_f1_mean" in pooled_baseline:
        print("\n" + "=" * 120)
        print("TABLE 4: POOLED PER-CLASS METRICS (40 FOLDS)")
        print("=" * 120)
        print(f"{'Variant':<30} {'NA F1':<17} {'Partial F1':<17} {'Compliant Prec':<17} {'Compliant Rec':<17} {'Compliant F1':<17}")
        print("-" * 120)
        
        print(f"{'Baseline (MiniLM+8)':<30} "
              f"{pooled_baseline['not_addressed_f1_mean']:.4f} ± {pooled_baseline['not_addressed_f1_std']:.4f}   "
              f"{pooled_baseline['partial_f1_mean']:.4f} ± {pooled_baseline['partial_f1_std']:.4f}   "
              f"{pooled_baseline['compliant_precision_mean']:.4f} ± {pooled_baseline['compliant_precision_std']:.4f}   "
              f"{pooled_baseline['compliant_recall_mean']:.4f} ± {pooled_baseline['compliant_recall_std']:.4f}   "
              f"{pooled_baseline['compliant_f1_mean']:.4f} ± {pooled_baseline['compliant_f1_std']:.4f}")
        
        print(f"{'E5+8':<30} "
              f"{pooled_A_e5['not_addressed_f1_mean']:.4f} ± {pooled_A_e5['not_addressed_f1_std']:.4f}   "
              f"{pooled_A_e5['partial_f1_mean']:.4f} ± {pooled_A_e5['partial_f1_std']:.4f}   "
              f"{pooled_A_e5['compliant_precision_mean']:.4f} ± {pooled_A_e5['compliant_precision_std']:.4f}   "
              f"{pooled_A_e5['compliant_recall_mean']:.4f} ± {pooled_A_e5['compliant_recall_std']:.4f}   "
              f"{pooled_A_e5['compliant_f1_mean']:.4f} ± {pooled_A_e5['compliant_f1_std']:.4f}")
        
        print(f"{'E5+8+all-rules':<30} "
              f"{pooled_all_rules['not_addressed_f1_mean']:.4f} ± {pooled_all_rules['not_addressed_f1_std']:.4f}   "
              f"{pooled_all_rules['partial_f1_mean']:.4f} ± {pooled_all_rules['partial_f1_std']:.4f}   "
              f"{pooled_all_rules['compliant_precision_mean']:.4f} ± {pooled_all_rules['compliant_precision_std']:.4f}   "
              f"{pooled_all_rules['compliant_recall_mean']:.4f} ± {pooled_all_rules['compliant_recall_std']:.4f}   "
              f"{pooled_all_rules['compliant_f1_mean']:.4f} ± {pooled_all_rules['compliant_f1_std']:.4f}")
    
    # Analysis
    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)
    
    # Training vs Confirmation deltas
    delta_A_e5 = confirm_A_e5["aggregates"]["macro_f1_mean"] - train_A_e5["aggregates"]["macro_f1_mean"]
    delta_all_rules = confirm_all_rules["aggregates"]["macro_f1_mean"] - train_all_rules["aggregates"]["macro_f1_mean"]
    
    print(f"\n1. Generalization Performance:")
    print(f"   E5+8:          Training {train_A_e5['aggregates']['macro_f1_mean']:.4f} → Confirmation {confirm_A_e5['aggregates']['macro_f1_mean']:.4f} (Δ {delta_A_e5:+.4f})")
    print(f"   E5+8+all-rules: Training {train_all_rules['aggregates']['macro_f1_mean']:.4f} → Confirmation {confirm_all_rules['aggregates']['macro_f1_mean']:.4f} (Δ {delta_all_rules:+.4f})")
    
    # All-rules benefit
    train_benefit = train_all_rules["aggregates"]["macro_f1_mean"] - train_A_e5["aggregates"]["macro_f1_mean"]
    confirm_benefit = confirm_all_rules["aggregates"]["macro_f1_mean"] - confirm_A_e5["aggregates"]["macro_f1_mean"]
    pooled_benefit = pooled_all_rules["macro_f1_mean"] - pooled_A_e5["macro_f1_mean"]
    
    print(f"\n2. All-Rules Similarity Benefit:")
    print(f"   Training (42-44):      {train_benefit:+.4f} ({train_benefit/train_A_e5['aggregates']['macro_f1_mean']*100:+.1f}%)")
    print(f"   Confirmation (45-49):  {confirm_benefit:+.4f} ({confirm_benefit/confirm_A_e5['aggregates']['macro_f1_mean']*100:+.1f}%)")
    print(f"   Pooled (40 folds):     {pooled_benefit:+.4f} ({pooled_benefit/pooled_A_e5['macro_f1_mean']*100:+.1f}%)")
    
    if confirm_benefit < 0:
        print(f"\n   ⚠️ WARNING: All-rules shows NEGATIVE benefit on confirmation folds!")
        print(f"   This suggests overfitting to training fold structure.")
    elif confirm_benefit < train_benefit / 2:
        print(f"\n   ⚠️ CAUTION: All-rules benefit on confirmation is less than half of training benefit.")
        print(f"   Some overfitting may be present, though pooled performance is still positive.")
    else:
        print(f"\n   ✅ All-rules benefit generalizes well to unseen folds.")
    
    # Save summary
    summary_data = {
        "corrected_folds": True,
        "groups": "639 unique (60 multi-member), IDENTICAL for all seeds",
        "training_folds": "seeds 42-44 (15 folds)",
        "confirmation_folds": "seeds 45-49 (25 folds)",
        "pooled_folds": "seeds 42-49 (40 folds)",
        "baseline": {
            "training": train_baseline["aggregates"],
            "confirmation": confirm_baseline["aggregates"],
            "pooled": pooled_baseline
        },
        "E5_8": {
            "training": train_A_e5["aggregates"],
            "confirmation": confirm_A_e5["aggregates"],
            "pooled": pooled_A_e5
        },
        "E5_8_all_rules": {
            "training": train_all_rules["aggregates"],
            "confirmation": confirm_all_rules["aggregates"],
            "pooled": pooled_all_rules
        },
        "all_rules_benefit": {
            "training": float(train_benefit),
            "confirmation": float(confirm_benefit),
            "pooled": float(pooled_benefit)
        }
    }
    
    output_path = Path("reports/experiments/confirmation/final_comparison_corrected.json")
    output_path.write_text(json.dumps(summary_data, indent=2))
    print(f"\n[SUCCESS] Saved summary to {output_path}")


if __name__ == "__main__":
    main()
