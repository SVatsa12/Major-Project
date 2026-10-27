"""
src/evaluation/generate_paper_results.py
Generate paper-ready results tables (CSV + LaTeX) for confirmation and pooled results.

Reads the *_labelfree runs (input = "query: " + clause_text). Earlier runs built
their inputs with build_context_texts, which leaked the label, and are INVALID.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_results(tag):
    """Load experiment results."""
    path = Path(f"reports/experiments/confirmation/{tag}.json")
    if not path.exists():
        path = Path(f"reports/experiments/{tag}.json")
    
    data = json.loads(path.read_text())
    
    # Normalize: training uses "aggregate_metrics", confirmation uses "aggregates"
    if "aggregate_metrics" in data and "aggregates" not in data:
        data["aggregates"] = data["aggregate_metrics"]
    
    # Normalize per-class naming
    if "per_class_metrics" in data:
        per_class = data["per_class_metrics"]
        # Rename to match confirmation format
        normalized = {}
        for cls_name, metrics in per_class.items():
            if cls_name == "Not Addressed":
                normalized["not_addressed"] = metrics
            elif cls_name == "Partially Compliant":
                normalized["partial"] = metrics
            elif cls_name == "Compliant":
                normalized["compliant"] = metrics
        data["per_class_normalized"] = normalized
    
    return data


def extract_metrics(results, metric_type="aggregates"):
    """Extract metrics from results."""
    if metric_type == "aggregates":
        agg = results["aggregates"]
        return {
            "accuracy_mean": agg.get("accuracy_mean", 0),
            "accuracy_std": agg.get("accuracy_std", 0),
            "macro_precision_mean": agg.get("macro_precision_mean", 0),
            "macro_precision_std": agg.get("macro_precision_std", 0),
            "macro_recall_mean": agg.get("macro_recall_mean", 0),
            "macro_recall_std": agg.get("macro_recall_std", 0),
            "macro_f1_mean": agg.get("macro_f1_mean", 0),
            "macro_f1_std": agg.get("macro_f1_std", 0),
            "na_precision_mean": agg.get("not_addressed_precision_mean", 0),
            "na_precision_std": agg.get("not_addressed_precision_std", 0),
            "na_recall_mean": agg.get("not_addressed_recall_mean", 0),
            "na_recall_std": agg.get("not_addressed_recall_std", 0),
            "na_f1_mean": agg.get("not_addressed_f1_mean", 0),
            "na_f1_std": agg.get("not_addressed_f1_std", 0),
            "partial_precision_mean": agg.get("partial_precision_mean", 0),
            "partial_precision_std": agg.get("partial_precision_std", 0),
            "partial_recall_mean": agg.get("partial_recall_mean", 0),
            "partial_recall_std": agg.get("partial_recall_std", 0),
            "partial_f1_mean": agg.get("partial_f1_mean", 0),
            "partial_f1_std": agg.get("partial_f1_std", 0),
            "compliant_precision_mean": agg.get("compliant_precision_mean", 0),
            "compliant_precision_std": agg.get("compliant_precision_std", 0),
            "compliant_recall_mean": agg.get("compliant_recall_mean", 0),
            "compliant_recall_std": agg.get("compliant_recall_std", 0),
            "compliant_f1_mean": agg.get("compliant_f1_mean", 0),
            "compliant_f1_std": agg.get("compliant_f1_std", 0),
        }
    elif metric_type == "per_class":
        pc = results.get("per_class_normalized", {})
        return {
            "na_precision_mean": pc.get("not_addressed", {}).get("precision", 0),
            "na_precision_std": pc.get("not_addressed", {}).get("precision_std", 0),
            "na_recall_mean": pc.get("not_addressed", {}).get("recall", 0),
            "na_recall_std": pc.get("not_addressed", {}).get("recall_std", 0),
            "na_f1_mean": pc.get("not_addressed", {}).get("f1", 0),
            "na_f1_std": pc.get("not_addressed", {}).get("f1_std", 0),
            "partial_precision_mean": pc.get("partial", {}).get("precision", 0),
            "partial_precision_std": pc.get("partial", {}).get("precision_std", 0),
            "partial_recall_mean": pc.get("partial", {}).get("recall", 0),
            "partial_recall_std": pc.get("partial", {}).get("recall_std", 0),
            "partial_f1_mean": pc.get("partial", {}).get("f1", 0),
            "partial_f1_std": pc.get("partial", {}).get("f1_std", 0),
            "compliant_precision_mean": pc.get("compliant", {}).get("precision", 0),
            "compliant_precision_std": pc.get("compliant", {}).get("precision_std", 0),
            "compliant_recall_mean": pc.get("compliant", {}).get("recall", 0),
            "compliant_recall_std": pc.get("compliant", {}).get("recall_std", 0),
            "compliant_f1_mean": pc.get("compliant", {}).get("f1", 0),
            "compliant_f1_std": pc.get("compliant", {}).get("f1_std", 0),
        }


def pool_results(training_results, confirmation_results):
    """Pool results across all folds."""
    all_folds = (training_results["per_fold_results"] + 
                confirmation_results["per_fold_results"])
    
    # Available metrics - check first fold
    available_metrics = list(all_folds[0].keys())
    
    pooled = {"n_folds": len(all_folds)}
    for metric in available_metrics:
        if metric in ['fold', 'seed'] or not isinstance(all_folds[0][metric], (int, float)):
            continue
        values = [r[metric] for r in all_folds if metric in r]
        if values:
            pooled[f"{metric}_mean"] = float(np.mean(values))
            pooled[f"{metric}_std"] = float(np.std(values))
    
    return pooled


def metric_row(name, agg):
    """One table row: accuracy, macro P/R/F1 and per-class P/R/F1 as mean ± std over folds."""
    def fmt(key):
        return f"{agg[key + '_mean']:.4f} ± {agg[key + '_std']:.4f}"
    row = {
        "Model": name,
        "Accuracy": fmt("accuracy"),
        "Macro-Prec": fmt("macro_precision"),
        "Macro-Rec": fmt("macro_recall"),
        "Macro-F1": fmt("macro_f1"),
    }
    for label, key in [("NA", "not_addressed"), ("Partial", "partial"), ("Compliant", "compliant")]:
        row[f"{label}-Prec"] = fmt(f"{key}_precision")
        row[f"{label}-Rec"] = fmt(f"{key}_recall")
        row[f"{label}-F1"] = fmt(f"{key}_f1")
    return row


def main():
    print("=" * 120)
    print("GENERATING PAPER RESULTS TABLES")
    print("=" * 120)
    
    # Load results
    print("\n[INFO] Loading results...")
    
    # Confirmation (seeds 45-49) - PRIMARY
    confirm_baseline = load_results("confirm_baseline_v1_labelfree")
    confirm_e5 = load_results("confirm_fixed_A_e5_labelfree")
    confirm_all_rules = load_results("confirm_all_rules_labelfree")
    
    # Training (seeds 42-44)
    train_baseline = load_results("fixed_baseline_v1_labelfree")
    train_e5 = load_results("fixed_A_e5_labelfree")
    train_all_rules = load_results("incr_1_all_rules_labelfree")
    
    # Table 1: Confirmation Results (PRIMARY, seeds 45-49, 25 folds)
    print("\n[INFO] Generating Table 1: Confirmation Results...")
    
    confirm_rows = []
    for name, results in [("Baseline (MiniLM+8)", confirm_baseline),
                          ("E5+8", confirm_e5),
                          ("E5+8+all-rules", confirm_all_rules)]:
        row = metric_row(name, results["aggregates"])
        confirm_rows.append(row)
    
    df_confirm = pd.DataFrame(confirm_rows)
    
    # Save CSV
    csv_path = Path("reports/paper/confirmation_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_confirm.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved confirmation results to {csv_path}")
    
    # Generate LaTeX table
    latex_confirm = df_confirm.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df_confirm.columns) - 1))
    latex_path = Path("reports/paper/confirmation_results.tex")
    latex_path.write_text(latex_confirm)
    print(f"[SUCCESS] Saved LaTeX table to {latex_path}")
    
    # Table 2: Pooled Results (SECONDARY, seeds 42-49, 40 folds)
    print("\n[INFO] Generating Table 2: Pooled Results...")
    
    pooled_baseline = pool_results(train_baseline, confirm_baseline)
    pooled_e5 = pool_results(train_e5, confirm_e5)
    pooled_all_rules = pool_results(train_all_rules, confirm_all_rules)
    
    pooled_rows = []
    for name, pooled in [("Baseline (MiniLM+8)", pooled_baseline),
                         ("E5+8", pooled_e5),
                         ("E5+8+all-rules", pooled_all_rules)]:
        assert pooled["n_folds"] == 40, f"{name}: expected 40 pooled folds, got {pooled['n_folds']}"
        row = metric_row(name, pooled)
        pooled_rows.append(row)
    
    df_pooled = pd.DataFrame(pooled_rows)
    
    # Save CSV
    csv_pooled_path = Path("reports/paper/pooled_results.csv")
    df_pooled.to_csv(csv_pooled_path, index=False)
    print(f"[SUCCESS] Saved pooled results to {csv_pooled_path}")
    
    # Generate LaTeX table
    latex_pooled = df_pooled.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df_pooled.columns) - 1))
    latex_pooled_path = Path("reports/paper/pooled_results.tex")
    latex_pooled_path.write_text(latex_pooled)
    print(f"[SUCCESS] Saved LaTeX table to {latex_pooled_path}")
    
    print("\n[SUCCESS] Paper results tables generated")
    print(f"  - Confirmation (PRIMARY): {csv_path}")
    print(f"  - Pooled (SECONDARY): {csv_pooled_path}")


if __name__ == "__main__":
    main()
