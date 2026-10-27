"""
src/evaluation/run_honest_variants.py
Label-free variants of the final model on the confirmation folds (seeds 45-49).

One change at a time against the current final model (E5 + 8 + all-rules,
LogisticRegression C=1, inner-CV thresholds):
  ref_argmax       current final features, argmax decision (reproduces 0.6106)
  v1_final         current final features, inner-CV thresholds (reference)
  v2_tfidf         TF-IDF (word 1-2 grams, sublinear tf, min_df=2) + LogReg, alone
  v3_tfidf_stacked TF-IDF stacked with E5 + 8 + all-rules
  v4_C_select      current final features, C in {0.1, 0.3, 1, 3} chosen by inner CV
  v5_two_step      best of v1-v4: (a) Not Addressed vs Addressed, (b) Partial vs
                   Compliant trained/evaluated on truly-Addressed rows

Every model is a Pipeline (TF-IDF / RobustScaler -> LogisticRegression), so each
fit -- outer training fold, inner CV fold for C selection, inner CV fold for
threshold tuning -- refits the vectorizer and scaler on its own training rows
only. Only clause_text reaches the features (model_inputs + label-free guard).

Run: python src/evaluation/run_honest_variants.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.feature_variants import extract_encoder_swap_features
from src.evaluation.fold_metrics import CLASS_KEYS, fold_metrics
from src.evaluation.incremental_features import extract_all_rules_similarity
from src.training.model_inputs import build_input_texts, model_inputs
from src.training.train_classifier import apply_thresholds, tune_thresholds_inner_cv

ENCODER_NAME = "intfloat/e5-base-v2"
CLASSES = [cls for cls, _ in CLASS_KEYS]
C_GRID = [0.1, 0.3, 1.0, 3.0]
INNER_CV_SPLITS = 3
OUTPUT_DIR = Path("reports/experiments/confirmation")
SUMMARY_PREFIX = Path("reports/experiments/honest_variants_summary")


def make_lr(C):
    """Same LogisticRegression as run_confirmation_experiments.py, with C exposed."""
    return LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=42
    )


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)


def dense_model(C):
    """Current final model: RobustScaler on all features -> LogReg."""
    return Pipeline([("scale", RobustScaler()), ("clf", make_lr(C))])


def tfidf_model(C):
    """TF-IDF on clause text -> LogReg. X is a 1-D array of texts."""
    return Pipeline([("tfidf", make_tfidf()), ("clf", make_lr(C))])


def stacked_model(C):
    """TF-IDF (column 0) stacked with RobustScaler-ed dense features (columns 1:) -> LogReg."""
    features = ColumnTransformer(
        [("tfidf", make_tfidf(), 0), ("dense", RobustScaler(), slice(1, None))],
        sparse_threshold=1.0
    )
    return Pipeline([("features", features), ("clf", make_lr(C))])


def select_C(X, y, model_for_C, Cs):
    """
    Choose C by inner stratified CV on the training rows only: mean inner macro-F1
    of argmax predictions, refitting the whole pipeline per inner fold. Ties go to
    the smaller C (stronger regularisation).
    """
    if len(Cs) == 1:
        return Cs[0], {}
    inner = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)
    scores = {}
    for C in Cs:
        f1s = []
        for tr, va in inner.split(np.zeros(len(y)), y):
            model = model_for_C(C).fit(X[tr], y[tr])
            f1s.append(f1_score(y[va], model.predict(X[va]), average="macro", zero_division=0))
        scores[C] = float(np.mean(f1s))
    best = max(scores.values())
    return min(C for C, s in scores.items() if s == best), scores


def tune_binary_threshold_inner_cv(X, y, make_model, positive):
    """
    Binary analogue of tune_thresholds_inner_cv (same inner split, refit per inner
    fold, same best-inner-fold selection): predict `positive` when
    p(positive) >= tau, tau in [0.20, 0.80) step 0.05, maximising inner macro-F1.
    """
    inner = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)
    best_tau, best_f1 = 0.5, 0.0
    for tr, va in inner.split(np.zeros(len(y)), y):
        model = make_model().fit(X[tr], y[tr])
        p_pos = model.predict_proba(X[va])[:, list(model.classes_).index(positive)]
        negative = [c for c in model.classes_ if c != positive][0]
        for tau in np.arange(0.20, 0.80, 0.05):
            preds = np.where(p_pos >= tau, positive, negative)
            f1 = f1_score(y[va], preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, float(tau)
    return best_tau


def run_three_class(X, y, folds, model_for_C, Cs, decision):
    """Outer loop over confirmation folds for a 3-class variant."""
    results = []
    for f in folds:
        tr, te = np.array(f["train"]), np.array(f["test"])
        C, c_scores = select_C(X[tr], y[tr], model_for_C, Cs)
        make_model = lambda: model_for_C(C)
        model = make_model().fit(X[tr], y[tr])
        probs = model.predict_proba(X[te])
        record = {"seed": f["seed"], "fold": f["fold"], "C": C}
        if c_scores:
            record["inner_C_macro_f1"] = {str(k): v for k, v in c_scores.items()}
        if decision == "thresholds":
            tau_c, tau_p, _ = tune_thresholds_inner_cv(X[tr], y[tr], make_model, inner_cv_splits=INNER_CV_SPLITS)
            y_pred = np.array(apply_thresholds(probs, model.classes_, tau_c, tau_p))
            record.update(tau_compliant=tau_c, tau_partial=tau_p)
        else:
            y_pred = model.classes_[np.argmax(probs, axis=1)]
        record.update(fold_metrics(y[te], y_pred))
        record.update(y_true=[str(v) for v in y[te]], y_pred=[str(v) for v in y_pred])
        results.append(record)
    return results


def run_two_step(X, y, folds, model_for_C, Cs):
    """
    (a) Not Addressed vs Addressed on all rows; (b) Partially Compliant vs Compliant,
    trained on truly-Addressed training rows and scored on truly-Addressed test rows.
    Combined 3-class: test rows routed by (a); those predicted Addressed get (b)'s label.
    """
    y_a = np.where(y == "Not Addressed", "Not Addressed", "Addressed")
    results = []
    for f in folds:
        tr, te = np.array(f["train"]), np.array(f["test"])
        tr_b, te_b = tr[y[tr] != "Not Addressed"], te[y[te] != "Not Addressed"]

        C_a, _ = select_C(X[tr], y_a[tr], model_for_C, Cs)
        tau_a = tune_binary_threshold_inner_cv(X[tr], y_a[tr], lambda: model_for_C(C_a), "Addressed")
        model_a = model_for_C(C_a).fit(X[tr], y_a[tr])
        p_addr = model_a.predict_proba(X[te])[:, list(model_a.classes_).index("Addressed")]
        pred_a = np.where(p_addr >= tau_a, "Addressed", "Not Addressed")

        C_b, _ = select_C(X[tr_b], y[tr_b], model_for_C, Cs)
        tau_b = tune_binary_threshold_inner_cv(X[tr_b], y[tr_b], lambda: model_for_C(C_b), "Compliant")
        model_b = model_for_C(C_b).fit(X[tr_b], y[tr_b])
        p_comp = model_b.predict_proba(X[te])[:, list(model_b.classes_).index("Compliant")]
        pred_b_all = np.where(p_comp >= tau_b, "Compliant", "Partially Compliant")

        in_te_b = np.isin(te, te_b)
        combined = np.where(pred_a == "Addressed", pred_b_all, "Not Addressed")
        record = {
            "seed": f["seed"], "fold": f["fold"],
            "C_step_a": C_a, "C_step_b": C_b, "tau_step_a": tau_a, "tau_step_b": tau_b,
            "step_a_macro_f1": float(f1_score(y_a[te], pred_a, average="macro", zero_division=0)),
            "step_b_macro_f1": float(f1_score(y[te][in_te_b], pred_b_all[in_te_b], average="macro", zero_division=0)),
        }
        record.update(fold_metrics(y[te], combined))
        record.update(y_true=[str(v) for v in y[te]], y_pred=[str(v) for v in combined])
        results.append(record)
    return results


def summarize(results, extra_keys=()):
    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"] + [
        f"{key}_{m}" for _, key in CLASS_KEYS for m in ("precision", "recall", "f1")
    ] + list(extra_keys)
    return {f"{k}_{stat}": float(fn([r[k] for r in results]))
            for k in keys for stat, fn in (("mean", np.mean), ("std", np.std))}


def pooled_confusion(results):
    y_true = [v for r in results for v in r["y_true"]]
    y_pred = [v for r in results for v in r["y_pred"]]
    return confusion_matrix(y_true, y_pred, labels=CLASSES)


def print_class_report(title, results):
    agg = summarize(results)
    cm = pooled_confusion(results)
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"Macro-F1: {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}   "
          f"Accuracy: {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
    print(f"{'Class':<22}{'Precision':>18}{'Recall':>18}{'F1':>18}")
    for cls, key in CLASS_KEYS:
        print(f"{cls:<22}" + "".join(
            f"{agg[f'{key}_{m}_mean']:>10.4f} ± {agg[f'{key}_{m}_std']:.3f}" for m in ("precision", "recall", "f1")))
    print("\nPooled confusion matrix over 25 folds (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    print("\nRow-normalised (recall):")
    print(pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=CLASSES, columns=CLASSES).round(3).to_string())


def main():
    df = pd.read_csv("corpus/labeled_clauses_bootstrap.csv")
    taxonomy = json.loads(Path("corpus/dpdp_taxonomy_final.json").read_text(encoding="utf-8"))
    folds_data = json.loads(Path("datasets/folds/confirm_folds_v1.json").read_text())
    folds = [{"seed": int(s), **f} for s, fs in folds_data["seeds"].items() for f in fs]
    y = df["verdict"].values
    print(f"[INFO] {len(df)} clauses, {len(folds)} confirmation folds (seeds {list(folds_data['seeds'])})")

    # Label-free inputs only
    inputs = model_inputs(df)
    feature_data = extract_encoder_swap_features(inputs, taxonomy, ENCODER_NAME)
    all_rules = extract_all_rules_similarity(inputs, taxonomy, ENCODER_NAME, rule_text_mode="full")
    X_dense = np.hstack([feature_data["embeddings"], feature_data["structured"], all_rules]).astype(np.float64)
    texts = np.array(build_input_texts(inputs), dtype=object)
    X_stacked = np.empty((len(df), 1 + X_dense.shape[1]), dtype=object)
    X_stacked[:, 0] = texts
    X_stacked[:, 1:] = X_dense

    configs = {
        "ref_argmax": dict(X=X_dense, model=dense_model, Cs=[1.0], decision="argmax",
                           description="E5+8+all-rules, LogReg C=1, argmax (current final model as reported, 0.6106)"),
        "v1_final": dict(X=X_dense, model=dense_model, Cs=[1.0], decision="thresholds",
                         description="E5+8+all-rules, LogReg C=1, inner-CV thresholds (current final model)"),
        "v2_tfidf": dict(X=texts, model=tfidf_model, Cs=[1.0], decision="thresholds",
                         description="TF-IDF (1-2 grams, sublinear, min_df=2) + LogReg C=1, inner-CV thresholds"),
        "v3_tfidf_stacked": dict(X=X_stacked, model=stacked_model, Cs=[1.0], decision="thresholds",
                                 description="TF-IDF + E5+8+all-rules, LogReg C=1, inner-CV thresholds"),
        "v4_C_select": dict(X=X_dense, model=dense_model, Cs=C_GRID, decision="thresholds",
                            description="E5+8+all-rules, LogReg C in {0.1,0.3,1,3} by inner CV, inner-CV thresholds"),
    }

    all_results = {}
    for name, cfg in configs.items():
        print(f"\n[RUN] {name}: {cfg['description']}")
        all_results[name] = run_three_class(cfg["X"], y, folds, cfg["model"], cfg["Cs"], cfg["decision"])
        if name == "v1_final":
            print_class_report("1. CURRENT FINAL MODEL, argmax (as reported)", all_results["ref_argmax"])
            print_class_report("1. CURRENT FINAL MODEL, inner-CV thresholds (reference for v2-v5)", all_results["v1_final"])

    # 5. Two-step on the best single-model variant among v1-v4
    candidates = ["v1_final", "v2_tfidf", "v3_tfidf_stacked", "v4_C_select"]
    best = max(candidates, key=lambda n: np.mean([r["macro_f1"] for r in all_results[n]]))
    cfg = configs[best]
    print(f"\n[RUN] v5_two_step on best config: {best}")
    all_results["v5_two_step"] = run_two_step(cfg["X"], y, folds, cfg["model"], cfg["Cs"])
    configs["v5_two_step"] = dict(description=f"Two-step on {best}: (a) NA vs Addressed, (b) Partial vs Compliant "
                                              f"(truly-Addressed rows), binary inner-CV thresholds", Cs=cfg["Cs"])

    # Save per-variant JSON + summary table
    reference = np.array([r["macro_f1"] for r in all_results["v1_final"]])
    rows = []
    for name, results in all_results.items():
        extra = ("step_a_macro_f1", "step_b_macro_f1") if name == "v5_two_step" else ()
        agg = summarize(results, extra)
        f1s = np.array([r["macro_f1"] for r in results])
        diff = f1s - reference
        data = {
            "tag": f"honest_{name}", "variant": name, "description": configs[name]["description"],
            "timestamp": datetime.now().isoformat(),
            "config": {"folds": "datasets/folds/confirm_folds_v1.json", "seeds": [45, 46, 47, 48, 49],
                       "n_folds": len(results), "C_grid": configs[name]["Cs"],
                       "input_text": "clause_text only (label-free, src/training/model_inputs.py)"},
            "aggregates": agg,
            "vs_v1_final": {"mean_diff": float(diff.mean()), "std_diff": float(diff.std()),
                            "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
                            "wilcoxon_two_sided_p": float(wilcoxon(diff).pvalue) if diff.any() else None},
            "pooled_confusion_matrix": {"labels": CLASSES, "counts": pooled_confusion(results).tolist()},
            "per_fold_results": results,
        }
        if name == "v5_two_step":
            data["config"]["best_config"] = best
        (OUTPUT_DIR / f"honest_{name}.json").write_text(json.dumps(data, indent=2))

        row = {"variant": name, "description": configs[name]["description"],
               "macro_f1": f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}"}
        for cls, key in CLASS_KEYS:
            row[f"{cls} F1"] = f"{agg[f'{key}_f1_mean']:.4f} ± {agg[f'{key}_f1_std']:.4f}"
        row["Δ vs v1 (wins/25)"] = f"{diff.mean():+.4f} ({int((diff > 0).sum())}/25)"
        row["Wilcoxon p vs v1"] = f"{wilcoxon(diff).pvalue:.4f}" if diff.any() else ""
        if extra:
            row["step (a) macro-F1"] = f"{agg['step_a_macro_f1_mean']:.4f} ± {agg['step_a_macro_f1_std']:.4f}"
            row["step (b) macro-F1"] = f"{agg['step_b_macro_f1_mean']:.4f} ± {agg['step_b_macro_f1_std']:.4f}"
        if len(configs[name]["Cs"]) > 1:
            chosen = [str(c) for r in results for c in ([r["C"]] if "C" in r else [r["C_step_a"], r["C_step_b"]])]
            row["C chosen"] = ", ".join(f"{c}×{chosen.count(c)}" for c in sorted(set(chosen), key=float))
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(f"{SUMMARY_PREFIX}.csv", index=False)
    md_rows = [list(table.columns), ["---"] * len(table.columns)] + table.fillna("").astype(str).values.tolist()
    Path(f"{SUMMARY_PREFIX}.md").write_text(
        "\n".join("| " + " | ".join(r) + " |" for r in md_rows) + "\n", encoding="utf-8")
    print(f"\n{'=' * 100}\nSUMMARY (confirmation folds, seeds 45-49, 25 folds)\n{'=' * 100}")
    print(table.drop(columns=["description"]).fillna("").to_string(index=False))
    print(f"\n[SUCCESS] Saved {OUTPUT_DIR}/honest_*.json and {SUMMARY_PREFIX}.csv/.md")


if __name__ == "__main__":
    main()
