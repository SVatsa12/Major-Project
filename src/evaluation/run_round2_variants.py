"""
src/evaluation/run_round2_variants.py
Round-2 label-free variants, selected on the TRAINING folds only.

Reference v1 = current final model: E5 + 8 + all-rules, multinomial LogReg C=1,
inner-CV thresholds (tune_thresholds_inner_cv). One change per variant:
  r1_ordinal   two cumulative binary LogRegs, P(y > NA) and P(y > Partial);
               C in {0.1, 0.3, 1, 3} chosen by inner CV
  r2_context   + E5 of section heading (else previous clause), mean E5 of the
               previous/next clause in the same source document, relative position
  r3_augment   + 2 back-translation paraphrases (en->de->en, MarianMT) per Partially
               Compliant clause, added only from the rows a model is fit on
  r4_ensemble  mean probabilities of the best two eligible of r1-r3, inner-CV thresholds
ref_C_select (multinomial with the same C grid) is reported as context for r1 only;
it is not a candidate.

r2 eligibility (fixed before any result was seen): 55 corpus rows come from
hand-added single-clause supplemental documents (46 Compliant, 0 Not Addressed),
so "no document context" is a provenance proxy for Compliant. r2 is eligible for
selection and for the ensemble only if it also beats v1 on the rows that have real
document context (matched to the full extraction).

Every model is an index-based estimator, so the existing inner CV (C selection and
thresholds) refits scaler, classifier and augmentation on its own training rows.

Stages:
  python src/evaluation/run_round2_variants.py --stage train     # folds 42-44, all variants, writes selection
  python src/evaluation/run_round2_variants.py --stage confirm   # folds 45-49, selected config only, once
"""

import argparse
import difflib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.build_backtranslation_cache import CACHE_PATH as BT_CACHE_PATH, text_key
from src.evaluation.feature_variants import extract_encoder_swap_features, load_or_compute_embeddings
from src.evaluation.fold_metrics import CLASS_KEYS, fold_metrics
from src.evaluation.incremental_features import extract_all_rules_similarity
from src.evaluation.run_honest_variants import (
    C_GRID, ENCODER_NAME, INNER_CV_SPLITS, make_lr, pooled_confusion, print_class_report, summarize,
)
from src.training.model_inputs import build_input_texts, model_inputs
from src.training.train_classifier import apply_thresholds, tune_thresholds_inner_cv

FULL_CLAUSES = Path("datasets/SocialMediaPolicies/social_media_clauses.csv")
TRAIN_FOLDS = Path("datasets/folds/fixed_folds_v1.json")
CONFIRM_FOLDS = Path("datasets/folds/confirm_folds_v1.json")
OUT_DIR = Path("reports/experiments/round2")
SELECTION_PATH = OUT_DIR / "selection.json"
FUZZY_MATCH_RATIO = 0.90
CANDIDATES = ["r1_ordinal", "r2_context", "r3_augment"]


def normalise(text):
    return " ".join(str(text).split()).lower()


# ---------------------------------------------------------------- features

def base_features(inputs, taxonomy):
    """E5 + 8 actionability + all-rules (the v1 feature set) for label-free inputs."""
    feature_data = extract_encoder_swap_features(inputs, taxonomy, ENCODER_NAME)
    all_rules = extract_all_rules_similarity(inputs, taxonomy, ENCODER_NAME, rule_text_mode="full")
    return np.hstack([feature_data["embeddings"], feature_data["structured"], all_rules]).astype(np.float64)


def embed_texts(texts, cache_suffix):
    """E5 ("query: " + text) for document-structure texts (headings, neighbouring clauses)."""
    inputs = model_inputs(pd.DataFrame({"clause_text": texts}))
    return np.asarray(load_or_compute_embeddings(build_input_texts(inputs), ENCODER_NAME, cache_suffix=cache_suffix),
                      dtype=np.float64)


def match_to_full_extraction(df, full):
    """
    Locate each corpus clause in the full extraction: exact normalised text within the
    same source_document, else best difflib ratio >= FUZZY_MATCH_RATIO within it.
    Returns the full-extraction row index per corpus row (-1 if not found).
    """
    full = full.assign(_t=full["clause_text"].map(normalise))
    exact = {}
    for i, row in full.iterrows():
        exact.setdefault((row["source_document"], row["_t"]), i)
    by_doc = {doc: grp for doc, grp in full.groupby("source_document")}
    matches = []
    for _, row in df.iterrows():
        t = normalise(row["clause_text"])
        i = exact.get((row["source_document"], t), -1)
        if i < 0 and row["source_document"] in by_doc:
            cand = by_doc[row["source_document"]]
            ratios = cand["_t"].map(lambda x: difflib.SequenceMatcher(None, t, x).ratio())
            if ratios.max() >= FUZZY_MATCH_RATIO:
                i = ratios.idxmax()
        matches.append(i)
    return np.array(matches)


def context_features(df, own_embeddings):
    """
    Document-context features from clause text and document structure only:
    [E5(section heading, else previous clause) | mean E5(previous, next clause) | relative position].
    Rows without known document structure use their own embedding and position 0.5
    (no missingness indicator). Returns (features, matched_mask, provenance_counts).
    """
    full = pd.read_csv(FULL_CLAUSES)
    full = full.sort_values(["source_document", "clause_index_in_document"]).reset_index(drop=True)
    match = match_to_full_extraction(df, full)
    matched = match >= 0

    clause_texts = full["clause_text"].astype(str).tolist()
    full_emb = embed_texts(clause_texts, "_fullextraction")
    headings = full["section_heading"]
    heading_texts = sorted({str(h) for h in headings.dropna()})
    heading_emb = dict(zip(heading_texts, embed_texts(heading_texts, "_headings"))) if heading_texts else {}

    doc_bounds = full.groupby("source_document").indices
    n, d = own_embeddings.shape
    heading_feat = own_embeddings.copy()
    neighbour_feat = own_embeddings.copy()
    position = np.full((n, 1), 0.5)
    for r in np.where(matched)[0]:
        i = match[r]
        doc_rows = doc_bounds[full.at[i, "source_document"]]
        k = int(np.where(doc_rows == i)[0][0])
        prev_i = doc_rows[k - 1] if k > 0 else None
        next_i = doc_rows[k + 1] if k + 1 < len(doc_rows) else None
        if pd.notna(headings.iat[i]):
            heading_feat[r] = heading_emb[str(headings.iat[i])]
        elif prev_i is not None:
            heading_feat[r] = full_emb[prev_i]
        neighbours = [full_emb[j] for j in (prev_i, next_i) if j is not None]
        if neighbours:
            neighbour_feat[r] = np.mean(neighbours, axis=0)
        position[r, 0] = k / max(1, len(doc_rows) - 1)

    full_docs = set(full["source_document"])
    provenance = {
        "matched_to_full_extraction": int(matched.sum()),
        "unmatched_doc_known_text_edited": int((~matched & df["source_document"].isin(full_docs)).sum()),
        "unmatched_supplemental_document": int((~df["source_document"].isin(full_docs)).sum()),
    }
    return np.hstack([heading_feat, neighbour_feat, position]), matched, provenance


def augmentation_table(df, taxonomy):
    """
    {corpus row index: (2, d) v1-feature matrix of its back-translation paraphrases}
    for every Partially Compliant row. Paraphrases come from the cache keyed by the
    row's own text, and the cache entry's source text must equal that text.
    """
    cache = json.loads(BT_CACHE_PATH.read_text(encoding="utf-8"))["entries"]
    rows, texts = [], []
    for r in np.where(df["verdict"].values == "Partially Compliant")[0]:
        entry = cache[text_key(df.at[r, "clause_text"])]
        assert normalise(entry["source_text"]) == normalise(df.at[r, "clause_text"]), f"cache mismatch for row {r}"
        for p in entry["paraphrases"]:
            rows.append(r)
            texts.append(p)
    feats = base_features(model_inputs(pd.DataFrame({"clause_text": texts})), taxonomy)
    table = {}
    for r, f in zip(rows, feats):
        table.setdefault(int(r), []).append(f)
    return {r: np.vstack(fs) for r, fs in table.items()}, len(texts)


# ---------------------------------------------------------------- models

class OrdinalLR:
    """Two cumulative binary LogRegs on RobustScaled features: P(y > NA), P(y > Partial)."""

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        y = np.asarray(y)
        self.scaler_ = RobustScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.gt_na_ = make_lr(self.C).fit(Z, y != "Not Addressed")
        self.gt_partial_ = make_lr(self.C).fit(Z, y == "Compliant")
        self.classes_ = np.array(sorted({cls for cls, _ in CLASS_KEYS}))
        return self

    def predict_proba(self, X):
        Z = self.scaler_.transform(X)
        p_gt_na = self.gt_na_.predict_proba(Z)[:, list(self.gt_na_.classes_).index(True)]
        p_gt_partial = np.minimum(self.gt_partial_.predict_proba(Z)[:, list(self.gt_partial_.classes_).index(True)], p_gt_na)
        by_class = {"Not Addressed": 1 - p_gt_na, "Partially Compliant": p_gt_na - p_gt_partial, "Compliant": p_gt_partial}
        return np.column_stack([by_class[c] for c in self.classes_])


class IndexModel:
    """
    Classifier over corpus row indices. fit(idx, y) looks up features[idx]; with
    `augment`, it appends the paraphrase rows of Partially Compliant rows in idx
    (and only those). C is chosen by inner CV on the fit rows when Cs has > 1 value.
    """

    def __init__(self, features, kind="multinomial", Cs=(1.0,), augment=None):
        self.features, self.kind, self.Cs, self.augment = features, kind, list(Cs), augment

    def _make(self, C):
        if self.kind == "ordinal":
            return OrdinalLR(C)
        return Pipeline([("scale", RobustScaler()), ("clf", make_lr(C))])

    def _select_C(self, idx, y):
        inner = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=42)
        scores = {}
        for C in self.Cs:
            f1s = []
            for tr, va in inner.split(np.zeros(len(y)), y):
                m = IndexModel(self.features, self.kind, (C,), self.augment).fit(idx[tr], y[tr])
                f1s.append(f1_score(y[va], m.predict(idx[va]), average="macro", zero_division=0))
            scores[C] = float(np.mean(f1s))
        best = max(scores.values())
        return min(C for C, s in scores.items() if s == best)

    def fit(self, idx, y):
        idx, y = np.asarray(idx), np.asarray(y)
        self.C_ = self._select_C(idx, y) if len(self.Cs) > 1 else self.Cs[0]
        X, yy = self.features[idx], y
        self.augmented_sources_ = []
        if self.augment is not None:
            src = [i for i, lab in zip(idx, y) if lab == "Partially Compliant" and i in self.augment]
            assert set(src) <= set(idx.tolist()), "augmentation source outside fit rows"
            if src:
                X_aug = np.vstack([self.augment[i] for i in src])
                X = np.vstack([X, X_aug])
                yy = np.concatenate([y, np.full(len(X_aug), "Partially Compliant", dtype=object)])
            self.augmented_sources_ = src
        self.model_ = self._make(self.C_).fit(X, yy)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, idx):
        return self.model_.predict_proba(self.features[np.asarray(idx)])

    def predict(self, idx):
        return self.classes_[np.argmax(self.predict_proba(idx), axis=1)]


class EnsembleModel:
    """Mean of member probabilities (aligned by class name)."""

    def __init__(self, member_factories):
        self.member_factories = member_factories

    def fit(self, idx, y):
        self.members_ = [make().fit(idx, y) for make in self.member_factories]
        self.classes_ = np.array(sorted({cls for cls, _ in CLASS_KEYS}))
        self.augmented_sources_ = sorted({s for m in self.members_ for s in m.augmented_sources_})
        self.C_ = [m.C_ for m in self.members_]
        return self

    def predict_proba(self, idx):
        probs = []
        for m in self.members_:
            p = m.predict_proba(idx)
            probs.append(np.column_stack([p[:, list(m.classes_).index(c)] for c in self.classes_]))
        return np.mean(probs, axis=0)

    def predict(self, idx):
        return self.classes_[np.argmax(self.predict_proba(idx), axis=1)]


# ---------------------------------------------------------------- evaluation

def load_folds(path):
    data = json.loads(Path(path).read_text())
    return [{"seed": int(s), **f} for s, fs in data["seeds"].items() for f in fs], [int(s) for s in data["seeds"]]


def run_variant(make_model, y, folds, matched, norm_texts):
    results = []
    for f in folds:
        tr, te = np.array(f["train"]), np.array(f["test"])
        model = make_model().fit(tr, y[tr])
        # No augmented text may derive from a test-fold row (by index or by identical text)
        test_texts = {norm_texts[i] for i in te}
        assert not set(model.augmented_sources_) & set(te.tolist()), "augmentation from a test-fold row"
        assert not {norm_texts[i] for i in model.augmented_sources_} & test_texts, "augmented text matches a test-fold text"
        tau_c, tau_p, _ = tune_thresholds_inner_cv(tr, y[tr], make_model, inner_cv_splits=INNER_CV_SPLITS)
        y_pred = np.array(apply_thresholds(model.predict_proba(te), model.classes_, tau_c, tau_p))
        te_m = matched[te]
        record = {"seed": f["seed"], "fold": f["fold"], "C": model.C_, "tau_compliant": tau_c, "tau_partial": tau_p,
                  "n_augmented_rows": 2 * len(model.augmented_sources_) if model.augmented_sources_ else 0}
        record.update(fold_metrics(y[te], y_pred))
        record["matched_rows_macro_f1"] = float(f1_score(y[te][te_m], y_pred[te_m], average="macro", zero_division=0))
        record.update(test_idx=te.tolist(), y_true=[str(v) for v in y[te]], y_pred=[str(v) for v in y_pred])
        results.append(record)
    return results


def compare(results, reference, key="macro_f1"):
    diff = np.array([r[key] for r in results]) - np.array([r[key] for r in reference])
    p = float(wilcoxon(diff).pvalue) if diff.any() else None
    return {"mean_diff": float(diff.mean()), "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": p}


def save_variant(path, name, description, results, reference, extra):
    agg = summarize(results, ("matched_rows_macro_f1",))
    data = {"tag": path.stem, "variant": name, "description": description, "timestamp": datetime.now().isoformat(),
            "config": extra, "aggregates": agg,
            "vs_v1": compare(results, reference) if reference is not results else None,
            "vs_v1_matched_rows": compare(results, reference, "matched_rows_macro_f1") if reference is not results else None,
            "pooled_confusion_matrix": {"labels": [c for c, _ in CLASS_KEYS], "counts": pooled_confusion(results).tolist()},
            "per_fold_results": results}
    path.write_text(json.dumps(data, indent=2))
    return data


def table_row(name, data, n_folds):
    agg, cmp_, cmp_m = data["aggregates"], data["vs_v1"], data["vs_v1_matched_rows"]
    row = {"variant": name, "macro_f1": f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}"}
    for cls, key in CLASS_KEYS:
        row[f"{cls} F1"] = f"{agg[f'{key}_f1_mean']:.4f} ± {agg[f'{key}_f1_std']:.4f}"
    row[f"wins vs v1 (/{n_folds})"] = f"{cmp_['wins']} ({cmp_['mean_diff']:+.4f})" if cmp_ else "—"
    row["Wilcoxon p"] = f"{cmp_['p']:.4f}" if cmp_ and cmp_["p"] is not None else "—"
    row["matched-rows macro-F1"] = f"{agg['matched_rows_macro_f1_mean']:.4f}"
    row["Δ matched vs v1"] = f"{cmp_m['mean_diff']:+.4f}" if cmp_m else "—"
    return row


def write_table(rows, prefix):
    table = pd.DataFrame(rows)
    table.to_csv(f"{prefix}.csv", index=False)
    md = [list(table.columns), ["---"] * len(table.columns)] + table.astype(str).values.tolist()
    Path(f"{prefix}.md").write_text("\n".join("| " + " | ".join(r) + " |" for r in md) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "confirm"], required=True)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("corpus/labeled_clauses_bootstrap.csv")
    taxonomy = json.loads(Path("corpus/dpdp_taxonomy_final.json").read_text(encoding="utf-8"))
    y = df["verdict"].values
    norm_texts = [normalise(t) for t in df["clause_text"]]

    # Label-free inputs only; labels are used solely as training targets
    inputs = model_inputs(df)
    B = base_features(inputs, taxonomy)
    own_emb = B[:, :768]
    ctx, matched, provenance = context_features(df, own_emb)
    BC = np.hstack([B, ctx])
    aug, n_paraphrases = augmentation_table(df, taxonomy)
    print(f"[INFO] base {B.shape}, base+context {BC.shape}, provenance {provenance}, "
          f"{n_paraphrases} paraphrases for {len(aug)} Partially Compliant rows")

    factories = {
        "v1_final": lambda: IndexModel(B),
        "ref_C_select": lambda: IndexModel(B, Cs=C_GRID),
        "r1_ordinal": lambda: IndexModel(B, kind="ordinal", Cs=C_GRID),
        "r2_context": lambda: IndexModel(BC),
        "r3_augment": lambda: IndexModel(B, augment=aug),
    }
    descriptions = {
        "v1_final": "E5+8+all-rules, multinomial LogReg C=1, inner-CV thresholds (reference)",
        "ref_C_select": "v1 with C in {0.1,0.3,1,3} by inner CV (context for r1; not a candidate)",
        "r1_ordinal": "ordinal: cumulative LogRegs P(y>NA), P(y>Partial); C in {0.1,0.3,1,3} by inner CV",
        "r2_context": "v1 + E5(section heading, else previous clause) + mean E5(prev/next clause) + relative position",
        "r3_augment": "v1 + 2 back-translation paraphrases (en-de-en MarianMT) per Partially Compliant training row",
    }

    if args.stage == "train":
        folds, seeds = load_folds(TRAIN_FOLDS)
        results = {}
        for name in ["v1_final", "ref_C_select"] + CANDIDATES:
            print(f"\n[RUN] {name}: {descriptions[name]}")
            results[name] = run_variant(factories[name], y, folds, matched, norm_texts)
        print_class_report("v1 on training folds (reference)", results["v1_final"])

        mean_f1 = {n: float(np.mean([r["macro_f1"] for r in rs])) for n, rs in results.items()}
        matched_f1 = {n: float(np.mean([r["matched_rows_macro_f1"] for r in rs])) for n, rs in results.items()}
        eligible = [n for n in CANDIDATES if n != "r2_context" or matched_f1["r2_context"] > matched_f1["v1_final"]]
        members = sorted(eligible, key=lambda n: mean_f1[n], reverse=True)[:2]
        descriptions["r4_ensemble"] = f"mean probabilities of {members[0]} + {members[1]}, inner-CV thresholds"
        factories["r4_ensemble"] = lambda: EnsembleModel([factories[members[0]], factories[members[1]]])
        print(f"\n[RUN] r4_ensemble: {descriptions['r4_ensemble']}")
        results["r4_ensemble"] = run_variant(factories["r4_ensemble"], y, folds, matched, norm_texts)
        mean_f1["r4_ensemble"] = float(np.mean([r["macro_f1"] for r in results["r4_ensemble"]]))

        rows, saved = [], {}
        for name, rs in results.items():
            saved[name] = save_variant(OUT_DIR / f"train_{name}.json", name, descriptions[name], rs, results["v1_final"],
                                       {"folds": str(TRAIN_FOLDS), "seeds": seeds, "provenance": provenance})
            rows.append(table_row(name, saved[name], len(folds)))
        print(f"\n{'=' * 100}\nTRAINING FOLDS (seeds {seeds}, {len(folds)} folds)\n{'=' * 100}")
        write_table(rows, OUT_DIR / "train_summary")

        selectable = ["v1_final"] + eligible + ["r4_ensemble"]
        best = max(selectable, key=lambda n: mean_f1[n])
        selection = {"best": best, "selected_by": "training-fold mean macro-F1 only", "candidates": selectable,
                     "training_macro_f1": {n: mean_f1[n] for n in selectable}, "ensemble_members": members,
                     "r2_eligible": "r2_context" in eligible,
                     "r2_matched_rows_macro_f1": {"r2_context": matched_f1["r2_context"], "v1_final": matched_f1["v1_final"]},
                     "timestamp": datetime.now().isoformat()}
        SELECTION_PATH.write_text(json.dumps(selection, indent=2))
        print(f"\n[SELECTED] {best} (training macro-F1 {mean_f1[best]:.4f}); r2 eligible: {'r2_context' in eligible}; "
              f"ensemble members: {members}")

    else:
        selection = json.loads(SELECTION_PATH.read_text())
        best = selection["best"]
        out = OUT_DIR / f"confirm_{best}.json"
        assert not out.exists(), f"{out} exists: the selected config is run on the confirmation folds only once"
        if best == "v1_final":
            print("[INFO] v1 was best on training folds; its confirmation result is honest_v1_final.json")
            return
        if best == "r4_ensemble":
            members = selection["ensemble_members"]
            descriptions[best] = f"mean probabilities of {members[0]} + {members[1]}, inner-CV thresholds"
            factories[best] = lambda: EnsembleModel([factories[members[0]], factories[members[1]]])
        folds, seeds = load_folds(CONFIRM_FOLDS)
        v1_confirm = json.loads(Path("reports/experiments/confirmation/honest_v1_final.json").read_text())["per_fold_results"]
        assert [(r["seed"], r["fold"]) for r in v1_confirm] == [(f["seed"], f["fold"]) for f in folds]
        print(f"[RUN] {best} on confirmation folds (once): {descriptions[best]}")
        results = run_variant(factories[best], y, folds, matched, norm_texts)
        for r, ref in zip(results, v1_confirm):  # matched-row F1 for the stored v1 predictions
            te_m = matched[np.array(r["test_idx"])]
            ref["matched_rows_macro_f1"] = float(f1_score(np.array(ref["y_true"])[te_m], np.array(ref["y_pred"])[te_m],
                                                          average="macro", zero_division=0))
        data = save_variant(out, best, descriptions[best], results, v1_confirm,
                            {"folds": str(CONFIRM_FOLDS), "seeds": seeds, "selection": selection, "provenance": provenance})
        print_class_report(f"{best} on confirmation folds", results)
        write_table([table_row(best, data, len(folds))], OUT_DIR / "confirm_summary")


if __name__ == "__main__":
    main()
