# DPDP Compliance Classifier - Experiment Log

> ## ⛔ ALL RESULTS BELOW THIS BANNER ARE INVALID — LABEL LEAKAGE (found 2026-10-27)
>
> Every experiment from commit `1789082` (2026-10-12) to `9e410c5` built model inputs with
> `build_context_texts()`, which inserted the **LLM-chosen rule** into the text:
> `"[app] Mandate: {checkable_test of best_match_taxonomy_id} | Policy: {clause}"`, or
> `"Mandate: General Policy"` when the LLM chose `NONE`. In `labeled_clauses_bootstrap.csv`,
> `best_match_taxonomy_id == "NONE"` ⇔ `verdict == "Not Addressed"` for all 586 Not Addressed rows
> (and for none of the 240 others), so the label was in every model's input. This covers Phase B, Phase E,
> every ablation variant, every training-fold, confirmation and pooled number, the paired comparison,
> the confusion matrix and the 0.8132 headline. They are kept below for the record only and **must not be cited**.
>
> **Fix:** `build_context_texts` was removed. Inputs are now `"query: " + clause_text` (E5; MiniLM gets no prefix),
> built by `src/training/model_inputs.py`. Feature builders reject any LLM output column
> (`tests/test_label_leakage.py`). Same fold files (`groups_v1.json`, `fixed_folds_v1.json`, `confirm_folds_v1.json`),
> same classifiers and scripts; nothing re-tuned.

## Label-free results (VALID, 2026-10-27)

Mean ± std over folds. Result files: `reports/experiments/**/*_labelfree.json`; tables: `reports/paper/`.

### Confirmation folds (seeds 45–49, 25 folds) — primary

| Model | Macro-F1 | Accuracy | NA P / R / F1 | Partial P / R / F1 | Compliant P / R / F1 |
|---|---|---|---|---|---|
| Baseline (MiniLM+8) | 0.5662 ± 0.0425 | 0.7096 ± 0.0308 | 0.857 ± 0.039 / 0.819 ± 0.038 / 0.837 ± 0.027 | 0.355 ± 0.106 / 0.414 ± 0.105 / 0.378 ± 0.099 | 0.490 ± 0.115 / 0.502 ± 0.119 / 0.484 ± 0.081 |
| E5+8 | **0.6127 ± 0.0353** | 0.7391 ± 0.0327 | 0.855 ± 0.040 / 0.846 ± 0.032 / 0.850 ± 0.025 | 0.367 ± 0.089 / 0.387 ± 0.102 / 0.372 ± 0.084 | 0.625 ± 0.108 / 0.625 ± 0.103 / 0.616 ± 0.071 |
| E5+8+all-rules | 0.6106 ± 0.0323 | 0.7381 ± 0.0290 | 0.855 ± 0.038 / 0.846 ± 0.033 / 0.850 ± 0.023 | 0.367 ± 0.092 / 0.386 ± 0.101 / 0.372 ± 0.085 | 0.619 ± 0.104 / 0.619 ± 0.101 / 0.610 ± 0.072 |

### Pooled (seeds 42–49, 40 folds = 15 training + 25 confirmation)

| Model | Macro-F1 | Accuracy | NA P / R / F1 | Partial P / R / F1 | Compliant P / R / F1 |
|---|---|---|---|---|---|
| Baseline (MiniLM+8) | 0.5766 ± 0.0438 | 0.7065 ± 0.0396 | 0.855 ± 0.042 / 0.803 ± 0.053 / 0.827 ± 0.037 | 0.359 ± 0.096 / 0.470 ± 0.127 / 0.401 ± 0.095 | 0.556 ± 0.152 / 0.486 ± 0.106 / 0.502 ± 0.077 |
| E5+8 | 0.6085 ± 0.0349 | 0.7270 ± 0.0400 | 0.848 ± 0.045 / 0.828 ± 0.045 / 0.837 ± 0.035 | 0.362 ± 0.089 / 0.425 ± 0.117 / 0.384 ± 0.086 | 0.661 ± 0.131 / 0.580 ± 0.107 / 0.604 ± 0.071 |
| E5+8+all-rules | **0.6119 ± 0.0301** | 0.7391 ± 0.0293 | 0.853 ± 0.044 / 0.850 ± 0.033 / 0.850 ± 0.024 | 0.371 ± 0.089 / 0.386 ± 0.091 / 0.373 ± 0.077 | 0.628 ± 0.101 / 0.615 ± 0.102 / 0.612 ± 0.069 |

### Training folds only (seeds 42–44, 15 folds)

| Model | Run | Macro-F1 |
|---|---|---|
| Baseline (MiniLM+8) | `fixed_baseline_v1_labelfree` | 0.5939 ± 0.0404 |
| E5+8 | `fixed_A_e5_labelfree` | 0.6014 ± 0.0329 |
| E5+8+all-rules | `incr_1_all_rules_labelfree` | 0.6139 ± 0.0257 |

**Reading these numbers**
- Removing the leak costs about 0.20 macro-F1 (confirmation, E5+8+all-rules: 0.8132 → 0.6106). Most of the loss is on Partially Compliant (F1 ≈ 0.37) and on separating it from Not Addressed: 46% of Partially Compliant clauses are predicted Not Addressed (`reports/paper/confusion_matrix_normalized.csv`).
- E5 beats MiniLM (paired, confirmation: E5+8+all-rules wins 22/25 folds, mean +0.044, Wilcoxon p < 0.001). E5+8 and E5+8+all-rules are **tied** (confirmation 0.6127 vs 0.6106; pooled 0.6085 vs 0.6119, both well within 1 std). The earlier claim that all-rules features "generalize better" is not supported.
- Caveat inherited from the existing scripts (unchanged here): the pooled baseline and E5+8 rows mix two decision rules. Their 15 training folds come from `run_feature_experiments.py` (inner-CV thresholds via `train_and_evaluate_fold`), while their 25 confirmation folds and every all-rules fold use plain argmax LogisticRegression.
- Caveat: the near-duplicate groups in `fixed_folds_v1.json`/`groups_v1.json` were computed (MiniLM, cos ≥ 0.90) on the old leaky context texts. Groups only decide fold membership and never enter features, so they cannot leak the label. They may be slightly different from groups computed on clause text alone. Kept unchanged as instructed.

### Final model (retrained 2026-10-27)

`models/final_classifier/`: E5+8+all-rules (825 features) trained on all 826 clauses with `"query: " + clause_text` inputs, the same feature functions as evaluation, and thresholds tuned by real 3-fold inner CV (refit per inner fold, `tune_thresholds_inner_cv`): τ_compliant = 0.50, τ_partial = 0.25. No probability calibration. The 8 true actionability feature names are in `feature_config.json`. The 20-clause smoke test passes: training vs inference features max |diff| 4.2e-7, classifier vs API 20/20 identical.

---


This document chronicles all experimental phases of the DPDP compliance classifier development, from initial feature engineering through final model selection.

---

## Phase B: Feature Union Experiment — ⛔ INVALID (label leakage)

**Date**: 2026-10-14  
**Branch**: `phase-b-feature-union` (created from main @ commit 1789082)  
**Status**: ❌ **REJECTED** — Failed acceptance bar (gain too low)  
**Command**: `python -u src/evaluation/evaluate_5fold_cv.py`

### Results Summary — ⛔ INVALID (label leakage)

| Metric | Baseline | Phase B | Δ | Status |
|--------|----------|---------|---|--------|
| ⛔ INVALID (label leakage): Mean Macro-F1 | 0.7712 | 0.7732 | +0.0020 | ✅ PASS* |
| ⛔ INVALID (label leakage): Gain (Δ F1) | — | +0.0020 | Required ≥ 0.01 | ❌ **FAIL** |
| ⛔ INVALID (label leakage): Std Macro-F1 | 0.0398 | 0.0331 | -0.0067 | ✅ Better |
| ⛔ INVALID (label leakage): Accuracy | 0.8826 | 0.8862 | +0.0036 | ✅ Better |
| ⛔ INVALID (label leakage): Max Class F1 Drop | — | -0.003 | Required ≤ 0.02 | ✅ PASS |
| Features | 8 | 15 | +7 | N/A |

*Metric 1 passed (F1 > 0.7712) but insufficient gain fails overall acceptance.

### Acceptance Bar Evaluation — ⛔ INVALID (label leakage)

> ⛔ INVALID (label leakage): numbers in this block came from leaky inputs.
```
Criterion 1: Mean Macro-F1 > 0.7712       ✅ PASS   (0.7732 > 0.7712)
Criterion 2: Gain >= 0.01                 ❌ FAIL   (0.0020 < 0.01) ← BLOCKER
Criterion 3: No class F1 drop > 0.02      ✅ PASS   (max drop -0.003)

OVERALL VERDICT: ❌ REJECTED
```

### Recommendation — ⛔ INVALID (label leakage)

**Do NOT merge Phase B to main.** The feature union does not justify the expanded feature set. Consider alternative directions for Phase A/C/D/E.

---

## Phase E: MPNet Encoder Experiment — ⛔ INVALID (label leakage)

**Date**: 2026-10-14  
**Branch**: `phase-e-mpnet-encoder` (created from main @ commit 1789082)  
**Status**: ❌ **REJECTED** — Failed all acceptance criteria (F1 regression, per-class instability)  
**Command**: `python -u src/evaluation/evaluate_5fold_cv.py`

### Results Summary — ⛔ INVALID (label leakage)

| Metric | Baseline | Phase E | Δ | Status |
|--------|----------|---------|---|--------|
| ⛔ INVALID (label leakage): Mean Macro-F1 | 0.7712 | 0.7353 | -0.0359 | ❌ FAIL |
| ⛔ INVALID (label leakage): Std Macro-F1 | 0.0398 | 0.0349 | -0.0049 | ✅ Better |
| ⛔ INVALID (label leakage): Accuracy | 0.8826 | 0.8499 | -0.0327 | ❌ FAIL |
| ⛔ INVALID (label leakage): F1: Not Addressed | 0.9614 | 0.9331 | -0.0283 | ❌ DROP |
| ⛔ INVALID (label leakage): F1: Partially Compliant | 0.7405 | 0.6994 | -0.0411 | ❌ FAIL |
| ⛔ INVALID (label leakage): F1: Compliant | 0.6082 | 0.5677 | -0.0405 | ❌ FAIL |

### Acceptance Bar Evaluation — ⛔ INVALID (label leakage)

> ⛔ INVALID (label leakage): numbers in this block came from leaky inputs.
```
CRITERION 1: Mean Macro-F1 > 0.7712       ❌ FAIL (regression -0.0359, -4.66%)
CRITERION 2: Gain >= 0.01                 ❌ FAIL (negative gain)
CRITERION 3: No class F1 drop > 0.02      ❌ FAIL (max drop -0.0411)

OVERALL VERDICT: ❌ REJECTED
```

### Recommendation — ⛔ INVALID (label leakage)

**Do NOT use all-mpnet-base-v2 for production.** Keep all-MiniLM-L6-v2 (384-d) as baseline encoder.

---

## Taxonomy V2 & Final Model Development — ⛔ INVALID (label leakage)

**Date**: 2026-10-22 to 2026-10-23  
**Branch**: `taxonomy-v2`  
**Status**: ✅ **ACCEPTED** — Selected as final model for paper

### Context — ⛔ INVALID (label leakage)

After initial experiments with feature engineering (Phase B) and encoder swaps (Phase E), the project pivoted to:
1. Upgrading the taxonomy from 42 to 38 high-quality rules (taxonomy v2)
2. Systematic feature ablation study
3. Encoder comparison (MiniLM, E5-small, E5-base, BGE-base, InLegalBERT)
4. All-rules similarity features
5. Confirmation fold validation (seeds 45-49)

### Feature Ablation Results — ⛔ INVALID (label leakage)

Comprehensive ablation study conducted with 5-fold CV (seeds 42, 43, 44):

| Configuration | Features | Macro-F1 | Std | Status |
|--------------|----------|----------|-----|--------|
| ⛔ INVALID (label leakage): Baseline (MiniLM+8) | 392 | 0.7473 | 0.0354 | Baseline |
| ⛔ INVALID (label leakage): E5-small+8 | 392 | 0.7703 | 0.0293 | Better |
| ⛔ INVALID (label leakage): E5-base+8 | 776 | **0.8118** | 0.0315 | **Best (train)** |
| ⛔ INVALID (label leakage): E5-base+8+all-rules | **825** | 0.8118 | 0.0315 | **SELECTED** |
| ⛔ INVALID (label leakage): BGE-base+8 | 776 | 0.7823 | 0.0292 | Good |
| ⛔ INVALID (label leakage): InLegalBERT+8 | 776 | 0.6802 | 0.0470 | Poor |

**Key Finding**: E5-base encoder provides best performance. All-rules similarity features add interpretability without hurting performance.

### Confirmation Fold Validation (Seeds 45-49) — ⛔ INVALID (label leakage)

Independent validation on 25 new folds to check generalization:

| Configuration | Macro-F1 (train) | Macro-F1 (confirm) | Δ | Status |
|--------------|------------------|-------------------|---|--------|
| ⛔ INVALID (label leakage): Baseline (MiniLM+8) | 0.7473 ± 0.035 | 0.7347 ± 0.052 | -1.3% | Stable |
| ⛔ INVALID (label leakage): E5-base+8 | 0.8118 ± 0.032 | **0.8160 ± 0.034** | +0.4% | **Variable** |
| ⛔ INVALID (label leakage): E5-base+8+all-rules | 0.8118 ± 0.032 | **0.8132 ± 0.030** | +0.1% | **Consistent** |

**Analysis Correction**: Initial analysis incorrectly flagged all-rules as "overfitting" due to -0.3% change. Corrected analysis shows:
- E5-only: 0.7703 → 0.8160 (+4.57% variance, seed-dependent)
- E5+all-rules: 0.8118 → 0.8132 (+0.14% consistent generalization)
- **Conclusion**: All-rules generalizes well; E5-only shows high variance

See `docs/EXPERIMENT_LOG_CORRECTION.md` for detailed correction analysis.

### Final Model Selection: E5-base + 8 + all-rules — ⛔ INVALID (label leakage)

**Configuration**:
- **Encoder**: intfloat/e5-base-v2 (768 dims, L2-normalized)
- **Structured Features**: 8 actionability features (RobustScaler)
- **All-Rules Similarity**: 49 features (38 rule cosine similarities + 11 category max)
- **Total Features**: 825
- **Classifier**: LogisticRegression (C=1.0, class_weight='balanced', max_iter=1000)
- **Threshold Calibration**: 3-fold inner CV for optimal decision boundaries

**Rationale**:
1. **Best pooled performance**: 0.8127 macro-F1 across 40 folds (+1.4% over E5-only)
2. **Consistent generalization**: +0.14% from train to confirm (vs +4.57% for E5-only)
3. **Interpretability**: All-rules features provide statutory rule attribution
4. **Lower variance**: 0.0308 std (vs 0.0398 for E5-only)

### Final Model Training — ⛔ INVALID (label leakage)

**Date**: 2026-10-23 02:05 UTC  
**Script**: `src/training/train_final_model.py`  
**Git Commit**: a33477d3b0002c55416085ec5488206f1a90fa1c

**Training Configuration**:
```
Corpus: labeled_clauses_bootstrap.csv (826 clauses)
Taxonomy: dpdp_taxonomy_final.json (38 rules)
Class Distribution:
  - Not Addressed: 586 (71.0%)
  - Partially Compliant: 143 (17.3%)
  - Compliant: 97 (11.7%)

Feature Extraction:
  - E5-base embeddings: 768 dims
  - Actionability features: 8 dims
  - All-rules similarity: 49 dims (38 rule + 11 category max)
  - Total: 825 features

Model: LogisticRegression
  - C: 1.0
  - class_weight: balanced
  - max_iter: 1000
  - random_state: 42

Threshold Calibration: 3-fold inner CV
  - Method: Grid search (tau_c: 0.3-0.8, tau_p: 0.3-0.8)
  - Objective: Maximize macro-F1
```

**Output**:
- Model: `models/final_classifier/model.joblib`
- Scaler: `models/final_classifier/scaler.joblib`
- Thresholds: `models/final_classifier/thresholds.json`
- Configuration: `models/final_classifier/model_config.json`
- Features: `models/final_classifier/feature_config.json`

### Confirmation Results (Paper - PRIMARY) — ⛔ INVALID (label leakage)

| Model | Accuracy | Macro-F1 | Compliant-F1 |
|-------|----------|----------|--------------|
| ⛔ INVALID (label leakage): Baseline (MiniLM+8) | 0.8740 ± 0.033 | 0.7347 ± 0.052 | 0.5471 ± 0.109 |
| ⛔ INVALID (label leakage): E5+8 | 0.9182 ± 0.020 | 0.8160 ± 0.034 | 0.6728 ± 0.082 |
| ⛔ INVALID (label leakage): **E5+8+all-rules (FINAL)** | **0.9166 ± 0.019** | **0.8132 ± 0.030** | **0.6688 ± 0.073** |

**Statistical Significance**: 
- Wilcoxon signed-rank test: E5+8+all-rules vs Baseline
- Final model wins: 24/25 folds (96.0%)
- p-value: < 0.0001 (highly significant)
- Mean difference: +0.0786 ± 0.054 macro-F1

### Paper Deliverables Generated — ⛔ INVALID (label leakage)

**Date**: 2026-10-23  
**Location**: `reports/paper/`

1. **Results Tables** (CSV + LaTeX)
   - Confirmation results (primary): 25 folds, seeds 45-49
   - Pooled results (secondary): 40 folds, seeds 42-49

2. **Confusion Matrix**
   - PNG visualization
   - Counts CSV (2805/573/246 correct predictions per class)
   - Row-normalized CSV (0.957/0.801/0.507 recall per class)

3. **Paired Comparison**
   - Fold-by-fold comparison CSV
   - Summary statistics CSV
   - Wilcoxon test results JSON (p < 0.0001)

4. **Top Features**
   - Top 10 most influential all-rules features per class
   - CSV and JSON formats with rule IDs and statutory text

5. **Final Model Artifacts**
   - Trained model (825 features, all 826 clauses)
   - Calibrated thresholds
   - Full configuration and metadata
   - Git commit hash: a33477d3b0002c55416085ec5488206f1a90fa1c

---

## Final Status — ⛔ INVALID (label leakage)

✅ **PRODUCTION READY**

**Selected Model**: E5-base + 8 actionability + all-rules similarity (825 features)  
**Performance**: 0.8132 ± 0.030 macro-F1 on confirmation folds  
**Tag**: `paper-results-v1`  
**Branch**: `taxonomy-v2` → `cleanup/taxonomy-v2`

See `docs/PAPER_DELIVERABLES_SUMMARY.md` for complete paper results documentation.

---

## Files Generated — ⛔ INVALID (label leakage)

- `reports/experiments/fixed_baseline_v1.json` - Baseline results (seeds 42-44)
- `reports/experiments/fixed_A_e5.json` - E5-base results (seeds 42-44)
- `reports/experiments/incr_1_all_rules.json` - E5+all-rules results (seeds 42-44)
- `reports/experiments/confirmation/confirm_baseline_v1_corrected.json` - Baseline (seeds 45-49)
- `reports/experiments/confirmation/confirm_fixed_A_e5_corrected.json` - E5-base (seeds 45-49)
- `reports/experiments/confirmation/confirm_all_rules_corrected.json` - E5+all-rules (seeds 45-49)
- `reports/paper/*` - All paper deliverables (tables, confusion matrix, comparisons, features)
- `models/final_classifier/*` - Final trained model and artifacts

---

**Last Updated**: 2026-10-23  
**Pipeline Status**: Complete and reproducible
