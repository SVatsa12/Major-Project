# Complete Ablation Study: All Variants Tested

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

### Final model (retrained 2026-10-27, ordinal): **0.6436 ± 0.0374** confirmation macro-F1

`models/final_classifier/` is the round-2 winner **r1_ordinal**. It was selected by training-fold macro-F1 only and run once on the confirmation folds; see "Round 2" below.

**Results**
- **Confirmation folds** (seeds 45–49): macro-F1 **0.6436 ± 0.0374**, accuracy 0.7421. Per-class F1 is 0.844 (Not Addressed), 0.415 (Partially Compliant), 0.672 (Compliant).
- **Training folds** (seeds 42–44): 0.6311 ± 0.0262.
- **Vs the previous final model:** +0.030, 23/25 folds, p = 1.8e-5.

**Configuration**
- **Features:** E5+8+all-rules (825 features), built from `"query: " + clause_text` with the same feature functions as evaluation.
- **Classifier:** `src/training/ordinal.py:OrdinalLRCV`. Two cumulative binary LogRegs, P(y > Not Addressed) and P(y > Partially Compliant), on RobustScaled features. Balanced class weights.
- **C:** chosen by 3-fold inner CV on all 826 rows: **C = 0.1**. Inner macro-F1 by C: 0.1 → 0.655, 0.3 → 0.644, 1 → 0.629, 3 → 0.636.
- **Thresholds:** `tune_thresholds_inner_cv`, with the full model, including its C selection, refit on each inner split: **τ_compliant = 0.60, τ_partial = 0.40**. τ_partial is at the upper edge of the existing grid (0.20–0.40); the grid was not widened.
- **Probabilities:** no probability calibration. Scaling lives inside the model, so there is no `scaler.joblib`.
- **Configs:** `feature_config.json` lists the 8 true actionability feature names and the label-free input format. `model_config.json` records the C search, threshold tuning and the evaluation results above. `/api/metrics` serves those numbers from `model_config.json`.

**Checks**
- `OrdinalLRCV` reproduces the evaluated r1 model exactly on a training fold (same C, probability difference 0.0).
- `run_round2_variants.py --stage confirm --verify` reproduces the confirmation run.
- 20-clause smoke test: training vs inference features max |diff| 4.2e-7, probabilities 7.3e-7; classifier vs API 20/20 identical.

**Caveat:** on the training folds the one-step LogReg with C chosen by inner CV scores about the same (0.6291 vs 0.6311). The gain is mainly from stronger regularisation; a benefit from the ordinal structure is not established.

*Superseded:* the earlier 2026-10-27 retrain (one-step LogReg, C = 1, τ = 0.50 / 0.25; confirmation 0.6132 with thresholds, 0.6106 argmax).

---


This table documents every variant tested during the experiment series, including those excluded due to methodological flaws.

## Table 1: All Experiments (Training Folds 42-44) — ⛔ INVALID (label leakage)

| Variant | Features | Folds | Grouped | Label Source | Macro-F1 | Status | Exclusion Reason |
|---------|----------|-------|---------|--------------|----------|--------|------------------|
| **Baselines** |
| ⛔ INVALID (label leakage): baseline_v1 | MiniLM + 8 | 42-44 | ❌ No | N/A | 0.7599 ± 0.0466 | ❌ Excluded | No near-duplicate grouping |
| ⛔ INVALID (label leakage): baseline_v1_grouped | MiniLM + 8 | 42-44 | ✅ Yes (60) | N/A | 0.7473 ± 0.0354 | ❌ Excluded | Encoder-specific grouping |
| ⛔ INVALID (label leakage): baseline_v2_grouped | MiniLM + 8 | 42-44 | ✅ Yes (61) | N/A | 0.7469 ± 0.0373 | ❌ Excluded | Different corpus (v2 labels) |
| ⛔ INVALID (label leakage): **fixed_baseline_v1** | **MiniLM + 8** | **42-44** | **✅ Yes (60)** | **N/A** | **0.7473 ± 0.0354** | **✅ Valid** | **Reference baseline** |
| **Encoder Swap Experiments** |
| ⛔ INVALID (label leakage): variant_A_bge | BGE + 8 | 42-44 | ✅ Yes (67) | N/A | 0.7638 ± 0.0519 | ❌ Excluded | Encoder-specific grouping |
| ⛔ INVALID (label leakage): variant_A_e5 (1st run) | E5 + 8 | 42-44 | ⚠️ Fallback (2) | N/A | 0.5878 ± 0.3081 | ❌ Excluded | No grouping (plain KFold) |
| ⛔ INVALID (label leakage): variant_A_e5 (2nd run) | E5 + 8 | 42-44 | ⚠️ Fallback (2) | N/A | 0.7772 ± 0.0384 | ❌ Excluded | No grouping (plain KFold) |
| ⛔ INVALID (label leakage): variant_A_inlegal | InLegalBERT + 8 | 42-44 | ⚠️ Fallback (1) | N/A | 0.7307 ± 0.0405 | ❌ Excluded | No grouping (plain KFold) |
| ⛔ INVALID (label leakage): **fixed_A_e5** | **E5 + 8** | **42-44** | **✅ Yes (60)** | **N/A** | **0.7703 ± 0.0314** | **✅ Valid** | **Improved baseline** |
| **Clause-Rule Pairing (LEAKY)** |
| ⛔ INVALID (label leakage): variant_B_bge | BGE pairing + 8 | 42-44 | ✅ Yes (67) | LLM output | 0.7846 ± 0.0461 | ❌ Excluded | Label leakage + encoder grouping |
| ⛔ INVALID (label leakage): variant_B_e5 | E5 pairing + 8 | 42-44 | ⚠️ Fallback (2) | LLM output | 0.8050 ± 0.0372 | ❌ Excluded | Label leakage + no grouping |
| ⛔ INVALID (label leakage): variant_B_inlegal | InLegalBERT pairing + 8 | 42-44 | ⚠️ Fallback (1) | LLM output | 0.7954 ± 0.0430 | ❌ Excluded | Label leakage + no grouping |
| ⛔ INVALID (label leakage): fixed_B_e5_ref | E5 pairing + 8 | 42-44 | ✅ Yes (60) | LLM output | 0.7923 ± 0.0317 | ⚠️ Reference | Label leakage (reference only) |
| **Label-Independent Pairing** |
| ⛔ INVALID (label leakage): fixed_B2a_e5 | E5 pairing (cosine) + 8 | 42-44 | ✅ Yes (60) | Cosine sim | 0.7663 ± 0.0274 | ✅ Valid | No improvement vs baseline |
| **Two-Stage Classification** |
| ⛔ INVALID (label leakage): variant_C_e5 | Two-stage (B pairing) | 42-44 | ⚠️ Fallback (2) | LLM output | 0.7803 ± 0.0346 | ❌ Excluded | Label leakage + no grouping |
| ⛔ INVALID (label leakage): fixed_C_e5 | Two-stage (B2a pairing) | 42-44 | ✅ Yes (60) | Cosine sim | 0.7675 ± 0.0339 | ✅ Valid | No improvement vs baseline |
| **Incremental Features** |
| ⛔ INVALID (label leakage): **incr_1_all_rules** | **E5 + 8 + all-rules (49)** | **42-44** | **✅ Yes (60)** | **E5 embeddings** | **0.8118 ± 0.0315** | **✅ Valid** | **BEST on training folds** |
| ⛔ INVALID (label leakage): incr_2_rubric | E5 + 8 + rubric (14) | 42-44 | ✅ Yes (60) | N/A | 0.7797 ± 0.0461 | ✅ Valid | High variance |
| ⛔ INVALID (label leakage): incr_3_combined | E5 + 8 + all-rules + rubric | 42-44 | ✅ Yes (60) | E5 embeddings | 0.7779 ± 0.0417 | ✅ Valid | Worse than all-rules alone |
| ⛔ INVALID (label leakage): incr_4_tuned | Combined + tuning | 42-44 | ✅ Yes (60) | E5 embeddings | 0.7795 ± 0.0339 | ✅ Valid | No improvement |
| ⛔ INVALID (label leakage): incr_5_ensemble_v3 | E5 + 8 + all-rules (5-seed) | 42-44 | ✅ Yes (60) | E5 embeddings | 0.8118 ± 0.0315 | ✅ Valid | Identical to single classifier |

---

## Table 2: Confirmation Experiments (Unseen Folds 45-49, CORRECTED) — ⛔ INVALID (label leakage)

**Note:** Initial confirmation experiments used wrong grouping (815 groups instead of 639). Results below are from corrected experiments with IDENTICAL grouping.

| Variant | Features | Folds | Grouped | Macro-F1 | Δ vs Training | Status |
|---------|----------|-------|---------|----------|---------------|--------|
| ⛔ INVALID (label leakage): ~~confirm_baseline_v1~~ | ~~MiniLM + 8~~ | ~~45-49~~ | ~~❌ Wrong (815)~~ | ~~0.7323 ± 0.0381~~ | ~~-1.5%~~ | ❌ Invalid grouping |
| ⛔ INVALID (label leakage): **confirm_baseline_v1_corrected** | **MiniLM + 8** | **45-49** | **✅ Yes (639)** | **0.7347 ± 0.0519** | **-1.3%** | **✅ Valid** |
| ⛔ INVALID (label leakage): ~~confirm_fixed_A_e5~~ | ~~E5 + 8~~ | ~~45-49~~ | ~~❌ Wrong (815)~~ | ~~0.7975 ± 0.0294~~ | ~~+2.7%~~ | ❌ Invalid grouping |
| ⛔ INVALID (label leakage): **confirm_fixed_A_e5_corrected** | **E5 + 8** | **45-49** | **✅ Yes (639)** | **0.8160 ± 0.0340** | **+4.6%** | **✅ Valid** |
| ⛔ INVALID (label leakage): ~~confirm_all_rules~~ | ~~E5 + 8 + all-rules~~ | ~~45-49~~ | ~~❌ Wrong (815)~~ | ~~0.7977 ± 0.0284~~ | ~~-1.4%~~ | ❌ Invalid grouping |
| ⛔ INVALID (label leakage): **confirm_all_rules_corrected** | **E5 + 8 + all-rules** | **45-49** | **✅ Yes (639)** | **0.8132 ± 0.0303** | **+0.1%** | ⚠️ **OVERFITTING** |

**Key Insight (CORRECTED):** 
- All-rules similarity: **Training +4.1%** → **Confirmation -0.3%** (NEGATIVE!)
- Overfitting to training fold structure **CONFIRMED** with correct grouping
- E5+8 alone shows better generalization: **Training +2.3%** → **Confirmation +4.6%**

## Table 3: Pooled Results (All 8 Seeds, 40 Folds) — ⛔ INVALID (label leakage)

| Variant | Features | N Folds | Macro-F1 | All-Rules Benefit | Status |
|---------|----------|---------|----------|-------------------|--------|
| ⛔ INVALID (label leakage): Baseline | MiniLM + 8 | 40 | 0.7394 ± 0.0468 | N/A | ✅ Valid |
| ⛔ INVALID (label leakage): E5+8 | E5 + 8 | 40 | 0.7989 ± 0.0398 | N/A | ✅ Valid |
| ⛔ INVALID (label leakage): E5+8+all-rules | E5 + 8 + all-rules | 40 | 0.8127 ± 0.0308 | **+0.0138 (+1.7%)** | ⚠️ Marginal

---

## Exclusion Criteria Summary — ⛔ INVALID (label leakage)

### 1. No Near-Duplicate Grouping — ⛔ INVALID (label leakage)
**Count:** 5 variants
- baseline_v1, variant_A_e5 (both runs), variant_A_inlegal, variant_B_e5, variant_B_inlegal, variant_C_e5
- **Problem:** Plain StratifiedKFold allows near-duplicates in train and test, inflating scores
- **Evidence:** E5 without grouping: 0.7772, with grouping: 0.7703 (-0.7%)

### 2. Encoder-Specific Grouping — ⛔ INVALID (label leakage)
**Count:** 3 variants
- baseline_v1_grouped, variant_A_bge, variant_B_bge
- **Problem:** Different encoders produce different groups, making comparisons invalid
- **Evidence:** BGE (67 groups) vs E5 (2 groups) vs MiniLM (60 groups)

### 3. Label Leakage — ⛔ INVALID (label leakage)
**Count:** 4 variants (1 kept as reference)
- variant_B_bge, variant_B_e5, variant_B_inlegal, variant_C_e5, fixed_B_e5_ref
- **Problem:** `best_match_taxonomy_id` column comes from LLM output during labeling
- **Evidence:** 100% correlation between verdict and rule ID presence
  - Not Addressed: 586/586 have rule='NONE'
  - Partially Compliant: 143/143 have real rules
  - Compliant: 97/97 have real rules
- **Impact:** True pairing improvement ~2% (not 8% as initially measured)

### 4. Wrong Corpus Version — ⛔ INVALID (label leakage)
**Count:** 1 variant
- baseline_v2_grouped
- **Problem:** Uses labeled_clauses_v2.csv (different labels) instead of bootstrap corpus

### 5. No Performance Improvement or Overfitting — ⛔ INVALID (label leakage)
**Count:** 4 variants
- fixed_B2a_e5, fixed_C_e5, incr_2-5 (documented but not recommended)
- **Reason:** Marginal or no F1 gain, higher complexity or variance

### 6. High Variance Across Fold Splits — ⛔ INVALID (label leakage)
**Count:** 0 variants causing exclusion
- **Note:** Initial analysis incorrectly flagged all-rules as overfitting
- **Correction:** All-rules shows **consistent** performance (0.8118 → 0.8132, +0.14%)
- **Actual finding:** E5-only shows high variance (0.7703 → 0.8160, +4.57%)
- **Interpretation:** All-rules stabilizes performance, E5-only is more sensitive to fold composition
- **Conclusion:** All-rules generalizes well, included in final model

---

## Valid Variants for Paper — ⛔ INVALID (label leakage)

### Recommended for Paper Citation: — ⛔ INVALID (label leakage)
1. **fixed_baseline_v1** - MiniLM + 8 structured (0.7473 ± 0.0354, training seeds 42-44)
2. **fixed_A_e5** - E5 + 8 structured (0.7703 ± 0.0314, training; 0.8160 ± 0.0340, confirmation)
3. **confirm_all_rules_corrected** - E5 + 8 + all-rules on confirmation (0.8132 ± 0.0303, seeds 45-49)
4. **Pooled E5+8+all-rules** - FINAL MODEL across all 8 seeds (0.8127 ± 0.0308, 40 folds)

### Reportable but Not Recommended for Final Model: — ⛔ INVALID (label leakage)
- fixed_B2a_e5 (0.7663) - Label-independent pairing, no gain
- fixed_C_e5 (0.7675) - Two-stage, no gain
- incr_2_rubric (0.7797) - Linguistic features, high variance
- incr_3_combined (0.7779) - Combined features, worse than all-rules alone
- incr_4_tuned (0.7795) - Hyperparameter tuning, no gain
- incr_5_ensemble (0.8118) - Seed ensemble, identical to single classifier

### Final Model Selected: — ⛔ INVALID (label leakage)
- **incr_1_all_rules** (0.8118 training, 0.8132 confirmation, 0.8127 pooled) - **SELECTED** ✅
- Consistent performance across fold splits
- Best pooled performance across 40 folds
- Interpretable rule-level features

---

## Performance Delta Summary — ⛔ INVALID (label leakage)

| Comparison | Δ Macro-F1 | Interpretation |
|------------|------------|----------------|
| E5 vs MiniLM (training, 42-44) | +2.3% | Encoder upgrade helps |
| E5 vs MiniLM (confirmation, 45-49) | +8.1% | Variable performance |
| E5 vs MiniLM (pooled, 40 folds) | +5.9% | Consistent improvement |
| All-rules vs E5 (training, 42-44) | +4.1% | Strong improvement |
| ⛔ INVALID (label leakage): All-rules vs E5 (confirmation, 45-49) | -0.3% | Tie within 1 std (0.8132 vs 0.8160) |
| **All-rules vs E5 (pooled, 40 folds)** | **+1.4%** | **Consistent benefit** ✅ |
| ⛔ INVALID (label leakage): All-rules consistency | 0.8118 → 0.8132 | Stable across fold splits (+0.14%) |
| ⛔ INVALID (label leakage): E5-only variance | 0.7703 → 0.8160 | Large variance across splits (+4.57%) |
| Leaky pairing vs E5 | +2.2% | True pairing benefit (after removing leakage) |
| Label-independent pairing vs E5 | -0.4% | No benefit |

---

## Methodological Lessons — ⛔ INVALID (label leakage)

1. **Fixed folds are essential** for reliable comparison
   - Without: E5 scores 0.7772 (2 groups, no protection)
   - With: E5 scores 0.7703 (60 groups, proper protection)

2. **Label leakage can inflate scores by 6%**
   - Leaky pairing: 0.7923
   - Clean pairing: 0.7663
   - True improvement: 2.2% (not 8%)

3. **All-rules similarity shows CONSISTENT generalization (CONFIRMED)**
   - All-rules: Training 0.8118, Confirmation 0.8132 (+0.14%, **consistent**)
   - E5-only: Training 0.7703, Confirmation 0.8160 (+4.57%, **variable**)
   - Pooled: All-rules 0.8127 vs E5-only 0.7989 (+1.4% benefit)
   - **Conclusion:** All-rules features generalize well and provide stable +1.4% improvement
   - Both models tie within 1 std on unbiased confirmation folds (0.8132 ± 0.030 vs 0.8160 ± 0.034)
   - **Final model:** E5+8+all-rules (825 features) for best pooled performance

4. **Simpler models often win**
   - E5 + 8 (776 features) ≈ E5 + 8 + all-rules (825 features) on unseen data
   - Combined features (839) < all-rules alone (825)

---

## Recommended Final Model — ⛔ INVALID (label leakage)

**For paper citation:**
> ⛔ INVALID (label leakage): numbers in this block came from leaky inputs.
```
Configuration: E5 clause embeddings + 8 structured + all-rules similarity
Features: 825 dimensions (768 E5 + 8 structured + 49 all-rules)
Classifier: LogisticRegression(C=1.0, class_weight='balanced')
CV Method: Fixed folds with near-duplicate grouping (MiniLM, cosine ≥ 0.90)
           639 groups (60 multi-member), identical for all seeds

Training Performance (seeds 42-44, 15 folds): 
  Macro-F1: 0.8118 ± 0.0315
  Accuracy: 0.9137 ± 0.0219

Confirmation Performance (seeds 45-49, 25 folds):
  Macro-F1: 0.8132 ± 0.0303
  Accuracy: 0.9166 ± 0.0192

Pooled Performance (seeds 42-49, 40 folds):
  Macro-F1: 0.8127 ± 0.0308
  Accuracy: 0.9155 ± 0.0203

Improvement over baseline (MiniLM+8):
  Confirmation:  +7.9% (+0.0785 F1) 
  Pooled:        +7.3% (+0.0733 F1)

Improvement over E5-only (E5+8):
  Pooled:        +1.4% (+0.0138 F1)
  Consistency:   Stable (0.8118 → 0.8132, +0.14%)
```

**Rationale:**
- ✅ No label leakage (uses E5 rule embeddings, not LLM output)
- ✅ Proper near-duplicate grouping (639 groups, verified identical across all seeds)
- ✅ **Consistent generalization** (0.8118 → 0.8132, stable across fold splits)
- ✅ **Best pooled performance** (0.8127 vs 0.7989 for E5-only, +1.4%)
- ✅ Ties with E5-only on confirmation within 1 std (0.8132 ± 0.030 vs 0.8160 ± 0.034)
- ✅ All-rules features provide interpretable rule-level signals for compliance assessment
- ✅ More feature-rich for production edge cases (825 vs 776 features) (-0.3% on confirmation)
