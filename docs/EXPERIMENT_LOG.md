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

### Label-free variants of the final model (2026-10-27, VALID)

**Script**: `python src/evaluation/run_honest_variants.py`. **Results**: `reports/experiments/confirmation/honest_*.json` (per-fold metrics and predictions, pooled confusion matrix) and `reports/experiments/honest_variants_summary.{csv,md}`.

Same confirmation folds (`confirm_folds_v1.json`, seeds 45–49, 25 folds), same labels, clause text only (no LLM columns). Each variant changes one thing against the current final model (E5+8+all-rules, LogReg C=1, balanced). Every model is a Pipeline, so the TF-IDF vectorizer and RobustScaler are refit on the training rows of each outer fold, each inner C-selection fold and each inner threshold-tuning fold. Thresholds use the existing `tune_thresholds_inner_cv`.

**1. Current final model: per-class report (25 folds, mean ± std)**

| Decision rule | Macro-F1 | NA P / R / F1 | Partial P / R / F1 | Compliant P / R / F1 |
|---|---|---|---|---|
| argmax (as reported, reproduces 0.6106) | 0.6106 ± 0.0323 | 0.855 / 0.846 / 0.850 | 0.367 / 0.386 / 0.372 | 0.619 / 0.619 / 0.610 |
| inner-CV thresholds (**v1, reference below**) | 0.6132 ± 0.0344 | 0.863 / 0.823 / 0.841 | 0.355 / 0.438 / 0.387 | 0.646 / 0.601 / 0.612 |

Pooled confusion matrix, v1 (rows = true, columns = predicted; row-normalised recall in brackets):

| | Not Addressed | Partially Compliant | Compliant |
|---|---|---|---|
| **Not Addressed** (2930) | **2413** (0.824) | 448 (0.153) | 69 (0.024) |
| **Partially Compliant** (715) | 299 (0.418) | **316** (0.442) | 100 (0.140) |
| **Compliant** (485) | 74 (0.153) | 124 (0.256) | **287** (0.592) |

**Variants (confirmation folds; Δ and two-sided paired Wilcoxon p are against v1 over the 25 folds)**

| Variant | Macro-F1 | NA F1 | Partial F1 | Compliant F1 | Δ vs v1 (wins/25) | p |
|---|---|---|---|---|---|---|
| ref: final model, argmax | 0.6106 ± 0.0323 | 0.8498 ± 0.0229 | 0.3718 ± 0.0845 | 0.6102 ± 0.0722 | −0.0026 (10) | 0.44 |
| v1: final model, inner-CV thresholds | 0.6132 ± 0.0344 | 0.8414 ± 0.0263 | 0.3865 ± 0.0817 | 0.6118 ± 0.0746 | — | — |
| v2: TF-IDF + LogReg alone | 0.6148 ± 0.0364 | 0.8304 ± 0.0436 | 0.3839 ± 0.0706 | 0.6302 ± 0.0694 | +0.0016 (12) | 0.81 |
| v3: TF-IDF + E5+8+all-rules | 0.6170 ± 0.0339 | 0.8423 ± 0.0252 | 0.3917 ± 0.0817 | 0.6170 ± 0.0755 | +0.0037 (10) | 0.11 |
| v4: C ∈ {0.1, 0.3, 1, 3} by inner CV | 0.6250 ± 0.0346 | 0.8385 ± 0.0291 | 0.4087 ± 0.0846 | 0.6279 ± 0.0702 | +0.0118 (17) | 0.012 |
| **v5: two-step on v4** | **0.6370 ± 0.0387** | 0.8598 ± 0.0257 | 0.4075 ± 0.0850 | 0.6439 ± 0.0742 | +0.0238 (19) | 0.002 |

v5 two-step details: step (a) Not Addressed vs Addressed macro-F1 **0.7658 ± 0.0359**; step (b) Partially Compliant vs Compliant on truly-Addressed test rows **0.7401 ± 0.0569**; combined 3-class macro-F1 **0.6370 ± 0.0387**. v5 vs v4: +0.0120, p = 0.071.

**Findings**
- TF-IDF alone matches the E5 model (0.6148 vs 0.6132). Stacking TF-IDF onto E5+8+all-rules adds +0.004, which is not significant (p = 0.11). Lexical features carry about the same signal as the embeddings.
- Stronger regularisation helps: inner CV picks C = 0.1 in 19 of 25 folds (0.3 ×4, 1 ×1, 3 ×1), giving +0.012 (17/25 wins, p = 0.012), mostly on Partially Compliant F1 (0.387 → 0.409).
- The two-step model is the best variant: +0.024 over v1 (19/25 wins, p = 0.002). Most of the gain is on Not Addressed and Compliant F1. Its improvement over the one-step v4 (+0.012, p = 0.07) is not significant.
- The bottleneck is unchanged: Partially Compliant F1 stays at about 0.39–0.41 in every variant, and 42% of Partially Compliant clauses are still predicted Not Addressed.
- Thresholds vs argmax for the current model: +0.003, not significant (p = 0.44).

**Caveats**
- Variant 5's base config (v4) was chosen as the best of v1–v4 by mean macro-F1 on these same confirmation folds, so the v5 figure carries a small selection bias. The training-fold check below does not reproduce a significant gain.
- The binary thresholds in v5 come from `tune_binary_threshold_inner_cv`, a binary analogue of `tune_thresholds_inner_cv` (same inner split, refit per inner fold, same best-inner-fold selection, τ ∈ [0.20, 0.80), no precision constraint). C is selected by mean inner-fold argmax macro-F1, with ties going to the smaller C.
- In v3, TF-IDF columns are L2-normalised and unscaled, while dense columns are RobustScaled. Under a single L2 penalty the two blocks are not equally weighted. This scaling was not tuned.
- Nothing was retrained: `models/final_classifier/` is still v1 (C = 1, one-step). *(Superseded: `models/final_classifier/` was retrained with r1_ordinal; see "Final model" above.)*

#### Training-fold check of v5 (2026-10-27)

**Command**: `python src/evaluation/run_honest_variants.py --folds datasets/folds/fixed_folds_v1.json --variants v1_final,v5_two_step --two-step-base v4_C_select --suffix train`  
**Results**: `reports/experiments/honest_v1_final_train.json`, `reports/experiments/honest_v5_two_step_train.json`, `reports/experiments/honest_variants_summary_train.{csv,md}`.

Same script settings as the confirmation run, on `fixed_folds_v1.json` (seeds 42–44, 15 folds). v5's base is fixed in advance to v4 (C ∈ {0.1, 0.3, 1, 3} by inner CV), so no variant selection happens on these folds.

| Variant (training folds, 15) | Macro-F1 | NA P / R / F1 | Partial P / R / F1 | Compliant P / R / F1 | Δ vs v1 (wins/15) | p |
|---|---|---|---|---|---|---|
| v1: final model, inner-CV thresholds | 0.6134 ± 0.0286 | 0.851 / 0.830 / 0.839 | 0.366 / 0.443 / 0.394 | 0.666 / 0.574 / 0.607 | — | — |
| v5: two-step on v4 | 0.6263 ± 0.0339 | 0.863 / 0.862 / 0.861 | 0.405 / 0.441 / 0.414 | 0.654 / 0.572 / 0.604 | +0.0129 (9 wins, 6 losses) | 0.25 |

v5 step (a), Not Addressed vs Addressed: macro-F1 0.7651 ± 0.0282. Step (b), Partially Compliant vs Compliant on truly-Addressed rows: 0.7200 ± 0.0580. C chosen across the 30 per-step selections: 0.1 ×17, 0.3 ×8, 3 ×5.

v1 pooled confusion matrix (training folds; rows = true, row-normalised recall in brackets): Not Addressed 1456 (0.828) / 267 / 35; Partially Compliant 187 / 190 (0.443) / 52; Compliant 57 / 68 / 166 (0.570).

**Conclusion**
- **v1 is stable across fold sets:** 0.6134 on training folds vs 0.6132 on confirmation folds.
- **v5's gain is about half as large and not significant:** +0.013 on training folds (9/15 wins, p = 0.25), against +0.024 (19/25, p = 0.002) on confirmation folds.
- **Where the gain comes from:** on both fold sets it is on Not Addressed and Partially Compliant F1. Compliant F1 is flat on training folds (0.607 → 0.604).
- **Pooled estimate:** over all 40 folds v5 − v1 = +0.0197 (28/40 wins, paired Wilcoxon p = 0.001). That pooled test includes the 25 confirmation folds that were used to choose v5's base, so it overstates the evidence.
- **Verdict:** on the independent 15 training folds alone the gain is not significant. A modest real improvement (about +0.01 to +0.02) is plausible but not established well enough to replace the final model on this evidence alone.

### Round 2: ordinal, context, augmentation, ensemble (2026-10-27, VALID)

**Protocol**
- **Inputs:** clause text and document structure only; no LLM columns.
- **Folds:** the same fold files. Every variant was run on the **training folds** (`fixed_folds_v1.json`, seeds 42–44, 15 folds) against v1. The best config was chosen by training-fold mean macro-F1 **only** and run **once** on the confirmation folds. The script refuses a second confirmation run.
- **Tuning:** inner CV only. Each model is an index-based estimator, so C selection, augmentation, scaler and classifier are all refit within every outer and inner training fold. Thresholds use the existing `tune_thresholds_inner_cv`.

**Commands**
- `python src/evaluation/build_backtranslation_cache.py`
- `python src/evaluation/run_round2_variants.py --stage train`
- `python src/evaluation/run_round2_variants.py --stage confirm`

**Results**
- `reports/experiments/round2/` (`train_*.json`, `train_summary.{csv,md}`, `selection.json`, `confirm_r1_ordinal.json`, `confirm_summary.{csv,md}`)
- Paraphrase cache: `reports/experiments/augmentation/backtranslation_en_de_en.json`

**Variants (one change each vs v1)**
- **r1 ordinal:** two cumulative binary LogRegs, P(y > NA) and P(y > Partial), on RobustScaled v1 features, with P(Partial) = P(>NA) − P(>Partial) and P(>Partial) capped at P(>NA). C ∈ {0.1, 0.3, 1, 3} by inner CV. `mord` was not installed, so I used this cumulative form.
- **r2 context:** v1 plus three things.
  - E5 of the clause's `section_heading` from `datasets/SocialMediaPolicies/social_media_clauses.csv`, or of the previous clause in the same document if there is no heading.
  - Mean E5 of the previous and next clause in that document.
  - Relative position in the document.
  - Clauses are located by exact normalised text within the same `source_document`, else by difflib ratio ≥ 0.90. This found 700 of 826 rows. The other 126 (55 from supplemental single-clause documents, 71 whose text was edited after extraction) use their own embedding and position 0.5, with no missingness flag.
- **r3 augmentation:** 2 back-translation paraphrases (en→de→en, `Helsinki-NLP/opus-mt-en-de` and `opus-mt-de-en`, beam 4) per Partially Compliant clause. They are cached by text hash and added only for Partially Compliant rows inside the rows a model is fit on: 208–250 extra rows per outer fold. Each outer fold asserts that no augmented row's source index and no source text belongs to the test fold.
  - Paraphrases are faithful; length ratio is always between 0.5 and 1.5. In 43 of 143 clauses both beams back-translate to the same sentence, so that paraphrase is added twice.
- **r4 ensemble:** mean probabilities of the best two eligible variants by training macro-F1, r1 + r3, with inner-CV thresholds.
- **ref C-select:** v1 with C ∈ {0.1, 0.3, 1, 3} by inner CV, reported only to separate r1's ordinal effect from its C tuning. It was not a candidate.

**Provenance check, fixed before any result was seen.** 55 corpus rows come from hand-added single-clause supplemental documents: 46 Compliant, 9 Partial, 0 Not Addressed, which is 47% of all Compliant labels. Rows without document context are 42% Compliant, against 6% for rows with context. So any feature exposing "no document context" is a label proxy. r2 was therefore eligible only if it also beat v1 on the 700 rows with real document context. Every variant's macro-F1 on those rows is reported in the "matched rows" column.

**Training folds (seeds 42–44, 15 folds)**: Δ, wins and two-sided paired Wilcoxon p are against v1.

| Variant | Macro-F1 | NA F1 | Partial F1 | Compliant F1 | Wins vs v1 /15 (Δ) | p | Matched-rows macro-F1 (Δ, p) |
|---|---|---|---|---|---|---|---|
| v1: final model | 0.6134 ± 0.0286 | 0.8390 ± 0.0276 | 0.3941 ± 0.0548 | 0.6070 ± 0.0625 | — | — | 0.5369 |
| ref: C-select multinomial (not a candidate) | 0.6291 ± 0.0271 | 0.8373 ± 0.0310 | 0.4200 ± 0.0575 | 0.6299 ± 0.0614 | 11 (+0.0157) | 0.011 | 0.5503 (+0.013, p = 0.022) |
| **r1: ordinal** | **0.6311 ± 0.0262** | 0.8477 ± 0.0240 | 0.4192 ± 0.0718 | 0.6265 ± 0.0653 | 12 (+0.0178) | 0.018 | 0.5336 (−0.003, p = 0.72) |
| r2: context | 0.6223 ± 0.0363 | 0.8563 ± 0.0260 | 0.3932 ± 0.0899 | 0.6175 ± 0.0764 | 8 (+0.0090) | 0.60 | 0.5084 (−0.028, p = 0.055) |
| r3: augmentation | 0.6096 ± 0.0275 | 0.8385 ± 0.0297 | 0.3886 ± 0.0581 | 0.6016 ± 0.0777 | 6 (−0.0038) | 0.56 | 0.5337 (−0.003, p = 0.64) |
| r4: ensemble r1 + r3 | 0.6204 ± 0.0329 | 0.8433 ± 0.0267 | 0.3992 ± 0.0721 | 0.6186 ± 0.0731 | 8 (+0.0070) | 0.39 | 0.5318 (−0.005, p = 0.64) |

**What the training folds show**
- **Context (r2) is ineligible:** its small overall gain reverses on rows with real document context (−0.028). The gain comes from the provenance proxy, not from document context.
- **Augmentation (r3) does not help:** Partially Compliant F1 even dips (0.394 → 0.389).
- **The ensemble (r4) inherits r3's weakness:** it lands between its members.
- **Ordinal (r1) is the best candidate, but mostly through C tuning:** C-tuned multinomial gets about the same (r1 − ref: +0.002, 9/15 wins, p = 0.60).
- **r1's training gain does not appear on rows with real document context** (−0.003), unlike C-tuned multinomial (+0.013, p = 0.022).

**Selection** (`reports/experiments/round2/selection.json`): candidates v1, r1, r3, r4 (r2 ineligible). Best by training-fold macro-F1: **r1_ordinal** (0.6311).

**Confirmation folds, r1_ordinal, run once (seeds 45–49, 25 folds)**

| | Macro-F1 | Accuracy | NA P / R / F1 | Partial P / R / F1 | Compliant P / R / F1 |
|---|---|---|---|---|---|
| v1 (from `honest_v1_final.json`) | 0.6132 ± 0.0344 | 0.7282 | 0.863 / 0.823 / 0.841 | 0.355 / 0.438 / 0.387 | 0.646 / 0.601 / 0.612 |
| **r1_ordinal** | **0.6436 ± 0.0374** | 0.7421 | 0.867 / 0.824 / 0.844 | 0.378 / 0.470 / 0.415 | 0.710 / 0.659 / 0.672 |

- **r1 vs v1:** +0.0304, 23 wins / 2 losses, Wilcoxon p = 1.8e-5. On matched rows: +0.020, 17/25 wins, p = 0.008.
- **C chosen:** 0.1 ×15, 0.3 ×8, 1 ×1, 3 ×1.
- **Pooled confusion matrix, r1:**
  - Not Addressed: 2420 / 465 / 45
  - Partially Compliant: 278 / 338 / 99
  - Compliant: 83 / 87 / 315
  - Recall 0.826 / 0.473 / 0.649. Partially Compliant → Not Addressed falls from 0.418 (v1) to 0.389.

**Interpretation**
- **The confirmation gain is significant:** +0.030 over v1, and larger than on training (+0.018). It appears on rows with real document context too.
- **But ordinal vs plain C tuning is inconsistent.** Comparing r1 with the earlier confirmation run of C-tuned multinomial (`honest_v4_C_select.json`, which selects C once per outer fold rather than nested in threshold tuning) gives +0.019, 19/25 wins, p = 0.006. On training the same comparison is +0.002, p = 0.60. The robust finding across both fold sets is "stronger regularisation (C ≈ 0.1–0.3) helps"; a specific benefit from the ordinal structure is not established.
- **Partially Compliant remains the bottleneck:** best F1 is 0.42.
- **No model was retrained:** `models/final_classifier/` is still v1. Adopting r1 (or C-tuned multinomial) would need a retrain plus predictor support for the cumulative model. *(Superseded: `models/final_classifier/` was retrained with r1_ordinal; see "Final model" above.)*

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
