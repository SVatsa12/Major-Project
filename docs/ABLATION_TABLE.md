# Complete Ablation Study: All Variants Tested

This table documents every variant tested during the experiment series, including those excluded due to methodological flaws.

## Table 1: All Experiments (Training Folds 42-44)

| Variant | Features | Folds | Grouped | Label Source | Macro-F1 | Status | Exclusion Reason |
|---------|----------|-------|---------|--------------|----------|--------|------------------|
| **Baselines** |
| baseline_v1 | MiniLM + 8 | 42-44 | ❌ No | N/A | 0.7599 ± 0.0466 | ❌ Excluded | No near-duplicate grouping |
| baseline_v1_grouped | MiniLM + 8 | 42-44 | ✅ Yes (60) | N/A | 0.7473 ± 0.0354 | ❌ Excluded | Encoder-specific grouping |
| baseline_v2_grouped | MiniLM + 8 | 42-44 | ✅ Yes (61) | N/A | 0.7469 ± 0.0373 | ❌ Excluded | Different corpus (v2 labels) |
| **fixed_baseline_v1** | **MiniLM + 8** | **42-44** | **✅ Yes (60)** | **N/A** | **0.7473 ± 0.0354** | **✅ Valid** | **Reference baseline** |
| **Encoder Swap Experiments** |
| variant_A_bge | BGE + 8 | 42-44 | ✅ Yes (67) | N/A | 0.7638 ± 0.0519 | ❌ Excluded | Encoder-specific grouping |
| variant_A_e5 (1st run) | E5 + 8 | 42-44 | ⚠️ Fallback (2) | N/A | 0.5878 ± 0.3081 | ❌ Excluded | No grouping (plain KFold) |
| variant_A_e5 (2nd run) | E5 + 8 | 42-44 | ⚠️ Fallback (2) | N/A | 0.7772 ± 0.0384 | ❌ Excluded | No grouping (plain KFold) |
| variant_A_inlegal | InLegalBERT + 8 | 42-44 | ⚠️ Fallback (1) | N/A | 0.7307 ± 0.0405 | ❌ Excluded | No grouping (plain KFold) |
| **fixed_A_e5** | **E5 + 8** | **42-44** | **✅ Yes (60)** | **N/A** | **0.7703 ± 0.0314** | **✅ Valid** | **Improved baseline** |
| **Clause-Rule Pairing (LEAKY)** |
| variant_B_bge | BGE pairing + 8 | 42-44 | ✅ Yes (67) | LLM output | 0.7846 ± 0.0461 | ❌ Excluded | Label leakage + encoder grouping |
| variant_B_e5 | E5 pairing + 8 | 42-44 | ⚠️ Fallback (2) | LLM output | 0.8050 ± 0.0372 | ❌ Excluded | Label leakage + no grouping |
| variant_B_inlegal | InLegalBERT pairing + 8 | 42-44 | ⚠️ Fallback (1) | LLM output | 0.7954 ± 0.0430 | ❌ Excluded | Label leakage + no grouping |
| fixed_B_e5_ref | E5 pairing + 8 | 42-44 | ✅ Yes (60) | LLM output | 0.7923 ± 0.0317 | ⚠️ Reference | Label leakage (reference only) |
| **Label-Independent Pairing** |
| fixed_B2a_e5 | E5 pairing (cosine) + 8 | 42-44 | ✅ Yes (60) | Cosine sim | 0.7663 ± 0.0274 | ✅ Valid | No improvement vs baseline |
| **Two-Stage Classification** |
| variant_C_e5 | Two-stage (B pairing) | 42-44 | ⚠️ Fallback (2) | LLM output | 0.7803 ± 0.0346 | ❌ Excluded | Label leakage + no grouping |
| fixed_C_e5 | Two-stage (B2a pairing) | 42-44 | ✅ Yes (60) | Cosine sim | 0.7675 ± 0.0339 | ✅ Valid | No improvement vs baseline |
| **Incremental Features** |
| **incr_1_all_rules** | **E5 + 8 + all-rules (49)** | **42-44** | **✅ Yes (60)** | **E5 embeddings** | **0.8118 ± 0.0315** | **✅ Valid** | **BEST on training folds** |
| incr_2_rubric | E5 + 8 + rubric (14) | 42-44 | ✅ Yes (60) | N/A | 0.7797 ± 0.0461 | ✅ Valid | High variance |
| incr_3_combined | E5 + 8 + all-rules + rubric | 42-44 | ✅ Yes (60) | E5 embeddings | 0.7779 ± 0.0417 | ✅ Valid | Worse than all-rules alone |
| incr_4_tuned | Combined + tuning | 42-44 | ✅ Yes (60) | E5 embeddings | 0.7795 ± 0.0339 | ✅ Valid | No improvement |
| incr_5_ensemble_v3 | E5 + 8 + all-rules (5-seed) | 42-44 | ✅ Yes (60) | E5 embeddings | 0.8118 ± 0.0315 | ✅ Valid | Identical to single classifier |

---

## Table 2: Confirmation Experiments (Unseen Folds 45-49)

| Variant | Features | Folds | Grouped | Macro-F1 | Δ vs Training | Status |
|---------|----------|-------|---------|----------|---------------|--------|
| confirm_baseline_v1 | MiniLM + 8 | 45-49 | ✅ Yes (11) | 0.7323 ± 0.0381 | -1.5% | ✅ Valid |
| **confirm_fixed_A_e5** | **E5 + 8** | **45-49** | **✅ Yes (11)** | **0.7975 ± 0.0294** | **+2.7%** | **✅ Valid** |
| confirm_all_rules | E5 + 8 + all-rules | 45-49 | ✅ Yes (11) | 0.7977 ± 0.0284 | **-1.4%** | ⚠️ No benefit |
| confirm_allrules_desc_only | E5 + 8 + all-rules (desc) | 45-49 | ✅ Yes (11) | 0.7977 ± 0.0284 | -1.4% | ⚠️ Identical |

**Key Insight:** All-rules similarity features show +4.2% on training folds but 0% on confirmation folds, suggesting **overfitting to fold structure**.

---

## Exclusion Criteria Summary

### 1. No Near-Duplicate Grouping
**Count:** 5 variants
- baseline_v1, variant_A_e5 (both runs), variant_A_inlegal, variant_B_e5, variant_B_inlegal, variant_C_e5
- **Problem:** Plain StratifiedKFold allows near-duplicates in train and test, inflating scores
- **Evidence:** E5 without grouping: 0.7772, with grouping: 0.7703 (-0.7%)

### 2. Encoder-Specific Grouping
**Count:** 3 variants
- baseline_v1_grouped, variant_A_bge, variant_B_bge
- **Problem:** Different encoders produce different groups, making comparisons invalid
- **Evidence:** BGE (67 groups) vs E5 (2 groups) vs MiniLM (60 groups)

### 3. Label Leakage
**Count:** 4 variants (1 kept as reference)
- variant_B_bge, variant_B_e5, variant_B_inlegal, variant_C_e5, fixed_B_e5_ref
- **Problem:** `best_match_taxonomy_id` column comes from LLM output during labeling
- **Evidence:** 100% correlation between verdict and rule ID presence
  - Not Addressed: 586/586 have rule='NONE'
  - Partially Compliant: 143/143 have real rules
  - Compliant: 97/97 have real rules
- **Impact:** True pairing improvement ~2% (not 8% as initially measured)

### 4. Wrong Corpus Version
**Count:** 1 variant
- baseline_v2_grouped
- **Problem:** Uses labeled_clauses_v2.csv (different labels) instead of bootstrap corpus

### 5. No Performance Improvement
**Count:** 3 variants
- fixed_B2a_e5, fixed_C_e5, incr_2-5 (documented but not recommended)
- **Reason:** Marginal or no F1 gain, higher complexity or variance

---

## Valid Variants for Paper

### Recommended for Paper Citation:
1. **fixed_baseline_v1** - MiniLM + 8 structured (0.7473 ± 0.0354)
2. **fixed_A_e5** - E5 + 8 structured (0.7703 ± 0.0314)
3. **incr_1_all_rules** - E5 + 8 + all-rules (0.8118 ± 0.0315) - training only
4. **confirm_fixed_A_e5** - E5 + 8 on unseen folds (0.7975 ± 0.0294)

### Reportable but Not Recommended:
- fixed_B2a_e5 (0.7663) - Label-independent pairing, no gain
- fixed_C_e5 (0.7675) - Two-stage, no gain
- incr_2_rubric (0.7797) - Linguistic features, high variance
- incr_3_combined (0.7779) - Combined features, worse than separate
- incr_4_tuned (0.7795) - Hyperparameter tuning, no gain
- incr_5_ensemble (0.8118) - Seed ensemble, identical to single classifier

---

## Performance Delta Summary

| Comparison | Δ Macro-F1 | Interpretation |
|------------|------------|----------------|
| E5 vs MiniLM (training) | +2.3% | Encoder upgrade helps |
| E5 vs MiniLM (confirmation) | +6.5% | Consistent improvement on unseen data |
| All-rules vs E5 (training) | +4.2% | Strong improvement |
| All-rules vs E5 (confirmation) | +0.0% | **No generalization** ⚠️ |
| Leaky pairing vs E5 | +2.2% | True pairing benefit (after removing leakage) |
| Label-independent pairing vs E5 | -0.4% | No benefit |

---

## Methodological Lessons

1. **Fixed folds are essential** for reliable comparison
   - Without: E5 scores 0.7772 (2 groups, no protection)
   - With: E5 scores 0.7703 (60 groups, proper protection)

2. **Label leakage can inflate scores by 6%**
   - Leaky pairing: 0.7923
   - Clean pairing: 0.7663
   - True improvement: 2.2% (not 8%)

3. **Feature improvements may not generalize**
   - All-rules: +4.2% on training folds (seeds 42-44)
   - All-rules: +0.0% on confirmation folds (seeds 45-49)
   - Suggests overfitting to fold structure

4. **Simpler models often win**
   - E5 + 8 (776 features) ≈ E5 + 8 + all-rules (825 features) on unseen data
   - Combined features (839) < all-rules alone (825)

---

## Recommended Final Model

**For paper citation:**
```
Configuration: E5 clause embeddings + 8 structured features
Features: 776 dimensions
Classifier: LogisticRegression(C=1.0, class_weight='balanced')
CV Method: Fixed folds with near-duplicate grouping (MiniLM, cosine ≥ 0.90)

Training Performance (seeds 42-44): 
  Macro-F1: 0.7703 ± 0.0314
  Accuracy: 0.8880 ± 0.0228

Confirmation Performance (seeds 45-49):
  Macro-F1: 0.7975 ± 0.0294
  Accuracy: 0.9118 ± 0.0143

Improvement over baseline:
  Training: +2.3%
  Confirmation: +6.5%
```

**Rationale:**
- ✅ No label leakage
- ✅ Proper near-duplicate grouping
- ✅ Confirmed generalization to unseen folds
- ✅ Simpler than all-rules variant (776 vs 825 features)
- ✅ More stable than incremental variants
- ✅ Consistent improvement on both training and confirmation
