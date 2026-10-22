# Experiment Log: Feature Variants & Leakage Audit

## Experiment History

### Initial Baseline (No Grouping)
| Tag | Data | Seeds | Grouped | Rule Source | Macro-F1 | Notes |
|-----|------|-------|---------|-------------|----------|-------|
| baseline_v1 | bootstrap | 42,43,44 | ❌ No | N/A | 0.7599 ± 0.0466 | No near-duplicate protection |

### Baseline with Grouping
| Tag | Data | Seeds | Grouped | n_groups | Rule Source | Macro-F1 | Notes |
|-----|------|-------|---------|----------|-------------|----------|-------|
| baseline_v1_grouped | bootstrap | 42,43,44 | ✅ Yes | 60 | N/A | 0.7473 ± 0.0354 | MiniLM grouping, proper |
| baseline_v2_grouped | v2 | 42,43,44 | ✅ Yes | 61 | N/A | 0.7469 ± 0.0373 | V2 labels |

### Variant A: Encoder Swap (Clause Only)
| Tag | Data | Seeds | Grouped | n_groups | Fallback? | Rule Source | Macro-F1 | Notes |
|-----|------|-------|---------|----------|-----------|-------------|----------|-------|
| variant_A_bge | bootstrap | 42,43,44 | ✅ Yes | 67 | No | N/A | 0.7638 ± 0.0519 | BGE encoder, proper grouping |
| variant_A_e5 | bootstrap | 42,43,44 | ⚠️ Fallback | 2 | **YES** | N/A | 0.7772 ± 0.0384 | **Used plain StratifiedKFold** |
| variant_A_inlegal | bootstrap | 42,43,44 | ⚠️ Fallback | 1 | **YES** | N/A | 0.7307 ± 0.0405 | **Used plain StratifiedKFold** |

### Variant B: Clause-Rule Pairing (⚠️ LEAKAGE SUSPECTED)
| Tag | Data | Seeds | Grouped | n_groups | Fallback? | Rule Source | Macro-F1 | Notes |
|-----|------|-------|---------|----------|-----------|-------------|----------|-------|
| variant_B_bge | bootstrap | 42,43,44 | ✅ Yes | 67 | No | **LLM output** | 0.7846 ± 0.0461 | ⚠️ LABEL LEAKAGE |
| variant_B_e5 | bootstrap | 42,43,44 | ⚠️ Fallback | 2 | **YES** | **LLM output** | 0.8050 ± 0.0372 | ⚠️ LEAKAGE + NO GROUPING |
| variant_B_inlegal | bootstrap | 42,43,44 | ⚠️ Fallback | 1 | **YES** | **LLM output** | 0.7954 ± 0.0430 | ⚠️ LEAKAGE + NO GROUPING |

### Variant C: Two-Stage Classification
| Tag | Data | Seeds | Grouped | n_groups | Fallback? | Rule Source | Macro-F1 | Notes |
|-----|------|-------|---------|----------|-----------|-------------|----------|-------|
| variant_C_e5 | bootstrap | 42,43,44 | ⚠️ Fallback | 2 | **YES** | **LLM output** | 0.7803 ± 0.0346 | ⚠️ LEAKAGE + NO GROUPING |

---

## Critical Audit Findings

### 1. Fixed Folds Issue (Step 1)
**Problem:** Experiments used encoder-specific grouping, causing inconsistent train/test splits
- BGE: 67 groups (proper StratifiedGroupKFold)
- E5: 2 groups → fallback to plain StratifiedKFold (no near-duplicate protection)
- InLegalBERT: 1 group → fallback to plain StratifiedKFold

**Solution:** Generated `datasets/folds/fixed_folds_v1.json`
- Uses all-MiniLM-L6-v2 embeddings for grouping (cosine ≥ 0.90)
- 639 total groups, 60 multi-member groups
- Fixed assignments for seeds 42, 43, 44
- All future experiments MUST use these fixed folds

**Status:** ✅ Completed and committed (commit 57fd6b7)

### 2. Label Leakage in Variant B (Step 2)
**Problem:** Rule IDs come from LLM labeling output, creating train-on-labels leakage

**Evidence:**
1. **Rule Source:** Column `best_match_taxonomy_id` from LLM output
2. **Correlation:** 
   - Not Addressed: 100% have rule='NONE' (586/586)
   - Partially Compliant: 100% have real rules (143/143), 27 distinct
   - Compliant: 100% have real rules (97/97), 28 distinct
3. **Mechanism:** LLM selected rule ID while assigning verdict
4. **Gating Candidates:** NOT saved in label_cache/*.json (cannot use for B2b)

**Impact:** All Variant B results (0.7846-0.8050 F1) are **INFLATED and UNTRUSTWORTHY**

**Status:** ✅ Audited and documented (commit 23822af)

---

## Completed Steps

### ✅ Step 1: Fixed Folds Generation
- Created `src/evaluation/generate_fixed_folds.py`
- Generated `datasets/folds/fixed_folds_v1.json`
- 639 groups (60 multi-member), 3 seeds × 5 folds
- **Commit:** 57fd6b7
- **Completed:** Updated `run_feature_experiments.py` to load fixed folds, removed fallback
- **Commit:** da20b28

### ✅ Step 2: Rule Leakage Audit
- Created `src/evaluation/audit_rule_leakage.py`
- Documented leakage mechanism and evidence
- Confirmed gating candidates not available
- **Commit:** 23822af

### ✅ Step 3: Label-Independent Rule Selection
- Implemented `extract_clause_rule_pairing_b2a()` in `feature_variants.py`
- Rule selection: highest cosine similarity (clause vs ALL 38 rules)
- Features: [clause, rule, |diff|, prod, cosine, top3_sims] + 8 structured
- NO LLM output used (assertion added)
- **Commit:** bb5a4e6

### ✅ Step 4: Re-run with Fixed Folds
- Ran 5 experiments: baseline, A_e5, B_e5 (leaky ref), B2a_e5, C_e5
- All use fixed folds (639 groups, proper grouping)
- **Commit:** 65a7083

---

## Final Results

### Experiments with Fixed Folds + Proper Grouping

| Variant | Rule Source | Grouped | Accuracy | Macro-F1 | Macro-Prec | Compliant F1 | Partial F1 |
|---------|-------------|---------|----------|----------|------------|--------------|------------|
| **fixed_B_e5_ref** | LLM output (LEAK) | ✅ YES | 0.9115 ± 0.0224 | **0.7923 ± 0.0317** | 0.8133 ± 0.0196 | 0.5994 | 0.7879 |
| **fixed_A_e5** | N/A | ✅ YES | 0.8880 ± 0.0228 | **0.7703 ± 0.0314** | 0.7887 ± 0.0059 | 0.5776 | 0.7625 |
| **fixed_C_e5** | B2a (independent) | ✅ YES | 0.8925 ± 0.0225 | **0.7675 ± 0.0339** | 0.7800 ± 0.0233 | 0.5696 | 0.7551 |
| **fixed_B2a_e5** | Label-independent | ✅ YES | 0.8889 ± 0.0217 | **0.7663 ± 0.0274** | 0.8031 ± 0.0242 | 0.5680 | 0.7643 |
| **fixed_baseline_v1** | N/A | ✅ YES | 0.8718 ± 0.0311 | **0.7473 ± 0.0354** | 0.7610 ± 0.0089 | 0.5710 | 0.7120 |

---

## Key Findings

### 1. Label Leakage Impact
- **Leaky B (LLM rule):** 0.7923 macro-F1
- **Clean B2a (independent rule):** 0.7663 macro-F1
- **Leakage inflation:** +2.6% absolute, ~3.4% relative

### 2. Missing Grouping Impact  
- **Previous B_e5 (no grouping):** 0.8050 macro-F1
- **Fixed B_e5 (with grouping):** 0.7923 macro-F1
- **Grouping correction:** -1.3% absolute

### 3. TRUE Improvements Over Baseline (0.7473)
- **E5 encoder (A):** +2.3% → 0.7703 (BEST single change)
- **B2a pairing (independent):** +1.9% → 0.7663
- **Two-stage (C):** +2.0% → 0.7675
- **Leaky B (reference only):** +4.5% → 0.7923 (UNTRUSTWORTHY)

### 4. Combined Effects
Previous inflated results had:
- Label leakage: ~+2.6%
- Missing grouping: ~+1.3%
- **Total inflation:** ~+3.9% over true performance

### 5. Trustworthy Conclusion
**Real clause-rule pairing improvement:** ~2% macro-F1
- Previous claim: 5-8% improvement
- Actual improvement: 1.9-2.0% improvement
- **Leakage + no grouping accounted for 3.9% of the 5.8% gain**

---

## Remaining Steps

None! All planned steps completed.

---

## Git Status
- Current branch: taxonomy-v2
- Last commit: 65a7083 (Step 4 completion)
- All steps 1-4 completed

---

**Updated:** 2026-10-23 (after Step 4 completion - ALL STEPS DONE)

## Incremental Feature Experiments (Starting from fixed_A_e5)

**Baseline:** fixed_A_e5 = 0.7703 ± 0.0314
- Features: E5 clause embeddings (768 dims) + 8 structured features
- Fixed folds (seeds 42,43,44), proper near-duplicate grouping
- Total: 776 features

### Experiment 1: All-Rules Similarity ✅ BEST
**Tag:** incr_1_all_rules
**Description:** Add similarity to ALL 38 taxonomy rules
**Features Added:** 
- 38 individual rule cosine similarities
- 11 category-level max similarities
- Total: 49 new features → 825 total

**Results:**
- Macro-F1: **0.8118 ± 0.0315** (+4.2% over baseline)
- Accuracy: 0.9137 ± 0.0219
- Compliant F1: 0.6711 ± 0.0080
- Partially Compliant F1: 0.7749 ± 0.0042

**Status:** ✅ **BEST SINGLE IMPROVEMENT**

---

### Experiment 2: Rubric Features
**Tag:** incr_2_rubric
**Description:** Add linguistic compliance signals
**Features Added:** 14 rubric features
- Time: specific_time, vague_time (counts + binary flags)
- Modals: strong/weak modal counts, strength ratio
- Hedging: hedge count and density
- Actions: user_action counts
- Structure: numeric/procedure counts

**Results:**
- Macro-F1: 0.7797 ± 0.0461 (+0.9% over baseline)
- Accuracy: 0.8868 ± 0.0403
- Compliant F1: 0.6281 ± 0.0103
- Partially Compliant F1: 0.7397 ± 0.0042

**Status:** ⚠️ Minor improvement, higher variance

---

### Experiment 3: Combined (All-Rules + Rubric)
**Tag:** incr_3_combined
**Description:** Combine all-rules similarity + rubric features
**Features:** E5 (768) + structured (8) + all-rules (49) + rubric (14) = 839 total

**Results:**
- Macro-F1: 0.7779 ± 0.0417 (+0.8% over baseline)
- Accuracy: 0.8838 ± 0.0334
- Compliant F1: 0.6266 ± 0.0046
- Partially Compliant F1: 0.7448 ± 0.0023

**Status:** ❌ **WORSE than all-rules alone** (feature redundancy/noise)

---

### Experiment 4: Classifier Tuning
**Tag:** incr_4_tuned
**Description:** Try LogisticRegression (C=0.1,0.3,1.0,3.0,10.0), LinearSVC, LightGBM
**Features:** Combined features (E5 + structured + all-rules + rubric) = 839

**Results:**
- Macro-F1: 0.7795 ± 0.0339 (+0.9% over baseline)
- Accuracy: 0.8837 ± 0.0358
- Compliant F1: 0.6356 ± 0.0023
- Partially Compliant F1: 0.7426 ± 0.0042

**Status:** ❌ No improvement over default LogReg (C=1.0)

---

### Experiment 5: Seed Ensemble
**Tag:** incr_5_ensemble_v3
**Description:** Train 5 classifiers with different random seeds (42-46), average predicted probabilities
**Features:** E5 (768) + structured (8) + all-rules (49) = 825 (best feature set)

**Results:**
- Macro-F1: 0.8118 ± 0.0315 (IDENTICAL to single classifier)
- Accuracy: 0.9137 ± 0.0219
- Compliant F1: 0.6711 ± 0.0080
- Partially Compliant F1: 0.7749 ± 0.0042

**Status:** ⚠️ **NO IMPROVEMENT** - seed ensemble provides no benefit with fixed folds

---

## Final Comparison Table

| Tag | Features | Macro-F1 | Δ vs Baseline | Recommended |
|-----|----------|----------|---------------|-------------|
| fixed_A_e5 | E5 + 8 structured | 0.7703 ± 0.0314 | baseline | - |
| incr_1_all_rules | + all-rules sim | **0.8118 ± 0.0315** | **+4.2%** | ✅ **YES** |
| incr_2_rubric | + rubric | 0.7797 ± 0.0461 | +0.9% | ❌ No (high variance) |
| incr_3_combined | + all-rules + rubric | 0.7779 ± 0.0417 | +0.8% | ❌ No (worse than #1) |
| incr_4_tuned | combined + tuning | 0.7795 ± 0.0339 | +0.9% | ❌ No |
| incr_5_ensemble_v3 | all-rules + ensemble | 0.8118 ± 0.0315 | +4.2% | ⚠️ No benefit over #1 |

---

## Recommended Model

**Configuration:**
- Encoder: intfloat/e5-base-v2 (clause embeddings)
- Features: E5 embeddings (768) + structured (8) + all-rules similarity (49) = **825 features**
- Classifier: LogisticRegression (C=1.0, balanced class weights)
- Cross-validation: Fixed folds with near-duplicate grouping (MiniLM, cosine ≥ 0.90)

**Performance:**
- Macro-F1: 0.8118 ± 0.0315
- Accuracy: 0.9137 ± 0.0219
- Compliant F1: 0.6711
- Partially Compliant F1: 0.7749

**Why This Model:**
1. ✅ No label leakage (uses E5 similarity, not LLM rule assignments)
2. ✅ Proper near-duplicate grouping (fixed folds)
3. ✅ +4.2% F1 improvement over baseline
4. ✅ Lower variance than rubric features
5. ✅ Simpler than combined models (fewer features = better generalization)

---

## Next Steps

1. **Train final model** on all 826 clauses using recommended configuration
2. **Save model artifacts**: trained classifier, scaler, taxonomy embeddings
3. **Create inference pipeline** for new policy clauses
4. **Document feature extraction** for production deployment
