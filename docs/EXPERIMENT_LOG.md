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

### ✅ Step 2: Rule Leakage Audit
- Created `src/evaluation/audit_rule_leakage.py`
- Documented leakage mechanism and evidence
- Confirmed gating candidates not available
- **Commit:** 23822af

---

## Remaining Steps

### ⏳ Step 3: Label-Independent Rule Selection
**Goal:** Implement variant B2a without using any LLM output

**Requirements:**
- B2a: Select rule by highest cosine similarity (E5 embeddings)
  - Compute clause embedding vs ALL 38 rule embeddings
  - Select max similarity rule
  - Features: [clause, rule, |diff|, prod, cosine] + top-3 similarities + 8 structured
- Add assertion: No LLM columns (verdict, best_match_taxonomy_id, justification) used as features
- Update feature_variants.py with new extractors

**Files to Create/Modify:**
- `src/evaluation/feature_variants.py` (add B2a extractor)
- Add assertion checks

### ⏳ Step 4: Re-run All Experiments with Fixed Folds
**Experiments to run:**
1. baseline_v1_grouped (MiniLM + 8 features)
2. variant_A_e5 (E5 clause only + 8 features)
3. variant_B_e5 (E5 pairing with LLM rule - for reference only)
4. variant_B2a_e5 (E5 pairing with label-independent rule)
5. variant_C_e5 (two-stage with B2a features)

**Requirements:**
- ALL must use fixed_folds_v1.json (no fallback)
- Seeds: 42, 43, 44
- Grouped: YES for all
- Output table: variant | rule source | grouped | acc | macro-F1 | macro-prec | C_F1 | PC_F1

**Files to Create/Modify:**
- Update `src/evaluation/run_feature_experiments.py` to load fixed folds
- Remove fallback logic (raise error if folds not found)
- Add B2a support

---

## Commands to Resume

### Step 3: Implement B2a
```bash
# Edit feature_variants.py to add label-independent rule selection
# Add assertion checks for no LLM columns
git add src/evaluation/feature_variants.py
git commit -m "Step 3: Implement label-independent rule selection (B2a)"
```

### Step 4: Re-run with Fixed Folds
```bash
# Update run_feature_experiments.py to use fixed folds
# Remove fallback logic

# Run experiments
python src/evaluation/run_feature_experiments.py --features baseline --tag fixed_baseline_v1 --seeds 42,43,44
python src/evaluation/run_feature_experiments.py --features encoder_e5 --tag fixed_A_e5 --seeds 42,43,44
python src/evaluation/run_feature_experiments.py --features pairing_e5 --tag fixed_B_e5_ref --seeds 42,43,44
python src/evaluation/run_feature_experiments.py --features pairing_e5_b2a --tag fixed_B2a_e5 --seeds 42,43,44
python src/evaluation/run_feature_experiments.py --features two_stage_e5_b2a --tag fixed_C_e5 --seeds 42,43,44

# Generate final table
python reports/experiments/generate_comparison_table.py

git add reports/experiments/
git commit -m "Step 4: Re-run all experiments with fixed folds"
```

---

## Expected Outcomes

### Hypothesis
- Variant B (LLM rule) results will **drop significantly** when using fixed folds
- Variant B2a (label-independent rule) will show TRUE performance without leakage
- Likely outcome: B2a < B (leaky) but still > baseline
- Real improvement should be ~2-3% F1, not the 5-8% seen with leakage

### Trustworthy Results Criteria
✅ Fixed folds used (no encoder-specific grouping)
✅ Near-duplicate protection applied (60 groups)
✅ No LLM output used as features
✅ Rule selection independent of labels

---

## Git Status
- Current branch: taxonomy-v2
- Last commit: 23822af (Step 2 audit)
- Uncommitted: docs/EXPERIMENT_LOG.md (this file)

---

**Updated:** 2026-10-23 (after Step 2 completion)
