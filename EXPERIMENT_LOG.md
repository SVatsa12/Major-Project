# DPDP Compliance Classifier - Experiment Log

## Phase B: Feature Union Experiment

**Date**: 2026-10-14  
**Branch**: `phase-b-feature-union` (created from main @ commit 1789082)  
**Status**: ❌ **REJECTED** — Failed acceptance bar (gain too low)  
**Command**: `python -u src/evaluation/evaluate_5fold_cv.py`

### Results Summary

| Metric | Baseline | Phase B | Δ | Status |
|--------|----------|---------|---|--------|
| Mean Macro-F1 | 0.7712 | 0.7732 | +0.0020 | ✅ PASS* |
| Gain (Δ F1) | — | +0.0020 | Required ≥ 0.01 | ❌ **FAIL** |
| Std Macro-F1 | 0.0398 | 0.0331 | -0.0067 | ✅ Better |
| Accuracy | 0.8826 | 0.8862 | +0.0036 | ✅ Better |
| Max Class F1 Drop | — | -0.003 | Required ≤ 0.02 | ✅ PASS |
| Features | 8 | 15 | +7 | N/A |

*Metric 1 passed (F1 > 0.7712) but insufficient gain fails overall acceptance.

### Acceptance Bar Evaluation

```
Criterion 1: Mean Macro-F1 > 0.7712       ✅ PASS   (0.7732 > 0.7712)
Criterion 2: Gain >= 0.01                 ❌ FAIL   (0.0020 < 0.01) ← BLOCKER
Criterion 3: No class F1 drop > 0.02      ✅ PASS   (max drop -0.003)

OVERALL VERDICT: ❌ REJECTED
```

### Features Added (Phase B)

**Extension of baseline 8-dim to 15-dim union:**

1. hedge_density (baseline)
2. action_verb_count (baseline)
3. has_ui_path (baseline)
4. has_contact_mechanism (baseline)
5. has_timeline_sla (baseline)
6. commitment_ratio (baseline)
7. clause_length_norm (baseline)
8. parental_verification_marker (baseline)
9. **hedge_phrase_count** (union) — Count-based lexicon
10. **action_phrase_count** (union) — Count-based lexicon
11. **requirement_count** (union) — Count-based lexicon
12. **commitment_phrase_count** (union) — Count-based lexicon
13. **mechanism_count** (union) — Count-based lexicon
14. **actionability_ratio** (union) — Derived ratio metric
15. **has_nav** (union) — Binary nav-specific indicator

**Duplicate Analysis**: No exact duplicates. All 15 features retained:
- `hedge_density` (ratio) vs `hedge_phrase_count` (count) → Different computation
- `action_verb_count` vs `action_phrase_count` → Different lexicon source
- `commitment_ratio` vs `commitment_phrase_count` → Different type (ratio vs count)
- `has_ui_path` vs `has_nav` → Different regex patterns

### Per-Class Metrics

| Class | Baseline F1 | Phase B F1 | Δ | Status |
|-------|-------------|-----------|---|--------|
| Not Addressed | 0.9614 | 0.9681 | +0.0067 | ✅ Improved |
| Partially Compliant | 0.7405 | 0.7375 | -0.0030 | ✅ Stable |
| Compliant | 0.6082 | 0.6127 | +0.0045 | ✅ Improved |

### Technical Details

- **Embedding**: all-MiniLM-L6-v2 (384-dim, L2-normalized)
- **Combined Space**: 384-dim embedding + 15-dim features = 399-dim
- **Scaler**: RobustScaler fit on training fold only, applied to feature block
- **Model**: Logistic Regression (C=1.0, balanced class weights, multinomial)
- **Threshold Tuning**: Baseline objective only
  - Maximize macro-F1 subject to Compliant precision ≥ 0.70
  - Inner stratified 3-fold CV on training partition
  - No Phase A variants (alternative threshold objectives)
- **Evaluation**: Stratified 5-fold cross-validation (leak-free)
- **Corpus**: 826 clauses (from `corpus/labeled_clauses_bootstrap.csv`)

### Threshold Results (Median)

- **Compliant threshold (τ_c)**: 0.60 (vs baseline 0.55, +0.05)
- **Partial threshold (τ_p)**: 0.35 (vs baseline 0.40, -0.05)

### Cross-Fold Stability

| Fold | Macro-F1 | Accuracy | Trend |
|------|----------|----------|-------|
| 1 | 0.7642 | 0.8855 | Baseline |
| 2 | 0.7527 | 0.8788 | Below |
| 3 | 0.7703 | 0.8848 | Above |
| 4 | 0.8367 | 0.9212 | **Peak** |
| 5 | 0.7423 | 0.8606 | Below |
| **Mean** | **0.7732** | **0.8862** | **Stable** |
| **Std** | **0.0331** | **0.0049** | **Improved** |

### Key Insights

1. **Marginal Improvement**: Feature union provides +0.26% F1 gain, insufficient for acceptance (need +1.3%)
2. **Improved Stability**: Std F1 reduced 16.83% (0.0398 → 0.0331), suggesting feature union reduces fold-to-fold variance
3. **No Regression**: All per-class metrics improved or held stable; no significant drops
4. **Threshold Shift**: Thresholds increased slightly, indicating slightly higher confidence requirements for positive classes
5. **Fold 4 Anomaly**: Fold 4 significantly outperforms (0.8367 F1, 0.9212 accuracy), possible class imbalance in split

### Files Changed

- `src/training/feature_extraction.py` — Extended to 15-dim union with count-based features
- `src/evaluation/evaluate_5fold_cv.py` — Phase B evaluation (baseline objective only)
- `reports/cross_validation_report.json` — New report with 15 features, Phase B metadata

### Recommendation

**Do NOT merge Phase B to main.** The feature union does not justify the expanded feature set:
- Gain of 0.26% falls well short of 1.0% acceptance threshold
- Improved stability is noteworthy but insufficient to offset marginal F1 improvement
- Consider exploring alternative directions:
  - **Phase A**: Threshold variants for better precision-recall trade-offs
  - **Phase C**: Context-formatted input for richer semantic representation
  - **Phase D**: Alternative classifiers (ensemble, SVM, gradient boosting)
  - **Phase E**: Feature selection/engineering (e.g., dimensionality reduction, interaction terms)

### Execution Timeline

- **Start**: 2026-10-14 11:30 UTC
- **Embeddings**: ~25 seconds (384-dim pre-computation)
- **5-Fold CV**: ~120 seconds (threshold calibration + evaluation)
- **Total**: ~150 seconds
- **Report Written**: 2026-10-14 12:40:37 UTC

---

**Branch Status**: Remains on `phase-b-feature-union`. No merge to main scheduled.

