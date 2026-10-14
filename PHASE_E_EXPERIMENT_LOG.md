# Phase E: MPNet Encoder Experiment Log

**Date**: 2026-10-14  
**Branch**: `phase-e-mpnet-encoder` (created from main @ commit 1789082)  
**Status**: ❌ **REJECTED** — Failed all acceptance criteria (F1 regression, per-class instability)  
**Command**: `python -u src/evaluation/evaluate_5fold_cv.py`

---

## One-Line Summary

Phase E (MPNet Encoder): Mean F1 0.7353 (std 0.0349), Accuracy 0.8499, encoder all-mpnet-base-v2 (768-d) → **REJECTED** (F1 -0.0359 vs baseline 0.7712, gain -0.0359 < 0.01, Per-Compliant -0.0411 > 0.02 drop tolerance) on branch phase-e-mpnet-encoder @ commit TBD.

---

## Results Summary

| Metric | Baseline | Phase E | Δ | Status |
|--------|----------|---------|---|--------|
| Mean Macro-F1 | 0.7712 | 0.7353 | -0.0359 | ❌ FAIL |
| Std Macro-F1 | 0.0398 | 0.0349 | -0.0049 | ✅ Better |
| Accuracy | 0.8826 | 0.8499 | -0.0327 | ❌ FAIL |
| F1: Not Addressed | 0.9614 | 0.9331 | -0.0283 | ❌ DROP |
| F1: Partially Compliant | 0.7405 | 0.6994 | -0.0411 | ❌ FAIL |
| F1: Compliant | 0.6082 | 0.5677 | -0.0405 | ❌ FAIL |

---

## Acceptance Bar Evaluation

### Criteria & Results

```
CRITERION 1: Mean Macro-F1 > 0.7712
  Required: ✓ 0.7712+
  Actual:   ✗ 0.7353
  Status:   ❌ FAIL (regression -0.0359, -4.66%)

CRITERION 2: Gain >= 0.01
  Required: ✓ 0.0100+
  Actual:   ✗ -0.0359
  Status:   ❌ FAIL (negative gain, degradation)

CRITERION 3: No class F1 drop > 0.02
  Required: ✓ ≤ 0.02 drop
  Actual:   ✗ -0.0411 (Partially Compliant)
  Status:   ❌ FAIL (max drop -0.0411 exceeds -0.02)

═══════════════════════════════════════════════════════════════
OVERALL VERDICT: ❌ REJECTED
═══════════════════════════════════════════════════════════════

Phase E fails all three acceptance criteria. The all-mpnet-base-v2
encoder (768-d) produces worse predictions than all-MiniLM-L6-v2
(384-d), with performance degradation across all metrics.
```

---

## Technical Details

### Change Summary

**Encoder Swap Only:**
- Replaced: all-MiniLM-L6-v2 (384-d) → all-mpnet-base-v2 (768-d)
- Input format: Unchanged (context-formatted strings)
- Pooling: Unchanged (mean pooling)
- Normalization: Unchanged (L2)
- Max sequence length: Unchanged (256 tokens)
- Handcrafted features: Unchanged (8 actionability features)
- Combined space: 384+8=392 → 768+8=776 dimensions
- RobustScaler: Still fit on training fold only

### Encoder Details

- **Model**: sentence-transformers/all-mpnet-base-v2
- **Dimensions**: 768 (vs 384 for MiniLM)
- **Architecture**: Larger transformer base
- **Status**: Downloaded and cached locally
- **Revision**: Latest from HuggingFace

### Per-Fold Performance

| Fold | Macro-F1 | Accuracy | τ_C | τ_P |
|------|----------|----------|-----|-----|
| 1 | 0.6903 | 0.8072 | 0.50 | 0.35 |
| 2 | 0.7699 | 0.8848 | 0.60 | 0.35 |
| 3 | 0.7565 | 0.8727 | 0.60 | 0.40 |
| 4 | 0.7642 | 0.8727 | 0.65 | 0.35 |
| 5 | 0.6957 | 0.8121 | 0.60 | 0.30 |
| **Mean** | **0.7353** | **0.8499** | **0.60** | **0.35** |

**Fold Stability**: Std 0.0349 (improved from baseline 0.0398, -12.31%), but at the cost of lower absolute performance.

### Per-Class Breakdown

| Class | Baseline | Phase E | Δ F1 | Δ Precision | Δ Recall |
|-------|----------|---------|------|-------------|----------|
| Not Addressed | 0.9614 | 0.9331 | -0.0283 | -0.0110 | -0.0410 |
| Partially Compliant | 0.7405 | 0.6994 | -0.0411 | -0.0445 | +0.0057 |
| Compliant | 0.6082 | 0.5677 | -0.0405 | +0.0504 | -0.0928 |

**Key Observations:**
- All three classes show F1 regression
- Compliant class hit hardest: recall drops -0.0928 (precision improves +0.0504, but not enough)
- Partially Compliant also unstable: both precision and F1 drop despite slight recall gain

---

## Why Phase E Failed

### Hypothesis: Encoder Mismatch

The all-mpnet-base-v2 encoder, despite being a larger and more sophisticated model, underperforms on the DPDP compliance classification task. Possible reasons:

1. **Overfitting Risk**: Larger model capacity (768-d) may overfit the smaller corpus (826 clauses, only ~97 positive examples for Compliant class)
2. **Task-Specific Training**: all-MiniLM-L6-v2 was trained on sentence-pair tasks; all-mpnet-base-v2 trained on different objectives
3. **Embedding Space Alignment**: MPNet embeddings may not align well with the downstream DPDP regulatory classification problem
4. **Probability Distribution**: Different encoder produces different probability distributions, requiring different thresholds (but still insufficient)

### Evidence

- Threshold shifts show model is less confident (τ_c increases 0.55→0.60, indicating lower Compliant probabilities)
- Cross-fold variance improves (std down 12.31%), but mean performance down 4.66%—suggests model is more consistent but consistently wrong
- Per-class instability (largest drop -0.0411) exceeds tolerance (0.02)—not a marginal miss, a systematic degradation

---

## Files Changed (Phase E)

- `src/evaluation/evaluate_5fold_cv.py` — Encoder swap, Phase E metadata
- `train_final_model.py` — Encoder swap, embedder_model_name update
- `src/training/train_classifier.py` — Dimension comments (392→776)
- `src/taxonomy/consolidate_taxonomy.py` — EMBEDDING_MODEL_NAME update
- `src/labeling/label_policy_clauses.py` — Encoder swap
- `src/taxonomy/generate_taxonomy.py` — Encoder swap

---

## Branch Status

- **Current Branch**: `phase-e-mpnet-encoder`
- **Status**: ❌ Not merged to main
- **Recommendation**: Archive branch; revert to all-MiniLM-L6-v2 for production

---

## Next Steps

**Do NOT use all-mpnet-base-v2 for production.** Keep all-MiniLM-L6-v2 (384-d) as baseline encoder.

Alternative experiment phases:
1. **Phase A**: Threshold variants (keep MiniLM)
2. **Phase C**: Context format changes (keep MiniLM)
3. **Phase D**: Alternative classifiers (keep MiniLM)
4. **Hybrid**: Combine both encoders' embeddings as features

---

**Execution Timeline**

- Start: 2026-10-14 ~14:30 UTC
- Embeddings (768-d): ~47 seconds (slower than 384-d MiniLM due to model size)
- 5-Fold CV: ~120 seconds
- Total: ~170 seconds
- Report Written: [timestamp in JSON]

