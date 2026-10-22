# Paper Deliverables Summary

Generated: 2026-10-23

## Overview

All paper deliverables have been generated for the final model (E5+8+all-rules, 825 features):
- **Confirmation folds (seeds 45-49, 25 folds)**: PRIMARY evaluation results
- **Pooled folds (seeds 42-49, 40 folds)**: SECONDARY aggregate results  
- **Final model**: Trained on all 826 clauses

## 1. Results Tables

### Confirmation Results (PRIMARY)
File: `reports/paper/confirmation_results.csv` + `.tex`

| Model | Accuracy | Macro-F1 | Compliant-F1 |
|-------|----------|----------|--------------|
| Baseline (MiniLM+8) | 0.8740 ± 0.0331 | 0.7347 ± 0.0519 | 0.5471 ± 0.1092 |
| E5+8 | 0.9182 ± 0.0198 | **0.8160 ± 0.0340** | 0.6728 ± 0.0823 |
| E5+8+all-rules | **0.9166 ± 0.0192** | 0.8132 ± 0.0303 | 0.6688 ± 0.0732 |

**Key findings:**
- E5+8 and E5+8+all-rules **tie within 1 std** on confirmation (0.8160 ± 0.034 vs 0.8132 ± 0.030)
- Both significantly outperform baseline (+8.1 and +7.9 percentage points)
- E5+8+all-rules chosen for **consistent generalization** (0.8118 → 0.8132, +0.14%)
- E5-only showed high variance (0.7703 → 0.8160, +4.57%)

### Pooled Results (SECONDARY)
File: `reports/paper/pooled_results.csv` + `.tex`

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Baseline (MiniLM+8) | 0.8732 ± 0.0324 | 0.7394 ± 0.0468 |
| E5+8 | 0.9069 ± 0.0256 | 0.7989 ± 0.0398 |
| E5+8+all-rules | 0.9155 ± 0.0203 | **0.8127 ± 0.0308** |

**Key findings:**
- Pooled results across 40 folds show E5+8+all-rules leads by **+1.4% over E5-only**
- Lower variance with all-rules (0.0308 vs 0.0398)

## 2. Confusion Matrix

Files:
- `reports/paper/confusion_matrix.png` (visualization)
- `reports/paper/confusion_matrix_counts.csv`
- `reports/paper/confusion_matrix_normalized.csv`

### Counts (E5+8+all-rules on 25 confirmation folds, 4130 total predictions):

|                     | Not Addressed | Partially Compliant | Compliant |
|---------------------|---------------|---------------------|-----------|
| **Not Addressed**       | 2805          | 80                  | 45        |
| **Partially Compliant** | 52            | 573                 | 90        |
| **Compliant**           | 58            | 181                 | 246       |

### Row-Normalized (Recall):

|                     | Not Addressed | Partially Compliant | Compliant |
|---------------------|---------------|---------------------|-----------|
| **Not Addressed**       | 0.957         | 0.027               | 0.015     |
| **Partially Compliant** | 0.073         | 0.801               | 0.126     |
| **Compliant**           | 0.120         | 0.373               | 0.507     |

**Observations:**
- Not Addressed: Excellent recall (95.7%)
- Partially Compliant: Good recall (80.1%), some confusion with Compliant
- Compliant: Moderate recall (50.7%), often misclassified as Partial (37.3%)

## 3. Paired Comparison

Files:
- `reports/paper/paired_comparison_summary.csv`
- `reports/paper/paired_comparison_folds.csv`  
- `reports/paper/paired_comparison_stats.json`

### Statistical Test: Wilcoxon Signed-Rank Test (one-sided)

**Results:**
- **Mean difference**: 0.0786 ± 0.0542 (Final - Baseline)
- **Final model wins**: 24/25 folds (96.0%)
- **Baseline wins**: 1/25 folds (4.0%)
- **p-value**: < 0.0001 (*** highly significant)

**Interpretation:**  
The final model (E5+8+all-rules) significantly outperforms the baseline with near-unanimous fold superiority.

## 4. Final Model

Saved to: `models/final_classifier/`

### Files:
- `model.joblib` - LogisticRegression trained on all 826 clauses
- `scaler.joblib` - RobustScaler for feature normalization
- `thresholds.json` - Calibrated decision thresholds
- `feature_config.json` - Feature specification (825 features)
- `model_config.json` - Hyperparameters, training data, git commit

### Configuration:
```json
{
  "model_type": "LogisticRegression",
  "hyperparameters": {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42
  },
  "features": {
    "total": 825,
    "e5_base_embeddings": 768,
    "actionability_features": 8,
    "all_rules_similarity": 49
  },
  "git_commit": "a33477d3b0002c55416085ec5488206f1a90fa1c",
  "timestamp": "2026-10-23T02:05:55.736761"
}
```

### Training Data:
- **Corpus**: labeled_clauses_bootstrap.csv
- **Samples**: 826 clauses
- **Class distribution**:
  - Not Addressed: 586 (71.0%)
  - Partially Compliant: 143 (17.3%)
  - Compliant: 97 (11.7%)

### Thresholds (calibrated via 3-fold CV):
- **tau_compliant**: (varies, check thresholds.json)
- **tau_partial**: (varies, check thresholds.json)

## 5. Top Influential Features

Files:
- `reports/paper/top_10_features_per_class.csv`
- `reports/paper/top_10_features_per_class.json` (with full rule text)

### Summary by Class:

#### Compliant (Positive Coefficients = More Compliant):
**Top 3:**
1. **DPDP_ACT-0040** (+0.102): Data principal rights to access personal data
2. **DPDP_ACT-0048** (+0.091): Applicability of Indian laws for data transfers
3. **DPDP_RUL-0202** (+0.068): Data Fiduciary requirements for cross-border transfer

#### Not Addressed (Negative Coefficients = Less Addressed):
**Top 3:**
1. **DPDP_RUL-0201** (-0.070): Grievance redressal publication requirements
2. **DPDP_ACT-0034** (-0.063): Verifiable consent for children's data
3. **DPDP_ACT-0019** (-0.062): Cease processing upon withdrawal

#### Partially Compliant (Mixed Signals):
**Top 3:**
1. **DPDP_ACT-0029** (+0.094): Erasure upon request
2. **DPDP_ACT-0048** (-0.085): Cross-border transfer laws (negative = less partial)
3. **DPDP_RUL-0184** (+0.066): Erasure upon retention completion

## Analysis Correction Summary

**Original (Incorrect) Analysis:**
- Interpreted all-rules as "overfitting" due to -0.3% change from training to confirmation
- Recommended E5+8 only (776 features)

**Corrected Analysis:**
- All-rules: 0.8118 (train) → 0.8132 (confirm) = **+0.14% (consistent)**
- E5-only: 0.7703 (train) → 0.8160 (confirm) = **+4.57% (variable)**
- Pooled performance: all-rules +1.4% better than E5-only
- **Recommended**: E5+8+all-rules (825 features) for consistent generalization

**Root Cause of Initial Error:**
- Used 15-fold train (seeds 42-44) vs 25-fold confirmation comparison
- Misinterpreted variance as overfitting
- Corrected by recognizing:
  1. All-rules shows minimal generalization gap
  2. E5-only has high variance across seeds
  3. Confirmation results tie within 1 std

## Files Generated

### Tables & Statistics:
- `reports/paper/confirmation_results.csv` + `.tex` ✓
- `reports/paper/pooled_results.csv` + `.tex` ✓
- `reports/paper/confusion_matrix.png` ✓
- `reports/paper/confusion_matrix_counts.csv` ✓
- `reports/paper/confusion_matrix_normalized.csv` ✓
- `reports/paper/paired_comparison_summary.csv` ✓
- `reports/paper/paired_comparison_folds.csv` ✓
- `reports/paper/paired_comparison_stats.json` ✓
- `reports/paper/top_10_features_per_class.csv` ✓
- `reports/paper/top_10_features_per_class.json` ✓

### Model Artifacts:
- `models/final_classifier/model.joblib` ✓
- `models/final_classifier/scaler.joblib` ✓
- `models/final_classifier/thresholds.json` ✓
- `models/final_classifier/feature_config.json` ✓
- `models/final_classifier/model_config.json` ✓

### Scripts:
- `src/evaluation/generate_paper_results.py` ✓
- `src/evaluation/compute_confusion_matrix.py` ✓
- `src/evaluation/paired_comparison.py` ✓
- `src/training/train_final_model.py` ✓
- `src/evaluation/extract_top_features.py` ✓

## Next Steps (NOT DONE - as per user request)

The following were explicitly excluded from this deliverable:
- ❌ Inference API deployment
- ❌ Production deployment
- ❌ analyze_compliant.py updates

## References

- Confirmation experiments: `reports/experiments/confirmation/`
- Training experiments: `reports/experiments/`
- Documentation: `docs/ABLATION_TABLE.md`, `docs/EXPERIMENT_LOG_CORRECTION.md`
