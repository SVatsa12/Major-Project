# DPDP Compliance Classifier

Automated classification of privacy policy clauses for compliance with India's Digital Personal Data Protection (DPDP) Act, 2023.

## Paper Results

> **⛔ Earlier results are INVALID (label leakage, found 2026-10-27).** All numbers published before commit `359f281`, including the former headline **0.8132 ± 0.030** macro-F1, used model inputs that contained the LLM-chosen rule (`build_context_texts`). That rule is `NONE` exactly when the label is "Not Addressed", so the label leaked into every input. Do not cite them. The valid label-free results are in `docs/EXPERIMENT_LOG.md` ("Label-free results") and `reports/paper/`.

This repository contains the code and data to reproduce the results from our paper. With label-free inputs (`"query: " + clause_text`), the final model achieves **0.6436 ± 0.0374 macro-F1** on confirmation folds (seeds 45-49). It was selected on the training folds only and run once on the confirmation folds. It uses:
- **E5-base embeddings** (768 dims)
- **8 actionability features** (structured)
- **All-rules similarity features** (49 dims)
- **Total**: 825 features
- **Ordinal classifier**: two cumulative logistic regressions, P(y > Not Addressed) and P(y > Partially Compliant), with C chosen by inner CV. Decision thresholds are tuned by inner CV.

**Tag**: ~~`paper-results-v1`~~ — ⛔ INVALID (label leakage): produced the old leaky results; do not use it to reproduce results  
**Paper Deliverables**: `reports/paper/` — Tables, confusion matrix, comparisons, feature analysis

## Quick Start

### Prerequisites

```bash
# Python 3.12+
pip install -r requirements.txt
```

### Reproduce Paper Results

```bash
# 1. Final-model selection: variants on the training folds (seeds 42-44), best by training macro-F1 only
python src/evaluation/build_backtranslation_cache.py      # paraphrases for the augmentation variant
python src/evaluation/run_round2_variants.py --stage train

# 2. Final model on the confirmation folds (seeds 45-49, 25 folds) - PRIMARY (runs once)
python src/evaluation/run_round2_variants.py --stage confirm

# 3. Feature-set comparison (MiniLM / E5 / E5+all-rules, one-step LogReg) on confirmation folds
python src/evaluation/run_confirmation_experiments.py \
  --variant all_rules \
  --tag confirm_all_rules_labelfree

# 4. Generate paper tables (CSV + LaTeX) for the feature-set comparison
python src/evaluation/generate_paper_results.py

# 5. Confusion matrix (PNG + CSV) for the E5+8+all-rules one-step model
python src/evaluation/compute_confusion_matrix.py

# 6. Paired statistical comparison (Wilcoxon test), E5+8+all-rules vs baseline
python src/evaluation/paired_comparison.py

# 7. Top influential all-rules features of the final ordinal model
python src/evaluation/extract_top_features.py
```

### Train Final Model

```bash
# Train on all 826 clauses and save to models/final_classifier/
python src/training/train_final_model.py
```

## Project Structure

```
.
├── corpus/                    # Training data
│   ├── labeled_clauses_bootstrap.csv   # 826 labeled clauses
│   └── dpdp_taxonomy_final.json        # 38 DPDP rules taxonomy
├── datasets/
│   └── folds/                 # Fold definitions for reproducibility
│       ├── fixed_folds_v1.json         # Training folds (seeds 42-44)
│       ├── confirm_folds_v1.json       # Confirmation folds (seeds 45-49)
│       └── groups_v1.json              # 639 near-duplicate groups
├── models/
│   └── final_classifier/      # Final trained model (paper)
│       ├── model.joblib                # Ordinal LogReg (src/training/ordinal.py), RobustScaler inside
│       ├── thresholds.json             # Inner-CV-tuned decision thresholds
│       ├── model_config.json           # Training configuration
│       └── feature_config.json         # Feature specification (825 dims)
├── reports/
│   ├── experiments/           # Experiment results (JSON/CSV)
│   │   ├── fixed_baseline_v1.json      # Baseline (MiniLM+8)
│   │   ├── fixed_A_e5.json             # E5-base+8
│   │   ├── incr_1_all_rules.json       # E5-base+8+all-rules
│   │   └── confirmation/               # Confirmation fold results
│   └── paper/                 # Paper deliverables ★
│       ├── confirmation_results.{csv,tex}       # PRIMARY results table
│       ├── pooled_results.{csv,tex}             # SECONDARY results table
│       ├── confusion_matrix.png                 # Confusion matrix viz
│       ├── confusion_matrix_{counts,normalized}.csv
│       ├── paired_comparison_*.{csv,json}       # Statistical tests
│       └── top_10_features_per_class.{csv,json} # Feature analysis
├── docs/                      # Documentation
│   ├── EXPERIMENT_LOG.md               # Complete experiment history
│   ├── EXPERIMENT_LOG_CORRECTION.md    # Analysis correction details
│   ├── ABLATION_TABLE.md               # Feature ablation study
│   ├── GROUPING_DISCREPANCY_ANALYSIS.md # Fold grouping verification
│   └── PAPER_DELIVERABLES_SUMMARY.md   # Paper results summary
├── src/                       # Core pipeline code
│   ├── extraction/            # Clause extraction
│   │   ├── extract_government_clauses.py   # Extract .gov policy clauses
│   │   └── extract_social_media_clauses.py # Extract social media clauses
│   ├── filtering/             # Clause filtering
│   │   ├── filter_government_clauses.py    # Filter .gov clauses
│   │   └── filter_social_media_clauses.py  # Filter social media clauses
│   ├── taxonomy/              # Taxonomy generation
│   │   ├── generate_taxonomy.py        # Generate DPDP taxonomy from Act
│   │   ├── triage_by_act.py            # Parse Act sections
│   │   └── consolidate_taxonomy.py     # Merge taxonomy variants
│   ├── labeling/              # Clause labeling
│   │   ├── label_policy_clauses.py     # Label clauses (LLM-assisted)
│   │   └── export_label_disagreements.py # Export disagreements for review
│   ├── evaluation/            # Evaluation & metrics
│   │   ├── evaluate_5fold_cv.py        # CV evaluation framework
│   │   ├── run_feature_experiments.py  # Feature ablation runner
│   │   ├── run_confirmation_experiments.py # Confirmation validation
│   │   ├── feature_variants.py         # Feature extraction variants
│   │   ├── incremental_features.py     # All-rules similarity features
│   │   ├── generate_paper_results.py   # Paper tables generator
│   │   ├── compute_confusion_matrix.py # Confusion matrix generator
│   │   ├── paired_comparison.py        # Statistical comparison
│   │   ├── extract_top_features.py     # Feature importance analysis
│   │   ├── generate_report_card.py     # Generate compliance report card
│   │   ├── generate_fixed_folds.py     # Training fold generator
│   │   └── generate_confirm_folds.py   # Confirmation fold generator
│   ├── training/              # Model training
│   │   ├── train_classifier.py         # Core training logic
│   │   ├── train_final_model.py        # Final model trainer
│   │   ├── feature_extraction.py       # Actionability features
│   │   └── two_stage_classifier.py     # Two-stage variant
│   └── api/
│       └── server.py                   # Flask API (not deployed)
├── scripts/                   # One-off/utility scripts
│   ├── analysis/              # Analysis utilities
│   ├── data_prep/             # Data preparation (align, recalibrate, split, ingest)
│   ├── debug/                 # Debug utilities (diagnose.py)
│   ├── reporting/             # Report generation (comparison tables, etc.)
│   └── validation/            # Validation scripts (audit_rule_leakage.py)
├── frontend/                  # Demo UI (React + Flask)
│   └── README.md              # Frontend documentation
└── archive_stale_scripts/     # Historical experiments (archived)
```

## Verification

To verify the label-free results (run from `taxonomy-v2` at or after `359f281`; the `paper-results-v1` tag is ⛔ INVALID, label leakage):

```bash
# 1. Label-leakage guard tests
python -m unittest discover -s tests -v

# 2. Reproduce the final model's confirmation run (seeds 45-49) into verify_* files
python src/evaluation/run_round2_variants.py --stage confirm --verify

# 3. Compare macro-F1 (should match to 4 decimals: 0.6436)
python -c "
import json
with open('reports/experiments/round2/verify_confirm_r1_ordinal.json') as f:
    data = json.load(f)
    print(f'Macro-F1: {data[\"aggregates\"][\"macro_f1_mean\"]:.4f}')
"

# 4. Training/inference parity of the saved model (start the API first: python src/api/server.py)
python test_smoke_parity.py
```

Expected output: `Macro-F1: 0.6436` (the old `0.8132` is ⛔ INVALID, label leakage)

## Features

### Embedding Features (768 dims)
- **Encoder**: intfloat/e5-base-v2
- **Input**: `"query: " + clause_text` only (label-free; `src/training/model_inputs.py`, guarded by `tests/test_label_leakage.py`)
- **Pooling**: Mean pooling, L2-normalized

### Structured Features (8 dims)
1. `hedge_density` - (hedge word count) / (clause word count)
2. `action_verb_count` - Raw count of specific action verbs (delete, erase, notify, provide, restrict, rectify, withdraw, inform, escalate, terminate, purge)
3. `has_ui_path` - Binary: regex for settings navigation patterns
4. `has_contact_mechanism` - Binary: regex for contact info (email, grievance officer, DPO)
5. `has_timeline_sla` - Binary: regex for time commitments (N days/hours, promptly)
6. `commitment_ratio` - C / (C + H + ε), where C = commitment verbs (will, shall, must, guarantee, ensure) and H = hedge words
7. `clause_length_norm` - min(1.0, word_count / 100.0)
8. `parental_verification_marker` - Binary: regex for child/parental consent markers

### All-Rules Similarity Features (49 dims)
- 38 cosine similarities: clause vs each DPDP rule
- 11 category max similarities: max similarity per rule category
- Enables interpretability: which statutory rules influence predictions

## Model Architecture

```
Input: Privacy policy clause text
  ↓
Label-free input: "query: " + clause_text
  ↓
E5-base Encoder (768 dims) + Actionability Features (8 dims) + All-Rules Similarity (49 dims)
  ↓
Feature Scaling: RobustScaler (fit on training rows, inside the model)
  ↓
Ordinal Logistic Regression: P(y > Not Addressed), P(y > Partially Compliant)
  (two cumulative binary LogRegs, balanced class weights, C by 3-fold inner CV; final model C = 0.1)
  ↓
Decision thresholds: 3-fold inner CV (final model τ_compliant = 0.60, τ_partial = 0.40)
  ↓
Output: {Not Addressed, Partially Compliant, Compliant}
```

## Performance

### Final model: confirmation folds (seeds 45-49, 25 folds) - PRIMARY

The final model was selected by training-fold macro-F1 only, from ordinal, context, augmentation and ensemble variants, and run once on the confirmation folds (`reports/experiments/round2/`). The previous final model, a one-step LogReg with C = 1 and inner-CV thresholds, is shown for comparison.

| Metric | Previous final model (one-step LogReg) | **Final model (ordinal)** |
|--------|----------------------------------------|---------------------------|
| Macro-F1 | 0.6132 ± 0.0344 | **0.6436 ± 0.0374** |
| Accuracy | 0.7282 ± 0.0321 | 0.7421 ± 0.0305 |
| Not Addressed P / R / F1 | 0.863 / 0.823 / 0.841 | 0.867 / 0.824 / 0.844 |
| Partially Compliant P / R / F1 | 0.355 / 0.438 / 0.387 | 0.378 / 0.470 / 0.415 |
| Compliant P / R / F1 | 0.646 / 0.601 / 0.612 | 0.710 / 0.659 / 0.672 |

- **Vs the previous final model:** +0.030 macro-F1, 23/25 folds, Wilcoxon p = 1.8e-5.
- **Training folds:** 0.6311 ± 0.0262 vs 0.6134 (12/15 folds, p = 0.018).
- **Caveat:** on the training folds most of the gain is also achieved by simply choosing C by inner CV in the one-step model. A benefit from the ordinal structure itself is not established (see `docs/EXPERIMENT_LOG.md`).

### Confusion Matrix (Recall), final model, confirmation folds

|                     | Not Addressed | Partially Compliant | Compliant |
|---------------------|---------------|---------------------|-----------|
| **Not Addressed**       | **82.6%**         | 15.9%               | 1.5%      |
| **Partially Compliant** | 38.9%             | **47.3%**           | 13.8%     |
| **Compliant**           | 17.1%             | 17.9%               | **64.9%** |

### Feature-set comparison: confirmation folds (seeds 45-49, 25 folds)

One-step LogReg (C = 1, argmax), label-free inputs (`*_labelfree` runs). Full per-class P/R/F1 and the 40-fold pooled table: `reports/paper/` and `docs/EXPERIMENT_LOG.md`.

| Metric | Baseline (MiniLM+8) | E5+8 | E5+8+all-rules |
|--------|---------------------|------|------------------------|
| Accuracy | 0.7096 ± 0.0308 | 0.7391 ± 0.0327 | 0.7381 ± 0.0290 |
| Macro-F1 | 0.5662 ± 0.0425 | 0.6127 ± 0.0353 | 0.6106 ± 0.0323 |
| Partially Compliant F1 | 0.3781 ± 0.0986 | 0.3721 ± 0.0844 | 0.3718 ± 0.0845 |
| Compliant F1 | 0.4841 ± 0.0809 | 0.6161 ± 0.0711 | 0.6102 ± 0.0722 |

E5+8 and E5+8+all-rules are tied (within 1 std).

**Statistical Test**: Wilcoxon signed-rank (E5+8+all-rules vs Baseline)
- E5+8+all-rules wins: 22/25 folds (88.0%)
- Mean difference: +0.0444 ± 0.0404
- p-value: < 0.001

Recall of the E5+8+all-rules one-step model (`reports/paper/confusion_matrix.png`): 84.7% / 39.2% / 61.0%.

> ⛔ INVALID (label leakage): the previously published table (macro-F1 0.7347 / 0.8160 / 0.8132, 24/25 fold wins, 95.7% / 80.1% / 50.7% recall) came from leaky inputs.

## Citation

If you use this code or data, please cite:

```bibtex
[Citation to be added upon publication]
```

## License

[License to be determined]

## Contact

For questions or issues, please open a GitHub issue or contact [contact information].

---

**Last Updated**: 2026-10-23  
**Repository Status**: Paper results finalized and reproducible  
**Demo UI**: See `frontend/README.md` for the web interface demo
