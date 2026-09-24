# India Data Governance & Compliance Audit Engine: End-to-End Technical Report

**Project Title:** Multi-Act Statutory Compliance Audit & NLP Classification Framework

**Jurisdictional Scope:**
- Digital Personal Data Protection Act, 2023 (DPDP Act 2023)
- DPDP Rules 2025
- Information Technology Act, 2000 & Intermediary Guidelines (IT Act 2000)

**Target Entities Audited:** WhatsApp, Meta, Telegram, Snapchat, Google, YouTube

**Underlying Architecture:** Legal NLP · Local LLM Inference (`qwen3:8b`) · Transformer Classification (`law-ai/InLegalBERT`) · Vector Embedding Search (`all-MiniLM-L6-v2`)

---

## 1. Executive Architecture & Pipeline Workflow

The repository implements a closed-loop regulatory engineering pipeline:

1. **Extraction** — Ingests raw statutory PDFs/DOCX files and social-media privacy terms, parsing them into structured clauses.
2. **Filtering** — Applies non-destructive, recall-first regex and domain heuristics to identify candidate clauses for compliance auditing.
3. **Taxonomy Engineering** — Dynamically generates a normalized statutory taxonomy across Indian data protection statutes via LLM extraction, semantic similarity clustering, and legal consolidation.
4. **LLM-Guided Labeling** — Annotates social media policy clauses against the statutory taxonomy using candidate gating, keyword routing, and semantic embedding fallbacks.
5. **Model Training & Active Learning** — Stratifies the corpus into balanced splits, applies gold-standard human-in-the-loop promotions, and fine-tunes domain-adapted transformers (InLegalBERT).
6. **Regulatory Compliance Auditing** — Computes multi-act statutory coverage, clause quality scores, domain gap analysis, and platform rankings.

```
[Raw Legal Acts & Policies]
         |
         v
[1. Extraction Engine] ──────────> government_clauses.csv / social_media_clauses.csv
         |
         v
[2. Filtering & Triage] ────────> government_clauses_filtered.csv / social_media_clauses_for_labeling.csv
         |
         v
[3. Taxonomy Generation] ───────> dpdp_taxonomy_final.csv / dpdp_taxonomy_final.json (38 Rules)
         |
         v
[4. Bootstrap Labeling] ────────> labeled_clauses_bootstrap.csv (771 clauses)
         |
         v
[5. Split & Training] ──────────> train.csv / val.csv / test.csv --> InLegalBERT Checkpoints
         |
         v
[6. Evaluation & Audit] ────────> platform_overall_rankings.csv / platform_act_compliance_matrix.csv
```

---

## 2. Stage 1: Document Ingestion & Clause Extraction

### 2.1 Government Acts Extraction

**Executable Script:** `src/extraction/extract_government_clauses.py`

**Source Inputs:** `datasets/GovernmentActs/source/*.pdf`
(e.g., `DPDP_2023.pdf`, `DPDP_2025.pdf`, `AirportAuthority.pdf`, IT Act 2000 Gazette texts)

**Engineering Logic:**
- Uses `pypdf` / `pdfplumber` to extract page-by-page text with OCR fallback via `pytesseract` for scanned or low-density pages.
- Employs conservative header/footer regex striping to remove repeated gazette running headers, page numbers, and court stamps.
- Splits statutory text into atomic legal provisions using hierarchical delimiters: Sections, Subsections, Clauses, Sub-clauses, Explanations, and Provisos.

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `datasets/GovernmentActs/government_clauses.csv` | CSV | Primary tabular corpus of raw extracted statutory clauses |
| `datasets/GovernmentActs/government_clauses.json` | JSON | Structured records including page provenance, line offsets, and chapter metadata |
| `datasets/GovernmentActs/government_clause_duplicate_report.json` | JSON | Deduplication log tracking identical legal subsections |

---

### 2.2 Social Media Policies Extraction

**Executable Script:** `src/extraction/extract_social_media_clauses.py`

**Source Inputs:** `datasets/SocialMediaPolicies/source/`

**Engineering Logic:**
- Ingests terms of service and consumer privacy declarations.
- Splits documents by logical paragraphs, section headers, bullet lists, and sentence boundaries.
- Assigns deterministic unique identifiers (`clause_id`) structured as `S{index:06d}`.

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `datasets/SocialMediaPolicies/social_media_clauses.csv` | CSV | Complete raw extracted policy clauses |
| `datasets/SocialMediaPolicies/social_media_clauses.json` | JSON | Structured document-level clause mapping |

---

## 3. Stage 2: Filtering, Normalization & Compliance Triage

### 3.1 Government Clauses Filtering

**Executable Script:** `src/filtering/filter_government_clauses.py`

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `datasets/GovernmentActs/government_clauses_filtered.csv` | CSV | Curated statutory clauses advancing to taxonomy synthesis |
| `datasets/GovernmentActs/government_clauses_cleaned.csv` | CSV | Normalized baseline legal text |
| `datasets/GovernmentActs/government_filter_summary.json` | JSON | Retention counts, regex hit statistics, and exclusion audit |

---

### 3.2 Social Media Pre-Filtering & Candidate Flagging

**Executable Script:** `src/filtering/filter_social_media_clauses.py`

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `datasets/SocialMediaPolicies/social_media_clauses_flagged.csv` | CSV | Full policy corpus enriched with category flag metadata |
| `datasets/SocialMediaPolicies/social_media_clauses_for_labeling.csv` | CSV | 771 curated candidate clauses for LLM labeling |

---

## 4. Stage 3: Multi-Act Taxonomy Synthesis & Consolidation

### 4.1 Candidate Extraction via Local LLM

**Executable Script:** `src/taxonomy/generate_taxonomy.py`

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `corpus/taxonomy_generation/taxonomy_candidates_raw.json` | JSON | Direct generative outputs from Ollama batches |
| `corpus/taxonomy_generation/taxonomy_candidates_normalized.json` | JSON | Validated records matching the strict Pydantic JSON schema |
| `corpus/taxonomy_generation/taxonomy_batch_manifest.json` | JSON | Checkpointed batch execution state (327 batches) |
| `corpus/taxonomy_generation/taxonomy_unmapped_clauses.json` | JSON | 1,977 unmapped administrative/procedural clauses |
| `corpus/taxonomy_generation/taxonomy_invalid_model_records.json` | JSON | Malformed JSON records rejected by schema validation |

### 4.2 Act-Level Triage

**Executable Script:** `src/taxonomy/triage_by_act.py`

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `corpus/taxonomy_generation/triage_by_act/ALL_TRIAGED_CANDIDATES.csv` | CSV | Unified triaged requirement catalog |
| `corpus/taxonomy_generation/triage_by_act/ALL_TRIAGED_CANDIDATES.json` | JSON | Full triaged structure with review flags |
| `corpus/taxonomy_generation/triage_by_act/ACT_TARGET_SUMMARY.json` | JSON | Candidate distribution per statutory act |

### 4.3 Final Taxonomy Consolidation

**Executable Script:** `src/taxonomy/consolidate_taxonomy.py`

**Logic:** Applies semantic deduplication (cosine similarity threshold = 0.85), trims out-of-scope non-consumer mandates, and generates canonical taxonomy rules.

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `corpus/dpdp_taxonomy_final.csv` | CSV | Final 38-rule statutory compliance benchmark |
| `corpus/dpdp_taxonomy_final.json` | JSON | Authoritative taxonomy records with checkable tests and keywords |
| `corpus/taxonomy_generation/final_consolidation_v2/CONSOLIDATION_SUMMARY.json` | JSON | Audit log of merged/pruned requirements |

---

## 5. Stage 4: Bootstrap Annotation & Dynamic Candidate Gating

### 5.1 Dynamic Gating & Labeling Engine

**Executable Script:** `src/labeling/label_policy_clauses.py`

**Model:** `qwen3:8b` (Temperature: 0.0)  **Prompt Version:** `policy-labeling-v11-debiased-semantic`

**Algorithmic Flow (v11):**

1. **Category Keyword Matching** — Expanded platform-native keywords across 10 statutory categories (added `"law enforcement"`, `"legal process"`, `"servers located"`, `"security incident"`, 30+ additional terms for IT Act and Cross-Border rules).
2. **Child-Safety Regulatory Context Gate** — Bare tokens (`child`, `minor`, `parent`) require compound statutory context terms (`consent`, `age limit`, `guardian`, `tracking`) to score category-keyword points.
3. **Per-Category Candidate Cap** — At most 2 rules from any single category can enter the 6-slot candidate set.
4. **Semantic Embedding Fallback** — If fewer than 3 candidates are found via keywords, pre-cached `all-MiniLM-L6-v2` embeddings (38 taxonomy vectors) compute cosine similarity to fill remaining slots.
5. **Strict JSON Classification** — Generates verdict, taxonomy rule match, verdict score, and legal justification enforced by Pydantic JSON schema.

**CLI Options:**

```bash
# Full corpus labeling
python -m src.labeling.label_policy_clauses --force-new-run

# Targeted platform re-labeling with merge-back
python -m src.labeling.label_policy_clauses --platform Meta,Youtube,Telegram --force-new-run
```

**Output Artifacts Generated:**

| File | Format | Description |
|---|---|---|
| `corpus/labeled_clauses_bootstrap.csv` | CSV | Master annotated dataset of 771 labeled policy clauses |
| `corpus/labeled_clauses_run_state.json` | JSON | Execution run metadata, prompt version, platform filter, and verdict distribution |
| `corpus/labeled_clauses_failures.json` | JSON | Parsing/execution error log |
| `corpus/label_cache/clause_*.json` | JSON | Per-clause cache keyed by SHA-256 of (prompt_version, model, clause_text, candidates) |

---

## 6. Stage 5: Active Learning, Gold Promotion & Dataset Splitting

### 6.1 Candidate Promotion & Gold Auditing

| File | Format | Description |
|---|---|---|
| `corpus/candidates_for_compliance.csv` | CSV | Prioritized audit review queue for legal researchers |
| `corpus/labeled_clauses_gold.csv` | CSV | Validated benchmark containing audited gold labels |

### 6.2 Stratified Dataset Partitioning

**Executable Script:** `src/training/split_dataset.py`  
**Strategy:** 70% Train (540) / 15% Val (115) / 15% Test (116) — stratified on `verdict`.

| File | Format | Description |
|---|---|---|
| `datasets/splits/train.csv` | CSV | 70% stratified training split (540 rows) |
| `datasets/splits/val.csv` | CSV | 15% stratified validation split (115 rows) |
| `datasets/splits/test.csv` | CSV | 15% stratified test split (116 rows) |
| `corpus/labeled_clauses_final_train.csv` | CSV | Canonical mirrored training dataset |
| `corpus/labeled_clauses_val.csv` | CSV | Canonical mirrored validation dataset |

---

## 7. Stage 6: Transformer Fine-Tuning & Model Evaluation

### 7.1 InLegalBERT Fine-Tuning

**Executable Script:** `src/training/train_transformer.py`  
**Base Architecture:** `law-ai/InLegalBERT`  
**Task:** 3-class classification (Not Addressed / Partially Compliant / Compliant)  
**Hyperparameters:** LR = 1.5e-5 · Epochs = 5 · Batch = 8 · MaxLen = 384  
**Loss:** Smoothed Weighted Cross-Entropy (W_c = N_total / (K * N_c))

| File | Format | Description |
|---|---|---|
| `models/inlegalbert_classifier/` | Checkpoint | Saved PyTorch weights, tokenizer, and classification heads |
| `models/baseline/eval_validation.json` | JSON | Validation metrics across epochs |
| `models/baseline/eval_test.json` | JSON | Uncalibrated test metrics |

### 7.2 Validation-Calibrated Evaluation

**Executable Script:** `src/evaluation/evaluate_transformer_calibrated.py`  
**Purpose:** Calibrates decision thresholds on validation logits to maximize Macro F1 on minority classes.

---

## 8. Stage 7: Statutory Audit & Report Card Generation

### 8.1 Rigorous Scoring Methodology & Formulations

#### Header & Metadata

- **Executable Script:** `src/evaluation/generate_report_card.py`
- **Corpus Evaluated:** `corpus/labeled_clauses_bootstrap.csv` (771 clauses across 6 platforms)

The compliance engine employs a multi-criteria decision analysis (MCDA) framework to produce defensible audit metrics.

---

#### Parameter Definitions & Valuation Rubric

* **R_U (Unique Rules Matched)**: Count of distinct taxonomy rules addressed by a platform.

* **P (Positive Clauses)**: Subset of clauses receiving a substantive positive verdict (`Compliant` or `Partially Compliant`).

* **V_i (Verdict Weight for Clause i)**:
  - `1.0` for **Fully Compliant** (actionable, timeline-bound commitments).
  - `0.5` for **Partially Compliant** (vague or non-enforceable commitments).
  - `0.0` for **Not Addressed / Non-Compliant** (statutory gap or explicit rights waiver).

---

#### Mathematical Formulations

##### Strict Coverage %

Evaluates breadth against the complete statutory universe.

```
Strict Coverage % = (R_U / 38) × 100
```

##### Applicable Scope %

Evaluates breadth against consumer-facing mandates.

```
Applicable Scope % = (R_U / 35) × 100
```

##### Clause Quality %

Evaluates the depth and implementation rigor of matched clauses.

```
Clause Quality % = (Sum of V_i for all i in P / Count of P) × 100
```

##### Effective Compliance %

A 50/50 MCDA linear blend requiring both breadth and depth.

```
Effective Compliance % = (0.5 × Clause Quality %) + (0.5 × Applicable Scope %)
```

##### Combined Final %

A macro-average of statutory compliance frameworks.

```
Combined Final % = (DPDP Act % + DPDP Rules % + IT Act %) / 3
```

---

#### Denominator Justification & Methodological Defense

**Derivation of Denominators (38 vs. 35)**

The absolute statutory taxonomy contains **38 rules** (16 DPDP Act, 15 DPDP Rules, 7 IT Act). However, the **Applicable Scope denominator is adjusted to 35**. Exactly 3 rules (`DPDP_ACT-0121`, `DPDP_ACT-0123`, `DPDP_ACT-0153`; assessability score = 2) govern internal Data Protection Board sanction procedures and voluntary undertakings, which legally cannot appear in consumer-facing privacy policies. Without this adjustment, an artificial 60.5% ceiling unfairly penalizes consumer terms for omitting government-facing board procedures.

---

#### Audit Defense Summary Table

| Metric Name | Academic/Technical Reference | Statutory Rationale |
|:---|:---|:---|
| **Strict Coverage % & Applicable Scope %** | Unique Rule Matching (R_U) | Measuring unique statutory rules matched prevents platforms from gaming the metric by repeating boilerplate terms (e.g., minor protection mentioned 20 times counts as 1 rule). |
| **Clause Quality %** | Ordinal Verdict Scaling (V_i) | The 1.0 vs. 0.5 graded scale mathematically separates actionable, timeline-bound commitments from vague declarations. |
| **Effective Compliance %** | MCDA Linear Weighting (50/50) | Ensures a high compliance score requires both statutory breadth and implementation depth. |
| **Combined Final %** | Macro-Averaging Across Acts | Under Indian jurisprudence, compliance is cumulative. High DPDP compliance (31 rules) cannot legally compensate for failing IT Act 2000 safe-harbor due diligence (7 rules). Equal weighting preserves the co-equal statutory force of each framework. |

### 8.4 Generated Report Files

| File | Format | Description |
|---|---|---|
| `reports/platform_overall_rankings.csv` | CSV | Full platform rankings with all metrics |
| `reports/platform_act_compliance_matrix.csv` | CSV | Act-by-act compliance + Combined Final % |
| `reports/category_gap_analysis.csv` | CSV | Regulatory category coverage gaps |
| `reports/audit_report_summary.json` | JSON | Machine-readable audit with scoring_metadata |

---

## 9. Complete Artifact Directory

| Step | File Path | Format | Description |
|---|---|---|---|
| 1. Extraction | `datasets/GovernmentActs/government_clauses.csv` | CSV | Raw statutory clauses |
| 1. Extraction | `datasets/GovernmentActs/government_clauses.json` | JSON | Clause records with page provenance |
| 1. Extraction | `datasets/GovernmentActs/government_clause_duplicate_report.json` | JSON | Duplicate detection report |
| 1. Extraction | `datasets/SocialMediaPolicies/social_media_clauses.csv` | CSV | Raw platform policy clauses |
| 1. Extraction | `datasets/SocialMediaPolicies/social_media_clauses.json` | JSON | Structured clause corpus |
| 2. Filtering | `datasets/GovernmentActs/government_clauses_filtered.csv` | CSV | Curated statutory clauses |
| 2. Filtering | `datasets/GovernmentActs/government_clauses_cleaned.csv` | CSV | Boilerplate-stripped legal text |
| 2. Filtering | `datasets/GovernmentActs/government_filter_summary.json` | JSON | Filter statistics |
| 2. Filtering | `datasets/SocialMediaPolicies/social_media_clauses_flagged.csv` | CSV | Category-enriched policy clauses |
| 2. Filtering | `datasets/SocialMediaPolicies/social_media_clauses_for_labeling.csv` | CSV | 771 LLM-labeling candidates |
| 3. Taxonomy | `corpus/taxonomy_generation/taxonomy_candidates_raw.json` | JSON | Raw qwen3:8b outputs |
| 3. Taxonomy | `corpus/taxonomy_generation/taxonomy_candidates_normalized.json` | JSON | Schema-validated requirements |
| 3. Taxonomy | `corpus/taxonomy_generation/taxonomy_batch_manifest.json` | JSON | 327-batch execution manifest |
| 3. Taxonomy | `corpus/taxonomy_generation/taxonomy_unmapped_clauses.json` | JSON | 1,977 unmapped clauses |
| 3. Taxonomy | `corpus/taxonomy_generation/taxonomy_invalid_model_records.json` | JSON | Rejected model records |
| 3. Taxonomy | `corpus/taxonomy_generation/triage_by_act/ALL_TRIAGED_CANDIDATES.csv` | CSV | Triaged requirements catalog |
| 3. Taxonomy | `corpus/taxonomy_generation/triage_by_act/ALL_TRIAGED_CANDIDATES.json` | JSON | Structured triaged requirements |
| 3. Taxonomy | `corpus/taxonomy_generation/triage_by_act/ACT_TARGET_SUMMARY.json` | JSON | Per-act candidate distribution |
| 3. Taxonomy | `corpus/taxonomy_generation/final_consolidation_v2/FINAL_TAXONOMY_PROVISIONAL.csv` | CSV | Working taxonomy export |
| 3. Taxonomy | `corpus/taxonomy_generation/final_consolidation_v2/CONSOLIDATION_SUMMARY.json` | JSON | Merge/prune decision log |
| 3. Taxonomy | `corpus/dpdp_taxonomy_final.csv` | CSV | 38-rule compliance benchmark |
| 3. Taxonomy | `corpus/dpdp_taxonomy_final.json` | JSON | Taxonomy with checkable tests |
| 4. Labeling | `corpus/labeled_clauses_bootstrap.csv` | CSV | 771 annotated policy clauses |
| 4. Labeling | `corpus/labeled_clauses_run_state.json` | JSON | Labeling run metadata |
| 4. Labeling | `corpus/labeled_clauses_failures.json` | JSON | Inference error log |
| 4. Labeling | `corpus/label_cache/clause_*.json` | JSON | SHA-256 keyed per-clause cache |
| 5. Active Learning | `corpus/candidates_for_compliance.csv` | CSV | Human review queue |
| 5. Active Learning | `corpus/labeled_clauses_gold.csv` | CSV | Audited gold labels |
| 5. Partitioning | `datasets/splits/train.csv` | CSV | 540-row training split |
| 5. Partitioning | `datasets/splits/val.csv` | CSV | 115-row validation split |
| 5. Partitioning | `datasets/splits/test.csv` | CSV | 116-row test split |
| 5. Partitioning | `corpus/labeled_clauses_final_train.csv` | CSV | Mirrored training set |
| 5. Partitioning | `corpus/labeled_clauses_val.csv` | CSV | Mirrored validation set |
| 6. Training | `models/inlegalbert_classifier/` | Checkpoint | PyTorch weights + tokenizer |
| 6. Training | `models/baseline/eval_validation.json` | JSON | Epoch validation metrics |
| 6. Training | `models/baseline/eval_test.json` | JSON | Test set baseline metrics |
| 7. Reporting | `reports/platform_overall_rankings.csv` | CSV | Platform rankings |
| 7. Reporting | `reports/platform_act_compliance_matrix.csv` | CSV | Act-by-act compliance matrix |
| 7. Reporting | `reports/category_gap_analysis.csv` | CSV | Regulatory gap analysis |
| 7. Reporting | `reports/audit_report_summary.json` | JSON | Machine-readable audit summary |

---

## 10. Execution Command Sequence (Full Replication)

```bash
# Step 1: Extraction
python -m src.extraction.extract_government_clauses
python -m src.extraction.extract_social_media_clauses

# Step 2: Filtering & Pre-labeling Triage
python -m src.filtering.filter_government_clauses
python -m src.filtering.filter_social_media_clauses

# Step 3: Taxonomy Candidate Synthesis & Consolidation
python -m src.taxonomy.generate_taxonomy
python -m src.taxonomy.triage_by_act
python -m src.taxonomy.consolidate_taxonomy

# Step 4: Bootstrap Labeling (Dynamic Gating & Embedding Fallback)
python -m src.labeling.label_policy_clauses --force-new-run

# Optional: Targeted platform re-labeling with merge-back (after keyword map updates)
python -m src.labeling.label_policy_clauses --platform Meta,Youtube,Telegram --force-new-run

# Step 5: Active Learning & Dataset Partitioning
python -m src.training.active_learning_review
python -m src.training.apply_gold_labels
python -m src.training.split_dataset

# Step 6: Transformer Training & Calibration
python -m src.training.train_transformer
python -m src.evaluation.evaluate_transformer_calibrated

# Step 7: Multi-Act Compliance Report Card Generation
python -m src.evaluation.generate_report_card
```

---

## 11. Known Limitations & Recommended Remediation

| Severity | Issue | Affected File(s) | Fix |
|---|---|---|---|
| **P0** | 15 taxonomy rules never matched; 60.5% structural score ceiling | `generate_report_card.py` | Expand keyword maps (v11 partially addresses); exclude uncheckable rules with documentation |
| **P0** | `DPDP_RUL-0227` accounts for 32.9% of all positive training labels | `labeled_clauses_bootstrap.csv` | Re-label with v11 child-keyword gating to reduce concentration |
| **P1** | All train/val/test splits from single `qwen3:8b` labeling run — no independent validation signal | `split_dataset.py`, `labeled_clauses_val.csv` | Independently re-label 30-50 val clauses with second model or human annotator |
| **P1** | `Non-Compliant` class (1 sample) silently dropped by `augment_minority.py` | `augment_minority.py` L34-52 | Add to TARGET_COUNTS or formally exclude with documentation |
| **P1** | Random oversampling — 25 Compliant clauses repeated ~6x per epoch | `augment_minority.py` L67-74 | Replace with paraphrase augmentation or EDA |
| **P2** | `taxonomy_unmapped_clauses.json` (1,977 clauses) has no documented rationale | `corpus/taxonomy_generation/` | Add README.md documenting unmapping criteria |
| **P2** | Hardcoded relative paths in `generate_report_card.py` — silent failure from wrong CWD | `generate_report_card.py` L35-40 | Convert to argparse with absolute path resolution |

---

*Report generated: 2026-09-25 | Pipeline Version: v11 (policy-labeling-v11-debiased-semantic)*
