# Repairing Map Report — Taxonomy v2 Step 1

**Build Date:** Oct 14, 2026  
**Source:** corpus/labeled_clauses_bootstrap.csv (826 rows)  
**V2 Taxonomy:** corpus/dpdp_taxonomy_v2.csv (42 rules)

## Summary

| Status | Count |
|--------|-------|
| KEEP | 215 |
| REPAIR | 25 |
| NEW-MATCH | 586 |
| **Total** | **826** |

## REPAIR Rows (25 clauses with dropped taxonomy IDs)

Clauses paired to taxonomy IDs that do not exist in v2, with best candidate among all 42 v2 rules:

| Clause ID | App | Current ID | Candidate Rule | Similarity |
|-----------|-----|-----------|-----------------|------------|
| S001844 | WhatsApp | DPDP_RUL-0170 | IT_ACT_2-0310 | 0.3943 |
| S001855 | WhatsApp | DPDP_RUL-0170 | DPDP_RUL-0095 | 0.4996 |
| S002666 | Meta | DPDP_ACT-0043 | DPDP_RUL-0184 | 0.5798 |
| S002667 | Meta | DPDP_ACT-0041 | DPDP_RUL-0178 | 0.3952 |
| S002670 | Meta | IT_ACT_2-0284 | DPDP_ACT-0038 | 0.5243 |
| S002671 | Meta | DPDP_ACT-0057 | DPDP_RUL-0201 | 0.4571 |
| S002677 | YouTube | DPDP_ACT-0043 | DPDP_RUL-0184 | 0.3320 |
| S002678 | Google | DPDP_ACT-0048 | DPDP_RUL-0119 | 0.3549 |
| S002679 | Meta | DPDP_ACT-0043 | DPDP_RUL-0184 | 0.6966 |
| S002680 | Meta | DPDP_ACT-0089 | DPDP_ACT-0041 | 0.3782 |
| S002683 | Meta | DPDP_ACT-0070 | DPDP_ACT-0025 | 0.5376 |
| S002685 | Meta | DPDP_ACT-0067 | IT_ACT_2-0285 | 0.2724 |
| S002686 | Meta | DPDP_ACT-0089 | DPDP_ACT-0041 | 0.2589 |
| S002691 | WhatsApp | DPDP_ACT-0043 | DPDP_RUL-0184 | 0.4453 |
| S002692 | Telegram | DPDP_ACT-0043 | DPDP_RUL-0184 | 0.3707 |
| S002693 | Snapchat | DPDP_ACT-0043 | DPDP_RUL-0175 | 0.3408 |
| S002697 | Meta | DPDP_ACT-0089 | DPDP_ACT-0029 | 0.3553 |
| S002701 | WhatsApp | DPDP_ACT-0041 | DPDP_RUL-0178 | 0.2996 |
| S002702 | Telegram | DPDP_ACT-0041 | DPDP_RUL-0178 | 0.3054 |
| S002707 | Meta | DPDP_ACT-0024 | DPDP_RUL-0119 | 0.4193 |
| S002708 | Google | DPDP_ACT-0024 | DPDP_RUL-0176 | 0.3824 |
| S002714 | Google | DPDP_ACT-0043 | DPDP_RUL-0184 | 0.3168 |
| S002715 | Meta | DPDP_ACT-0067 | DPDP_ACT-0027 | 0.2448 |
| S002718 | Telegram | DPDP_ACT-0070 | DPDP_RUL-0208 | 0.3307 |
| S002719 | Google | DPDP_ACT-0070 | DPDP_ACT-0027 | 0.3853 |

## NEW-MATCH Rows (586 clauses with no current match)

### Above-Threshold Candidates (similarity ≥ 0.35)

Breakdown by added v2 rule:

- **DPDP_ACT-0038**: 5 matches
- **DPDP_ACT-0041**: 31 matches
- **DPDP_ACT-0042**: 2 matches
- **DPDP_ACT-0074**: 1 matches
- **DPDP_ACT-0160**: 5 matches
- **DPDP_RUL-0094**: 2 matches
- **DPDP_RUL-0095**: 13 matches
- **DPDP_RUL-0103**: 3 matches
- **DPDP_RUL-0114**: 3 matches
- **DPDP_RUL-0119**: 15 matches
- **IT_SPDI-001**: 25 matches
- **IT_SPDI-002**: 30 matches

**Total above threshold: 135**

### Top 30 NEW-MATCH Rows by Similarity

| Clause ID | App | Best Candidate | Similarity | Above Threshold |
|-----------|-----|---|----------|-----------------|
| S002373 | YouTube | DPDP_ACT-0041 | 0.5621 | Y |
| S001506 | Telegram | IT_SPDI-002 | 0.5618 | Y |
| S000316 | Google | IT_SPDI-001 | 0.5369 | Y |
| S002338 | WhatsApp | IT_SPDI-001 | 0.5356 | Y |
| S001221 | Snapchat | DPDP_RUL-0119 | 0.5346 | Y |
| S001866 | WhatsApp | DPDP_RUL-0119 | 0.5272 | Y |
| S001500 | Telegram | IT_SPDI-002 | 0.5189 | Y |
| S002233 | WhatsApp | IT_SPDI-002 | 0.5171 | Y |
| S001090 | Snapchat | DPDP_RUL-0095 | 0.5089 | Y |
| S001189 | Snapchat | DPDP_RUL-0119 | 0.5045 | Y |
| S002381 | YouTube | DPDP_ACT-0160 | 0.4967 | Y |
| S000856 | Meta | DPDP_ACT-0041 | 0.4869 | Y |
| S001183 | Snapchat | DPDP_ACT-0041 | 0.4841 | Y |
| S002382 | YouTube | DPDP_ACT-0160 | 0.4791 | Y |
| S001879 | WhatsApp | DPDP_RUL-0119 | 0.4759 | Y |
| S000879 | Meta | IT_SPDI-001 | 0.4749 | Y |
| S001226 | Snapchat | IT_SPDI-002 | 0.4735 | Y |
| S001505 | Telegram | IT_SPDI-002 | 0.4721 | Y |
| S001884 | WhatsApp | DPDP_RUL-0119 | 0.4715 | Y |
| S000366 | Google | IT_SPDI-001 | 0.4684 | Y |
| S000855 | Meta | DPDP_ACT-0041 | 0.4677 | Y |
| S000942 | Meta | IT_SPDI-001 | 0.4628 | Y |
| S002170 | WhatsApp | DPDP_ACT-0038 | 0.4624 | Y |
| S001510 | Telegram | IT_SPDI-002 | 0.4607 | Y |
| S001220 | Snapchat | IT_SPDI-002 | 0.4598 | Y |
| S001188 | Snapchat | IT_SPDI-002 | 0.4587 | Y |
| S000778 | Meta | IT_SPDI-001 | 0.4582 | Y |
| S000815 | Meta | DPDP_RUL-0119 | 0.4577 | Y |
| S001212 | Snapchat | DPDP_RUL-0095 | 0.4572 | Y |
| S001623 | Telegram | DPDP_ACT-0041 | 0.4547 | Y |

## Notes

- All similarities computed using sentence-transformers all-MiniLM-L6-v2
- Cosine similarity metric
- Threshold for above_threshold flag: 0.35
- REPAIR candidates matched against all 42 v2 rules
- NEW-MATCH candidates matched against only the 12 added v2 rules
