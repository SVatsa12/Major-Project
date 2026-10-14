# Taxonomy v2 Build Notes

**Build Date:** Oct 14, 2026  
**Branch:** taxonomy-v2  
**Source:** corpus/dpdp_taxonomy_final.csv (38 rules)  
**Output:** corpus/dpdp_taxonomy_v2.{csv,json} (42 rules)

## Deletions (8 rules)

Rules removed as non-platform obligations, duplicates, or unassessable:

- **DPDP_ACT-0057**: Board investigation power
- **DPDP_ACT-0123**: Board penalty power
- **DPDP_ACT-0153**: Penalties are the Board's, not the fiduciary's
- **DPDP_ACT-0048**: Savings clause, no duty
- **DPDP_ACT-0059**: Government blocking power, platform duty duplicated by IT_ACT_2-0285
- **DPDP_ACT-0121**: Voluntary-undertaking process
- **DPDP_RUL-0277**: Vague conflict-of-interest rule, unassessable
- **DPDP_RUL-0170**: Duplicate of DPDP_ACT-0025

## Updates (1 rule)

- **IT_ACT_2-0310**: Sharpened `checkable_test` to explicitly require timeline mention:
  "Check if the document states that the organization notifies the competent authority 
  under the IT Act 2000 and the affected users about a breach of confidentiality or 
  privacy, and the timeline for doing so."

## Verified Unchanged (2 rules)

- **DPDP_RUL-0208**: Already binds platform (Consent Manager functionality)
- **DPDP_RUL-0212**: Already binds platform (access to consent records)

## Additions (12 rules)

### From Candidate Pool (10 rules)
- **DPDP_ACT-0038**: Grievance mechanism (pool)
- **DPDP_ACT-0074**: Respond within s.13(2) period (pool)
- **DPDP_ACT-0041**: No tracking/monitoring/targeting of children (pool)
- **DPDP_ACT-0042**: Verifiably safe processing (pool)
- **DPDP_ACT-0160**: No processing likely to cause detrimental effect (remapped from DPDP_ACT-0040)
- **DPDP_RUL-0103**: Prominently publish means to exercise rights (pool)
- **DPDP_RUL-0114**: Provide details of means to exercise rights (pool)
- **DPDP_RUL-0094**: Breach notice must describe nature, extent, timing (pool)
- **DPDP_RUL-0095**: Inform of breach consequences (pool)
- **DPDP_RUL-0119**: Cross-border transfers subject to CG restrictions (pool)

### New from SPDI Rules 2011 (2 rules)
- **IT_SPDI-001**: Consent for sensitive data collection (new)
- **IT_SPDI-002**: No third-party disclosure of sensitive data without permission (new)

### Special: Remapping
- **DPDP_ACT-0040** (pool) remapped to **DPDP_ACT-0160** (new ID)
  - Reason: DPDP_ACT-0040 already used in final taxonomy for different rule
  - Pool ID recorded in `source_candidate_ids` field

## Category Decisions

- **Sensitive Data Handling**: Retained and grounded in SPDI Rules 2011
- **Transparency & Accountability**: Folded into DPDP_RUL-0103/0114 (publish means)
- **Automated Decision Making**: Retired (no general ADM right in DPDP Act 2023/Rules 2025/IT Act 2000)

## Final Composition

**Total: 42 rules across 11 categories**

### By Act
- DPDP_ACT_2023: 15
- DPDP_RULES_2025: 18
- IT_ACT_2000: 9

### By Category
- Breach Notification: 5
- Children/Vulnerable Groups: 5
- Consent & Notice: 6
- Cross-Border Transfer: 2
- Data Principal Rights: 4
- Data Retention & Erasure: 4
- Grievance Redressal: 3
- Intermediary/Platform Liability: 4
- Security Safeguards: 6
- Sensitive Data Handling: 2
- Significant Data Fiduciary Obligations: 1

## Out of Scope

- No corpus changes
- No re-labeling (separate task post-review)
- No code changes
- No merge to main
