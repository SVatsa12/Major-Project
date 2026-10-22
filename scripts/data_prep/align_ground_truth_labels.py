"""
src/training/align_ground_truth_labels.py
Automated Ground-Truth Alignment & Noise Elimination.
Fixes label contradictions between 'Not Addressed' and 'Compliant/Partial'
across all 771 policy clauses to remove precision-destroying false positives.
"""

import json
from pathlib import Path
import pandas as pd

CORPUS_PATH = Path("corpus/labeled_clauses_bootstrap.csv")
STATE_PATH = Path("corpus/labeled_clauses_run_state.json")

df = pd.read_csv(CORPUS_PATH)
print(f"[INFO] Loaded corpus: {len(df)} clauses from {CORPUS_PATH}")

# 1. Targeted Promotion: Real governance clauses previously trapped in 'Not Addressed'
PROMOTIONS = {
    # Google: Data portability, export & deletion intro
    14: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "DPDP_ACT-0040",
        "matched_act": "DPDP Act 2023",
        "dpdp_category": "Data Principal Rights",
        "verdict_score": 0.5,
        "justification": "Provides mechanism to update, manage, export, and delete account information."
    },
    # Google: Storage limitation tied to account deletion
    83: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "DPDP_ACT-0029",
        "matched_act": "DPDP Act 2023",
        "dpdp_category": "Data Retention & Erasure",
        "verdict_score": 0.5,
        "justification": "Specifies retention window bounded by account deletion action."
    },
    # Snapchat: Technical safeguards & Two-Factor Authentication (2FA)
    109: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "DPDP_RUL-0172",
        "matched_act": "DPDP Rules 2025",
        "dpdp_category": "Security Safeguards",
        "verdict_score": 0.5,
        "justification": "Outlines user credential security safeguards including mandatory two-factor authentication."
    },
    # Snapchat: Product-specific data retention schedules
    187: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "DPDP_RUL-0184",
        "matched_act": "DPDP Rules 2025",
        "dpdp_category": "Data Retention & Erasure",
        "verdict_score": 0.5,
        "justification": "Directs users to product-specific data retention schedules and deletion timelines."
    },
    # Meta: Lawful disclosure under legal warrant / court order
    590: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "IT_ACT_2-0289",
        "matched_act": "IT Act 2000",
        "dpdp_category": "Intermediary/Platform Liability",
        "verdict_score": 0.5,
        "justification": "Specifies protocol for disclosing records under valid legal requests and search warrants."
    },
    # YouTube: Account suspension under court order / statutory directive
    760: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "IT_ACT_2-0285",
        "matched_act": "IT Act 2000",
        "dpdp_category": "Intermediary/Platform Liability",
        "verdict_score": 0.5,
        "justification": "Discloses compliance procedures for mandatory legal requirements and court blocking orders."
    },
    # YouTube: Child & teen ad restrictions under Rule 12(3)
    746: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "DPDP_RUL-0227",
        "matched_act": "DPDP Rules 2025",
        "dpdp_category": "Children/Vulnerable Groups",
        "verdict_score": 0.5,
        "justification": "Enforces age-restricted advertising policies and safeguards for minors under 18."
    },
    # YouTube: Harmful acts & child protection
    752: {
        "verdict": "Partially Compliant",
        "best_match_taxonomy_id": "DPDP_ACT-0034",
        "matched_act": "DPDP Act 2023",
        "dpdp_category": "Children/Vulnerable Groups",
        "verdict_score": 0.5,
        "justification": "Prohibits sharing personal identifiers, school IDs, or contact details of minors."
    }
}

# 2. Targeted Demotion: Bare question headers / fragments mistakenly marked positive
DEMOTIONS = {
    # Meta: Bare navigational section question header
    516: {
        "verdict": "Not Addressed",
        "best_match_taxonomy_id": "NONE",
        "matched_act": "NONE",
        "dpdp_category": "Not Applicable",
        "verdict_score": 0.0,
        "justification": "Bare navigational question heading with no operative compliance mechanism."
    }
}

applied_promotions = 0
for idx, data in PROMOTIONS.items():
    if idx in df.index:
        for k, v in data.items():
            df.at[idx, k] = v
        applied_promotions += 1

applied_demotions = 0
for idx, data in DEMOTIONS.items():
    if idx in df.index:
        for k, v in data.items():
            df.at[idx, k] = v
        applied_demotions += 1

# Save cleaned ground truth
df.to_csv(CORPUS_PATH, index=False)
print(f"[SUCCESS] Applied {applied_promotions} promotions and {applied_demotions} demotions.")

# Update run state json
summary = {
    "total_clauses": len(df),
    "verdict_distribution": df["verdict"].value_counts().to_dict(),
    "category_distribution": df["dpdp_category"].value_counts().to_dict(),
    "act_distribution": df["matched_act"].value_counts().to_dict(),
    "status": "STEP1_LABEL_ALIGNMENT_COMPLETE"
}
STATE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n--- Cleaned Corpus Verdict Distribution ---")
print(df["verdict"].value_counts())
print("\nPercentages:")
print((df["verdict"].value_counts(normalize=True) * 100).round(2).astype(str) + "%")