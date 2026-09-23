"""
Platform Compliance Report Card Generator.
Audits social media policies against Indian Data Governance Frameworks:
- Digital Personal Data Protection Act, 2023 (DPDP Act 2023)
- Digital Personal Data Protection Rules, 2025 (DPDP Rules 2025)
- Information Technology Act, 2000 & Intermediary Guidelines (IT Act 2000)

Generates multi-act compliance matrices, coverage depth ratios, and platform rankings.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

CORPUS_PATH = Path("corpus/labeled_clauses_final_train.csv")
TAXONOMY_PATH = Path("corpus/dpdp_taxonomy_final.json")
REPORT_OUTPUT_DIR = Path("reports")
REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data and Reference Taxonomy
if not CORPUS_PATH.exists():
    CORPUS_PATH = Path("corpus/labeled_clauses_bootstrap.csv")

df = pd.read_csv(CORPUS_PATH)

# Load taxonomy rules to determine act denominators
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    tax_data = json.load(f)

taxonomy_records = tax_data if isinstance(tax_data, list) else tax_data.get("records", [])
taxonomy_df = pd.DataFrame(taxonomy_records)

TOTAL_TAXONOMY_RULES = len(taxonomy_df) if len(taxonomy_df) > 0 else 30

# Statutory Acts to audit
CANONICAL_ACTS = ["DPDP Act 2023", "DPDP Rules 2025", "IT Act 2000"]

def normalize_act(act_str: str, tax_id: str) -> str:
    tax_str = str(tax_id).upper()
    act_str = str(act_str).upper()
    if "RUL" in tax_str or "DPDPR" in act_str or "RULES" in act_str:
        return "DPDP Rules 2025"
    if "ACT" in tax_str and "IT" not in tax_str and "DPDP" in tax_str:
        return "DPDP Act 2023"
    if "IT" in tax_str or "IT" in act_str:
        return "IT Act 2000"
    return "DPDP Rules 2025"

df["act_normalized"] = df.apply(
    lambda r: normalize_act(r.get("matched_act", ""), r.get("best_match_taxonomy_id", "")),
    axis=1,
)

# Verdict scoring weights
VERDICT_WEIGHTS = {
    "Compliant": 1.0,
    "Partially Compliant": 0.5,
    "Non-Compliant": 0.0,
    "Not Addressed": 0.0,
}
df["score"] = df["verdict"].map(VERDICT_WEIGHTS).fillna(0.0)

# Filter to substantive governance clauses
positive_df = df[df["verdict"].isin(["Compliant", "Partially Compliant"])].copy()

print("=" * 88)
print("              INDIA DATA GOVERNANCE COMPLIANCE AUDIT REPORT CARD              ")
print("=" * 88)

# 2. Platform Compliance Matrix Across All Three Statutory Acts
all_platforms = sorted(df["app_name"].dropna().unique())
act_matrix_rows = []

for platform in all_platforms:
    plat_pos = positive_df[positive_df["app_name"] == platform]
    row = {"Platform": platform}
    for act in CANONICAL_ACTS:
        act_clauses = plat_pos[plat_pos["act_normalized"] == act]
        if len(act_clauses) == 0:
            row[act] = 0.0
        else:
            # Average score across addressed clauses
            row[act] = round(act_clauses["score"].mean() * 100, 1)
    act_matrix_rows.append(row)

act_matrix_df = pd.DataFrame(act_matrix_rows).set_index("Platform")

print("\n--- STATUTORY COMPLIANCE MATRIX (% COMPLIANCE OF ADDRESSED POLICIES) ---")
print(act_matrix_df.to_string())

# 3. Overall Platform Rankings (Factoring in Breadth & Depth)
platform_summary_rows = []

for platform in all_platforms:
    plat_all = df[df["app_name"] == platform]
    plat_pos = positive_df[positive_df["app_name"] == platform]

    total_audited = len(plat_all)
    governance_clauses = len(plat_pos)
    compliant_count = int((plat_pos["verdict"] == "Compliant").sum())
    partial_count = int((plat_pos["verdict"] == "Partially Compliant").sum())
    unique_rules_covered = plat_pos["best_match_taxonomy_id"].nunique()

    # Metrics
    quality_score_pct = (plat_pos["score"].mean() * 100) if governance_clauses > 0 else 0.0
    coverage_breadth_pct = (unique_rules_covered / TOTAL_TAXONOMY_RULES) * 100
    # Composite Score balances writing quality (50%) with regulatory completeness (50%)
    composite_compliance_pct = (0.5 * quality_score_pct) + (0.5 * coverage_breadth_pct)

    platform_summary_rows.append({
        "app_name": platform,
        "audited_clauses": total_audited,
        "governance_clauses": governance_clauses,
        "unique_rules_covered": unique_rules_covered,
        "compliant_count": compliant_count,
        "partial_count": partial_count,
        "clause_quality_pct": round(quality_score_pct, 1),
        "statutory_coverage_pct": round(coverage_breadth_pct, 1),
        "composite_compliance_pct": round(composite_compliance_pct, 1),
    })

rankings_df = pd.DataFrame(platform_summary_rows).sort_values(
    by="composite_compliance_pct", ascending=False
)

print("\n--- OVERALL PLATFORM COMPLIANCE RANKINGS ---")
display_cols = [
    "app_name",
    "governance_clauses",
    "unique_rules_covered",
    "compliant_count",
    "partial_count",
    "clause_quality_pct",
    "statutory_coverage_pct",
    "composite_compliance_pct",
]
print(rankings_df[display_cols].to_string(index=False))

# 4. Category-Level Gap Analysis
cat_summary = positive_df.groupby("dpdp_category").agg(
    clauses_identified=("clause_id", "count"),
    unique_platforms_addressing=("app_name", "nunique"),
    avg_quality_score=("score", lambda s: round(s.mean() * 100, 1)),
).reset_index()

cat_summary["platforms_missing"] = len(all_platforms) - cat_summary["unique_platforms_addressing"]
cat_summary = cat_summary.sort_values(by="clauses_identified", ascending=False)

print("\n--- REGULATORY DOMAIN GAP ANALYSIS ---")
print(cat_summary.to_string(index=False))

# 5. Export Tables
act_matrix_df.to_csv(REPORT_OUTPUT_DIR / "platform_act_compliance_matrix.csv")
rankings_df.to_csv(REPORT_OUTPUT_DIR / "platform_overall_rankings.csv", index=False)
cat_summary.to_csv(REPORT_OUTPUT_DIR / "category_gap_analysis.csv", index=False)

summary_meta = {
    "total_audited_clauses": len(df),
    "total_governance_clauses": len(positive_df),
    "platforms": all_platforms,
    "platform_rankings": rankings_df.to_dict(orient="records"),
    "category_breakdown": cat_summary.to_dict(orient="records"),
}
(REPORT_OUTPUT_DIR / "audit_report_summary.json").write_text(
    json.dumps(summary_meta, indent=2), encoding="utf-8"
)

print("\n" + "=" * 88)
print(f"Audit report card and matrices successfully saved to '{REPORT_OUTPUT_DIR}/'")
print("=" * 88)