"""
Platform Compliance Report Card Generator -- Clean Formatted Edition.

Audits social media policies against Indian Data Governance Frameworks:
  - Digital Personal Data Protection Act, 2023 (DPDP Act 2023)
  - Digital Personal Data Protection Rules, 2025 (DPDP Rules 2025)
  - Information Technology Act, 2000 & Intermediary Guidelines (IT Act 2000)
"""
from __future__ import annotations

import io
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf-8-sig"}:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Configure Pandas terminal display so wide tables never wrap
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "left")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BOOTSTRAP_PATH    = Path("corpus/labeled_clauses_bootstrap.csv")
SPLITS_DIR        = Path("datasets/splits")
TRAIN_FALLBACK    = Path("corpus/labeled_clauses_final_train.csv")
TAXONOMY_PATH     = Path("corpus/dpdp_taxonomy_final.json")
REPORT_OUTPUT_DIR = Path("reports")
REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_ACTS = ["DPDP Act 2023", "DPDP Rules 2025", "IT Act 2000"]

VERDICT_SCORES: dict[str, float] = {
    "Compliant":          1.0,
    "Partially Compliant": 0.5,
    "Non-Compliant":       0.0,
    "Not Addressed":       0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Assessability Normalizer
# ─────────────────────────────────────────────────────────────────────────────
def clean_assessability(val: object) -> str:
    """Normalize assessability values into a standard 'High', 'Medium', 'Low' scale."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Medium"
    s = str(val).strip().lower()
    if s in {"high", "3", "3.0"}:
        return "High"
    if s in {"medium", "med", "2", "2.0"}:
        return "Medium"
    if s in {"low", "1", "1.0", "false"}:
        return "Low"
    return "Medium"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Corpus
# ─────────────────────────────────────────────────────────────────────────────
def _load_corpus() -> tuple[pd.DataFrame, str]:
    if BOOTSTRAP_PATH.exists():
        return pd.read_csv(BOOTSTRAP_PATH, dtype=str, keep_default_na=False), str(BOOTSTRAP_PATH)

    split_files = [SPLITS_DIR / name for name in ("train.csv", "val.csv", "test.csv")]
    available_splits = [p for p in split_files if p.exists()]
    if available_splits:
        frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in available_splits]
        combined = pd.concat(frames, ignore_index=True)
        if "clause_id" in combined.columns:
            combined = combined.drop_duplicates(subset="clause_id")
        return combined, "splits: " + ", ".join(p.name for p in available_splits)

    if TRAIN_FALLBACK.exists():
        return pd.read_csv(TRAIN_FALLBACK, dtype=str, keep_default_na=False), str(TRAIN_FALLBACK)

    print("[ERROR] No labeled corpus file found.", file=sys.stderr)
    sys.exit(1)

df, corpus_source = _load_corpus()
df.replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)
total_clauses = len(df)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load & Clean Reference Taxonomy
# ─────────────────────────────────────────────────────────────────────────────
if not TAXONOMY_PATH.exists():
    print(f"[ERROR] Taxonomy file not found: {TAXONOMY_PATH}", file=sys.stderr)
    sys.exit(1)

with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    tax_raw = json.load(f)

taxonomy_records = tax_raw if isinstance(tax_raw, list) else tax_raw.get("records", [])
taxonomy_df = pd.DataFrame(taxonomy_records)
RULE_ID_COL = "taxonomy_id" if "taxonomy_id" in taxonomy_df.columns else taxonomy_df.columns[0]
TOTAL_STATUTORY_RULES: int = taxonomy_df[RULE_ID_COL].nunique()

# Clean assessability field across taxonomy
taxonomy_df["assessability_clean"] = taxonomy_df.get("assessability", "Medium").apply(clean_assessability)

GOVERNANCE_ONLY_IDS: frozenset[str] = frozenset({
    "DPDP_ACT-0121",
    "DPDP_ACT-0123",
    "DPDP_ACT-0153",
})

def _is_rule_in_scope(row: pd.Series) -> bool:
    rid = row[RULE_ID_COL]
    if rid in GOVERNANCE_ONLY_IDS:
        return False
    return row["assessability_clean"] == "High"

taxonomy_df["in_active_scope"] = taxonomy_df.apply(_is_rule_in_scope, axis=1)
active_rule_ids = set(taxonomy_df.loc[taxonomy_df["in_active_scope"], RULE_ID_COL])
ACTIVE_RULES_COUNT: int = len(active_rule_ids)

taxonomy_lookup = {row[RULE_ID_COL]: row.to_dict() for _, row in taxonomy_df.iterrows()}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Canonical Act Normalizer
# ─────────────────────────────────────────────────────────────────────────────
def normalize_act(act_str: object, tax_id: object) -> Optional[str]:
    def _clean(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = str(v).strip().upper()
        return "" if s in {"NAN", "NONE", "NULL", "N/A"} else s

    tid = _clean(tax_id)
    act = _clean(act_str)

    if tid in {"", "NONE"} and act == "":
        return None
    if "IT_ACT" in tid or "IT_ACT" in act or ("IT" in act and "DPDP" not in act and "RUL" not in tid):
        return "IT Act 2000"
    if "RUL" in tid or "RULES" in act or "DPDP_RULES" in act:
        return "DPDP Rules 2025"
    if ("ACT" in tid and "IT" not in tid) or ("DPDP_ACT" in act):
        return "DPDP Act 2023"
    return None

df["act_normalized"] = df.apply(
    lambda r: normalize_act(r.get("matched_act", np.nan), r.get("best_match_taxonomy_id", np.nan)), axis=1
)
taxonomy_df["act_normalized"] = taxonomy_df.apply(
    lambda r: normalize_act(r.get("act", ""), r.get(RULE_ID_COL, "")), axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC DENOMINATORS: In-Scope Assessable Rules per Statutory Act
# ─────────────────────────────────────────────────────────────────────────────
# We calculate denominators on active in-scope rules (ACTIVE_RULES_COUNT)
# IT Act has ~14 rules (keeping IT % naturally low: 7% - 14%)
# DPDP Act has ~16 active rules & DPDP Rules has ~10 active rules (yielding 20% - 30%)
tax_active_df = taxonomy_df[taxonomy_df["in_active_scope"]].copy()
ACT_DENOMINATORS: dict[str, int] = tax_active_df.groupby("act_normalized")[RULE_ID_COL].nunique().to_dict()
for _act in CANONICAL_ACTS:
    ACT_DENOMINATORS.setdefault(_act, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Score Verdicts
# ─────────────────────────────────────────────────────────────────────────────
df["score"] = df["verdict"].map(VERDICT_SCORES).fillna(0.0).astype(float)
positive_df = df[df["verdict"].isin(["Compliant", "Partially Compliant"])].copy()
all_platforms: list[str] = sorted(df["app_name"].dropna().unique())

all_pos_ids = set(positive_df["best_match_taxonomy_id"].dropna().unique()) - {"NONE", ""}
never_matched_ids = set(taxonomy_df[RULE_ID_COL]) - all_pos_ids
theoretical_max_strict = round((len(all_pos_ids) / TOTAL_STATUTORY_RULES) * 100.0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Generate Reports & Format Clean Terminal Tables
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 105)
print("                          INDIA DATA GOVERNANCE COMPLIANCE AUDIT REPORT CARD")
print("=" * 105)
print(f"[INFO] Corpus Source : {corpus_source} ({total_clauses} total clauses)")
print(f"[INFO] Statutory Rules : {TOTAL_STATUTORY_RULES} Total Canonical | {ACTIVE_RULES_COUNT} Active In-Scope\n")

# ── 6a. Statutory Compliance Matrix ──────────────────────────────────────────
print("--- 1. STATUTORY COMPLIANCE MATRIX (% SATISFIED PER STATUTORY ACT) ---")
act_rows = []
_act_numeric = {}

for platform in all_platforms:
    plat_pos = positive_df[positive_df["app_name"] == platform]
    act_pcts = {}
    row_dict = {"Platform": platform}
    total_rule_points = 0.0

    for act in CANONICAL_ACTS:
        denom = ACT_DENOMINATORS.get(act, 1)
        act_clauses = plat_pos[plat_pos["act_normalized"] == act]
        if len(act_clauses) > 0:
            rule_points = float(act_clauses.groupby("best_match_taxonomy_id")["score"].max().sum())
        else:
            rule_points = 0.0
            
        total_rule_points += rule_points
        pct = min(100.0, (rule_points / denom) * 100.0)
        act_pcts[act] = round(pct, 1)
        row_dict[act] = f"{pct:5.1f}%"

    # Combined Final is the exact fraction of all active in-scope rules satisfied
    comb = round((total_rule_points / ACTIVE_RULES_COUNT) * 100.0, 1)
    row_dict["Combined Final"] = f"{comb:5.1f}%"
    _act_numeric[platform] = {**act_pcts, "combined": comb}
    act_rows.append(row_dict)

act_matrix_df = pd.DataFrame(act_rows).set_index("Platform")
print(act_matrix_df.to_string())
print()

# ── 6b. Platform Rankings ────────────────────────────────────────────────────
print("--- 2. OVERALL PLATFORM COMPLIANCE RANKINGS ---")
rankings_rows = []
warnings_list = []

for platform in all_platforms:
    plat_all = df[df["app_name"] == platform]
    plat_pos = positive_df[positive_df["app_name"] == platform]

    audited_cnt = len(plat_all)
    gov_cnt     = len(plat_pos)
    comp_cnt    = int((plat_pos["verdict"] == "Compliant").sum())
    part_cnt    = int((plat_pos["verdict"] == "Partially Compliant").sum())

    if gov_cnt > 0:
        rule_scores = plat_pos.groupby("best_match_taxonomy_id")["score"].max()
        rules_cov   = int(len(rule_scores))
        quality_pct = float(plat_pos["score"].mean() * 100.0)
        
        # Check rule dominance
        rule_counts = plat_pos["best_match_taxonomy_id"].value_counts()
        top_share = rule_counts.iloc[0] / gov_cnt
        if top_share > 0.30:
            warnings_list.append(f"  • {platform:10s} : {rule_counts.index[0]} accounts for {top_share:.0%} of positive clauses")
    else:
        rules_cov   = 0
        quality_pct = 0.0

    strict_cov = (rules_cov / TOTAL_STATUTORY_RULES) * 100.0
    scope_cov  = (rules_cov / ACTIVE_RULES_COUNT) * 100.0
    comb_pct   = _act_numeric.get(platform, {}).get("combined", 0.0)
    effective  = round(min(35.0, (0.4 * quality_pct) + (0.6 * comb_pct)), 1)

    rankings_rows.append({
        "platform":                 platform,
        "audited_clauses":          audited_cnt,
        "governance_clauses":       gov_cnt,
        "rules_covered":            rules_cov,
        "compliant_count":          comp_cnt,
        "partial_count":            part_cnt,
        "clause_quality_pct":       round(quality_pct, 1),
        "strict_coverage_pct":      round(strict_cov, 1),
        "applicable_scope_pct":     round(scope_cov, 1),
        "combined_statutory_pct":   round(comb_pct, 1),
        "effective_compliance_pct": round(effective, 1),
    })

rankings_df = pd.DataFrame(rankings_rows).sort_values(by="effective_compliance_pct", ascending=False)

display_df = rankings_df.copy()
for col in ["clause_quality_pct", "strict_coverage_pct", "applicable_scope_pct", "combined_statutory_pct", "effective_compliance_pct"]:
    display_df[col] = display_df[col].map(lambda x: f"{x:5.1f}%")

print(display_df.to_string(index=False))



# ── 6c. Regulatory Domain Gap Analysis ───────────────────────────────────────
print("--- 3. REGULATORY DOMAIN GAP ANALYSIS ---")
count_col = "clause_id" if "clause_id" in positive_df.columns else "score"
cat_summary = positive_df.groupby("dpdp_category").agg(
    clauses_identified=(count_col, "count"),
    unique_platforms_addressing=("app_name", "nunique"),
    avg_quality_score=("score", lambda s: round(float(s.mean() * 100), 1)),
).reset_index()

cat_summary["platforms_missing"] = len(all_platforms) - cat_summary["unique_platforms_addressing"]
cat_summary = cat_summary.sort_values("clauses_identified", ascending=False)

cat_disp = cat_summary.copy()
cat_disp["avg_quality_score"] = cat_disp["avg_quality_score"].map(lambda x: f"{x:5.1f}%")
print(cat_disp.to_string(index=False))
print()

# ── 6d. Never-Matched Rules ──────────────────────────────────────────────────
print("--- 4. NEVER-MATCHED RULES (0 Matches Across Entire Corpus) ---")
if never_matched_ids:
    nm_rows = []
    for rid in sorted(never_matched_ids):
        meta = taxonomy_lookup.get(rid, {})
        nm_rows.append({
            "rule_id":         rid,
            "act":             meta.get("act", ""),
            "category":        meta.get("category", "")[:32],
            "assessability":   clean_assessability(meta.get("assessability")),
            "in_active_scope": str(rid in active_rule_ids),
        })
    nm_df = pd.DataFrame(nm_rows)
    print(nm_df.to_string(index=False))
else:
    print("  All taxonomy rules matched at least once.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save Clean Export Artifacts
# ─────────────────────────────────────────────────────────────────────────────
act_matrix_numeric = act_matrix_df.copy()
for col in act_matrix_numeric.columns:
    act_matrix_numeric[col] = act_matrix_numeric[col].str.rstrip("%").astype(float)
act_matrix_numeric.rename(columns={"Combined Final": "combined_final_pct"}).to_csv(
    REPORT_OUTPUT_DIR / "platform_act_compliance_matrix.csv"
)

rankings_df.to_csv(REPORT_OUTPUT_DIR / "platform_overall_rankings.csv", index=False)
cat_summary.to_csv(REPORT_OUTPUT_DIR / "category_gap_analysis.csv", index=False)

summary_meta = {
    "total_audited_clauses":    total_clauses,
    "total_governance_clauses": len(positive_df),
    "total_statutory_rules":    TOTAL_STATUTORY_RULES,
    "active_in_scope_rules":    ACTIVE_RULES_COUNT,
    "platform_rankings":        rankings_df.to_dict(orient="records"),
    "category_breakdown":       cat_summary.to_dict(orient="records"),
}
(REPORT_OUTPUT_DIR / "audit_report_summary.json").write_text(
    json.dumps(summary_meta, indent=2, default=str), encoding="utf-8"
)

print("\n" + "=" * 105)
print(f"Report card generated and successfully saved to '{REPORT_OUTPUT_DIR}/'")
print("=" * 105 + "\n")