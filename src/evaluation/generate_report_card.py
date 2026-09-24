"""
Platform Compliance Report Card Generator -- Phase 1 Scoring Fixes.

Audits social media policies against Indian Data Governance Frameworks:
  - Digital Personal Data Protection Act, 2023 (DPDP Act 2023)
  - Digital Personal Data Protection Rules, 2025 (DPDP Rules 2025)
  - Information Technology Act, 2000 & Intermediary Guidelines (IT Act 2000)

Fixes applied in this revision
-------------------------------
1. Full-corpus evaluation (bootstrap > splits > train fallback).
2. Silent NaN default in normalize_act() replaced with an explicit None return.
3. Dual-metric scoring: strict_coverage_pct (denom=38) and
   applicable_scope_pct (denom=active in-scope rules).
4. scoring_metadata block appended to audit_report_summary.json.
"""
from __future__ import annotations

import io
import sys
# Force UTF-8 output on Windows consoles (cp1252 cannot encode box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf-8-sig"}:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BOOTSTRAP_PATH   = Path("corpus/labeled_clauses_bootstrap.csv")
SPLITS_DIR       = Path("datasets/splits")
TRAIN_FALLBACK   = Path("corpus/labeled_clauses_final_train.csv")
TAXONOMY_PATH    = Path("corpus/dpdp_taxonomy_final.json")
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
# 1. Load the widest available labeled corpus
# ─────────────────────────────────────────────────────────────────────────────

def _load_corpus() -> tuple[pd.DataFrame, str]:
    """
    Priority:
      1. corpus/labeled_clauses_bootstrap.csv  (all 771 audited clauses)
      2. datasets/splits/{train,val,test}.csv  concatenated, duplicates dropped
      3. corpus/labeled_clauses_final_train.csv (70 % fallback)

    Returns (dataframe, source_description).
    """
    # Priority 1 — bootstrap
    if BOOTSTRAP_PATH.exists():
        df = pd.read_csv(BOOTSTRAP_PATH, dtype=str, keep_default_na=False)
        return df, str(BOOTSTRAP_PATH)

    # Priority 2 — splits
    split_files = [SPLITS_DIR / name for name in ("train.csv", "val.csv", "test.csv")]
    available_splits = [p for p in split_files if p.exists()]
    if available_splits:
        frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in available_splits]
        combined = pd.concat(frames, ignore_index=True)
        if "clause_id" in combined.columns:
            before = len(combined)
            combined = combined.drop_duplicates(subset="clause_id")
            dropped = before - len(combined)
            if dropped:
                print(f"[WARN] Dropped {dropped} duplicate clause_id rows when concatenating splits.")
        source_desc = "splits: " + ", ".join(p.name for p in available_splits)
        return combined, source_desc

    # Priority 3 — train-only fallback
    if TRAIN_FALLBACK.exists():
        df = pd.read_csv(TRAIN_FALLBACK, dtype=str, keep_default_na=False)
        print("[WARN] Falling back to train split only — 30 % of labeled data is excluded.")
        return df, str(TRAIN_FALLBACK)

    print("[ERROR] No labeled corpus file found. Searched:", file=sys.stderr)
    for p in [BOOTSTRAP_PATH, *split_files, TRAIN_FALLBACK]:
        print(f"        {p}", file=sys.stderr)
    sys.exit(1)


df, corpus_source = _load_corpus()

# Normalise empty strings → NaN for numeric/categorical fields
df.replace({"": np.nan, "nan": np.nan, "None": np.nan}, inplace=True)

total_clauses = len(df)
print(f"\n[INFO] Corpus loaded from : {corpus_source}")
print(f"[INFO] Total clauses       : {total_clauses}")
if "app_name" in df.columns:
    platform_counts = df["app_name"].value_counts(dropna=False)
    print("[INFO] Rows per platform   :")
    for platform, count in platform_counts.items():
        print(f"         {platform}: {count}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load reference taxonomy
# ─────────────────────────────────────────────────────────────────────────────
if not TAXONOMY_PATH.exists():
    print(f"[ERROR] Taxonomy file not found: {TAXONOMY_PATH}", file=sys.stderr)
    sys.exit(1)

with open(TAXONOMY_PATH, "r", encoding="utf-8") as _f:
    _tax_raw = json.load(_f)

taxonomy_records = _tax_raw if isinstance(_tax_raw, list) else _tax_raw.get("records", [])
taxonomy_df = pd.DataFrame(taxonomy_records)

RULE_ID_COL = "taxonomy_id" if "taxonomy_id" in taxonomy_df.columns else taxonomy_df.columns[0]
TOTAL_STATUTORY_RULES: int = taxonomy_df[RULE_ID_COL].nunique()

# Build a fast lookup: rule_id -> {category, act, assessability}
taxonomy_lookup: dict[str, dict] = {
    row[RULE_ID_COL]: row.to_dict()
    for _, row in taxonomy_df.iterrows()
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Determine ACTIVE_IN_SCOPE rules
#    A rule is considered "out of scope" (not checkable against consumer-facing
#    policies) if it meets either criterion:
#      (a) assessability is "2" or numeric < 3  (internal board-only obligation)
#      (b) it is explicitly excluded by rule_id (known governance-only rules)
#    All other rules (assessability "High" or numeric >= 3) are active.
# ─────────────────────────────────────────────────────────────────────────────
# Internal board-sanctions (assessability == 2) identified from taxonomy JSON
GOVERNANCE_ONLY_IDS: frozenset[str] = frozenset({
    "DPDP_ACT-0121",  # assessability 2 — board-level sanction obligation
    "DPDP_ACT-0123",  # assessability 2 — board-level sanction obligation
    "DPDP_ACT-0153",  # assessability 2 — board-level sanction obligation
})


def _assessability_is_active(value: object) -> bool:
    """Return True if the assessability value represents a consumer-checkable rule."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True  # unknown — include by default
    s = str(value).strip()
    if s.lower() == "high":
        return True
    try:
        return int(s) >= 3
    except ValueError:
        return True  # non-numeric string other than "High" — include


active_rule_ids: set[str] = {
    row[RULE_ID_COL]
    for _, row in taxonomy_df.iterrows()
    if row[RULE_ID_COL] not in GOVERNANCE_ONLY_IDS
    and _assessability_is_active(row.get("assessability"))
}
ACTIVE_RULES_COUNT: int = len(active_rule_ids)

# Rules that never appeared in any positive match (informational)
all_positive_ids_in_corpus: set[str] = set()
if "best_match_taxonomy_id" in df.columns:
    all_positive_ids_in_corpus = (
        set(df.loc[
            df["verdict"].isin(["Compliant", "Partially Compliant"]),
            "best_match_taxonomy_id",
        ].dropna().unique())
        - {"NONE", ""}
    )
never_matched_ids = set(taxonomy_df[RULE_ID_COL]) - all_positive_ids_in_corpus
theoretical_max_strict = round(len(all_positive_ids_in_corpus) / TOTAL_STATUTORY_RULES * 100, 1)

print(
    f"[INFO] Strict Statutory Coverage Denominator: {TOTAL_STATUTORY_RULES} rules "
    f"(Theoretical Max: {theoretical_max_strict}% due to "
    f"{len(never_matched_ids)} unmapped/out-of-scope rules). "
    f"Active Scope Denominator: {ACTIVE_RULES_COUNT} rules."
)
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fixed normalize_act() — no silent NaN default
# ─────────────────────────────────────────────────────────────────────────────

def normalize_act(act_str: object, tax_id: object) -> Optional[str]:
    """
    Map raw act_str / taxonomy_id strings to one of the three canonical act
    labels, or return None if the clause has no statutory match.

    Returns None for:
      - tax_id of "NONE", None, NaN, or empty string
      - act_str that is null/NaN/empty with no informative tax_id
      - rows that cannot be mapped to any canonical act
    """
    # Normalise inputs to plain uppercase strings; flag missing values
    def _clean(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        s = str(v).strip().upper()
        return "" if s in {"NAN", "NONE", "NULL", "N/A"} else s

    tid = _clean(tax_id)
    act = _clean(act_str)

    # No match sentinel — row is definitively Not Addressed
    if tid in {"", "NONE"} and act == "":
        return None

    # IT Act 2000 — check before DPDP Act because "IT" appears in "DPDP_ACT" strings
    if "IT_ACT" in tid or "IT_ACT" in act or (
        "IT" in act and "DPDP" not in act and "RUL" not in tid
    ):
        return "IT Act 2000"

    # DPDP Rules 2025
    if "RUL" in tid or "RULES" in act or "DPDP_RULES" in act:
        return "DPDP Rules 2025"

    # DPDP Act 2023 — matched by "ACT" in either field, but not IT
    if ("ACT" in tid and "IT" not in tid) or ("DPDP_ACT" in act):
        return "DPDP Act 2023"

    # Nothing matched
    return None


# Apply to the full dataframe without overwriting NONE-rows with a default act
df["act_normalized"] = df.apply(
    lambda r: normalize_act(
        r.get("matched_act", np.nan),
        r.get("best_match_taxonomy_id", np.nan),
    ),
    axis=1,
)

# Apply to taxonomy for denominator grouping
taxonomy_df["act_normalized"] = taxonomy_df.apply(
    lambda r: normalize_act(r.get("act", ""), r.get(RULE_ID_COL, "")),
    axis=1,
)

# Per-act denominators from the full taxonomy (for the act-level matrix)
ACT_DENOMINATORS: dict[str, int] = (
    taxonomy_df.groupby("act_normalized")[RULE_ID_COL]
    .nunique()
    .to_dict()
)
# Ensure all three canonical acts are represented (some may be absent after filtering)
for _act in CANONICAL_ACTS:
    ACT_DENOMINATORS.setdefault(_act, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Score verdicts
# ─────────────────────────────────────────────────────────────────────────────
df["score"] = df["verdict"].map(VERDICT_SCORES).fillna(0.0).astype(float)

# Positive-only subset
positive_df = df[df["verdict"].isin(["Compliant", "Partially Compliant"])].copy()

all_platforms: list[str] = sorted(df["app_name"].dropna().unique())

# ─────────────────────────────────────────────────────────────────────────────
# 6. Print compliance report
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 100)
print("                  INDIA DATA GOVERNANCE COMPLIANCE AUDIT REPORT CARD                  ")
print("=" * 100)

# ── 6a. Statutory Compliance Matrix (per act, per platform) ──────────────────
print("\n--- STATUTORY COMPLIANCE MATRIX (% STATUTORY MANDATES SATISFIED PER ACT) ---\n")
print(
    "[INFO] Combined Final % is the macro-average of compliance across "
    "DPDP Act 2023, DPDP Rules 2025, and IT Act 2000.\n"
    "       Equal weight is assigned to each framework so poor IT Act "
    "performance cannot be diluted by DPDP volume.\n"
)

act_matrix_rows: list[dict] = []
# Parallel numeric store used later for rankings and exports
_act_numeric: dict[str, dict[str, float]] = {}  # platform -> {act -> float}

for platform in all_platforms:
    plat_pos = positive_df[positive_df["app_name"] == platform]
    row: dict[str, object] = {"Platform": platform}
    act_pcts: dict[str, float] = {}

    for act in CANONICAL_ACTS:
        denom = ACT_DENOMINATORS.get(act, 1)
        act_clauses = plat_pos[plat_pos["act_normalized"] == act]

        if len(act_clauses) == 0:
            pct = 0.0
        else:
            # Best score per unique rule (prevents multi-clause inflation)
            rule_best = act_clauses.groupby("best_match_taxonomy_id")["score"].max()
            pct = min(100.0, (rule_best.sum() / denom) * 100.0)

        act_pcts[act] = round(pct, 1)
        row[act] = f"{round(pct, 1)}%"

    # Macro-average: equal weight across all three statutory frameworks
    combined_val = round(float(np.mean([act_pcts[a] for a in CANONICAL_ACTS])), 1)
    row["Combined Final %"] = f"{combined_val}%"
    _act_numeric[platform] = {**act_pcts, "combined": combined_val}

    act_matrix_rows.append(row)

act_matrix_df = pd.DataFrame(act_matrix_rows).set_index("Platform")
print(act_matrix_df.to_string())

# ── 6b. Overall Platform Rankings with dual metrics ──────────────────────────
print("\n--- OVERALL PLATFORM COMPLIANCE RANKINGS ---\n")
print(
    f"  Metrics explanation:\n"
    f"    strict_coverage_pct      : unique rules covered / {TOTAL_STATUTORY_RULES} (full taxonomy)\n"
    f"    applicable_scope_pct     : unique rules covered / {ACTIVE_RULES_COUNT} (active in-scope rules)\n"
    f"    clause_quality_pct       : mean verdict score among positive clauses only\n"
    f"    combined_statutory_pct   : macro-average of DPDP Act 2023 / DPDP Rules 2025 / IT Act 2000 compliance\n"
    f"    effective_compliance_pct : 0.5 x clause_quality_pct + 0.5 x applicable_scope_pct\n"
)

rankings_rows: list[dict] = []

for platform in all_platforms:
    plat_all = df[df["app_name"] == platform]
    plat_pos = positive_df[positive_df["app_name"] == platform]

    total_audited    = len(plat_all)
    governance_count = len(plat_pos)
    compliant_count  = int((plat_pos["verdict"] == "Compliant").sum())
    partial_count    = int((plat_pos["verdict"] == "Partially Compliant").sum())

    if governance_count > 0:
        rule_scores             = plat_pos.groupby("best_match_taxonomy_id")["score"].max()
        unique_rules_covered    = int(len(rule_scores))
        statutory_points_earned = float(rule_scores.sum())
        clause_quality_pct      = float(plat_pos["score"].mean() * 100.0)
    else:
        unique_rules_covered    = 0
        statutory_points_earned = 0.0
        clause_quality_pct      = 0.0

    strict_coverage_pct      = (unique_rules_covered / TOTAL_STATUTORY_RULES) * 100.0
    applicable_scope_pct     = (unique_rules_covered / ACTIVE_RULES_COUNT) * 100.0
    effective_compliance_pct = 0.5 * clause_quality_pct + 0.5 * applicable_scope_pct

    # Combined statutory % — pull from the act-matrix numeric store computed above
    combined_statutory_pct = _act_numeric.get(platform, {}).get("combined", 0.0)

    # Dominant rule flag — alert if one rule accounts for >30% of this platform's positives
    dominant_rule_flag = ""
    if governance_count > 0:
        rule_counts = plat_pos["best_match_taxonomy_id"].value_counts()
        top_rule_share = rule_counts.iloc[0] / governance_count if len(rule_counts) > 0 else 0.0
        if top_rule_share > 0.30:
            dominant_rule_flag = f"[WARN] {rule_counts.index[0]} ({top_rule_share:.0%} of positives)"

    rankings_rows.append({
        "app_name":                  platform,
        "audited_clauses":           total_audited,
        "governance_clauses":        governance_count,
        "unique_rules_covered":      unique_rules_covered,
        "compliant_count":           compliant_count,
        "partial_count":             partial_count,
        "clause_quality_pct":        round(clause_quality_pct, 1),
        "strict_coverage_pct":       round(strict_coverage_pct, 1),
        "applicable_scope_pct":      round(applicable_scope_pct, 1),
        "combined_statutory_pct":    round(combined_statutory_pct, 1),
        "effective_compliance_pct":  round(effective_compliance_pct, 1),
        "dominant_rule_warning":     dominant_rule_flag,
    })

rankings_df = pd.DataFrame(rankings_rows).sort_values(
    by="effective_compliance_pct", ascending=False
)

display_cols = [
    "app_name",
    "audited_clauses",
    "governance_clauses",
    "unique_rules_covered",
    "compliant_count",
    "partial_count",
    "clause_quality_pct",
    "strict_coverage_pct",
    "applicable_scope_pct",
    "combined_statutory_pct",
    "effective_compliance_pct",
    "dominant_rule_warning",
]
print(rankings_df[display_cols].to_string(index=False))

# ── 6c. Regulatory Domain Gap Analysis ───────────────────────────────────────
print("\n--- REGULATORY DOMAIN GAP ANALYSIS ---\n")

count_col = "clause_id" if "clause_id" in positive_df.columns else "score"
cat_summary = positive_df.groupby("dpdp_category").agg(
    clauses_identified=(count_col, "count"),
    unique_platforms_addressing=("app_name", "nunique"),
    avg_quality_score=("score", lambda s: round(float(s.mean() * 100), 1)),
).reset_index()

cat_summary["platforms_missing"] = len(all_platforms) - cat_summary["unique_platforms_addressing"]
cat_summary = cat_summary.sort_values("clauses_identified", ascending=False)
print(cat_summary.to_string(index=False))

# ── 6d. Never-matched rules table ────────────────────────────────────────────
print("\n--- NEVER-MATCHED RULES (zero hits across all platforms) ---\n")
if never_matched_ids:
    nm_rows = []
    for rid in sorted(never_matched_ids):
        meta = taxonomy_lookup.get(rid, {})
        nm_rows.append({
            "rule_id":       rid,
            "act":           meta.get("act", ""),
            "category":      meta.get("category", ""),
            "assessability": meta.get("assessability", ""),
            "in_active_scope": rid in active_rule_ids,
        })
    nm_df = pd.DataFrame(nm_rows)
    print(nm_df.to_string(index=False))
else:
    print("  All taxonomy rules matched at least once — no dead rules found.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Export artefacts
# ─────────────────────────────────────────────────────────────────────────────

# 7a. Act matrix (strip % signs for numeric CSV)
act_matrix_numeric = act_matrix_df.copy()
for col in act_matrix_numeric.columns:
    act_matrix_numeric[col] = act_matrix_numeric[col].str.rstrip("%").astype(float)
act_matrix_numeric.to_csv(REPORT_OUTPUT_DIR / "platform_act_compliance_matrix.csv")

# Rename the Combined Final % column to a CSV-safe name
act_matrix_numeric = act_matrix_numeric.rename(
    columns={"Combined Final %": "combined_final_pct"}
)

# 7b. Rankings — drop the warning column for the CSV (kept in JSON)
rankings_df.to_csv(REPORT_OUTPUT_DIR / "platform_overall_rankings.csv", index=False)

# 7c. Category gap analysis
cat_summary.to_csv(REPORT_OUTPUT_DIR / "category_gap_analysis.csv", index=False)

# 7d. Audit summary JSON with scoring_metadata
scoring_metadata = {
    "total_statutory_rules":             TOTAL_STATUTORY_RULES,
    "active_in_scope_rules":             ACTIVE_RULES_COUNT,
    "governance_only_excluded_rules":    sorted(GOVERNANCE_ONLY_IDS),
    "never_matched_rule_count":          len(never_matched_ids),
    "never_matched_rule_ids":            sorted(never_matched_ids),
    "theoretical_max_strict_coverage_pct": theoretical_max_strict,
    "corpus_source":                     corpus_source,
    "total_clauses_evaluated":           total_clauses,
    "act_denominators":                  ACT_DENOMINATORS,
    "dual_metric_note": (
        "strict_coverage_pct uses denominator 38 (full taxonomy). "
        f"applicable_scope_pct uses denominator {ACTIVE_RULES_COUNT} "
        "(consumer-checkable rules only, assessability >= 3, excluding board-sanction rules). "
        "combined_statutory_pct = macro-average of DPDP Act 2023 / DPDP Rules 2025 / IT Act 2000 "
        "(equal weight per framework). "
        "effective_compliance_pct = 0.5 x clause_quality_pct + 0.5 x applicable_scope_pct."
    ),
}

summary_meta = {
    "scoring_metadata":         scoring_metadata,
    "total_audited_clauses":    total_clauses,
    "total_governance_clauses": len(positive_df),
    "platform_rankings":        rankings_df.to_dict(orient="records"),
    "category_breakdown":       cat_summary.to_dict(orient="records"),
    "act_compliance_matrix":    act_matrix_df.to_dict(orient="index"),
}

(REPORT_OUTPUT_DIR / "audit_report_summary.json").write_text(
    json.dumps(summary_meta, indent=2, default=str),
    encoding="utf-8",
)

print("\n" + "=" * 100)
print(f"Audit report card saved to '{REPORT_OUTPUT_DIR}/'")
print(
    f"  platform_act_compliance_matrix.csv  |  "
    f"platform_overall_rankings.csv  |  "
    f"category_gap_analysis.csv  |  "
    f"audit_report_summary.json"
)
print("=" * 100)