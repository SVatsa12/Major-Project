"""
src/training/active_learning_review.py
Surfaces high-signal clauses that genuinely fulfill statutory mandates
for expert human-in-the-loop review.
"""

from pathlib import Path
import pandas as pd

INPUT_PATH = Path("corpus/labeled_clauses_bootstrap.csv")
OUTPUT_REVIEW_PATH = Path("corpus/candidates_for_compliance.csv")

df = pd.read_csv(INPUT_PATH)
pc_df = df[df["verdict"] == "Partially Compliant"].copy()

# Concrete statutory triggers under DPDPA 2023 / DPDPR 2025
PATTERNS = {
    "Data Principal Rights (Access & Export)": r"(export|download\s+a\s+copy|access\s+your|copy\s+of\s+all\s+your)",
    "Data Principal Rights (Erasure & Rectification)": r"(delete\s+your\s+account|erasure|rectify|amend\s+your\s+personal\s+data)",
    "Grievance Redressal": r"(grievance|complaint|nodal\s+officer|businesscomplaints@)",
    "Security & Safeguards": r"(end-to-end\s+encrypt|keys?\s+(are\s+)?not\s+stored|do\s+not\s+retain\s+customer\s+payment)",
}

candidates = []
for category_name, regex_pattern in PATTERNS.items():
    matched = pc_df[pc_df["clause_text"].str.contains(regex_pattern, case=False, na=False)]
    for _, row in matched.iterrows():
        candidates.append({
            "clause_id": row["clause_id"],
            "app_name": row["app_name"],
            "statutory_target": category_name,
            "current_taxonomy_id": row["best_match_taxonomy_id"],
            "clause_text": row["clause_text"],
            "current_justification": row["justification"]
        })

candidate_df = pd.DataFrame(candidates).drop_duplicates(subset=["clause_id"])
candidate_df.to_csv(OUTPUT_REVIEW_PATH, index=False)

print("=" * 70)
print(f"Candidate discovery complete: {len(candidate_df)} high-signal clauses isolated.")
print("=" * 70)
print(f"Saved review sheet to: {OUTPUT_REVIEW_PATH}")
print("\nBreakdown by Statutory Target:")
print(candidate_df["statutory_target"].value_counts())