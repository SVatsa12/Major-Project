"""
Production-grade triage and compliance pre-filtering for social-media policy clauses.

Preserves the complete raw corpus while generating high-precision compliance subsets
for multi-act evaluation (DPDP Act 2023, DPDP Rules 2025, IT Act 2000, Intermediary Guidelines).

Input
-----
    datasets/SocialMediaPolicies/social_media_clauses.json (or .csv)

Output
------
    - social_media_clauses_flagged.json / .csv      (Full preserved corpus with rich metadata)
    - social_media_clauses_for_labeling.csv / .json (Curated high-density compliance candidate subset)
    - social_media_clause_review_queue.json
    - social_media_clause_duplicate_report.json
    - social_media_filter_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT = Path("datasets/SocialMediaPolicies/social_media_clauses.json")
DEFAULT_OUTPUT_DIR = Path("datasets/SocialMediaPolicies")

BOILERPLATE_PATTERNS = [
    (r"^back\s+to\s+top$", "navigation link"),
    (r"^table\s+of\s+contents$", "document navigation"),
    (r"^contents$", "document navigation"),
    (r"^page\s+\d+(?:\s+of\s+\d+)?$", "page number artifact"),
    (r"^last\s+updated(?:\s*[:\-].*)?$", "revision-date stamp"),
    (r"^effective\s+\w+\s+\d{1,2},?\s+\d{4}$", "effective-date stamp"),
    (r"^(?:privacy\s+policy|terms\s+of\s+service|cookies\s+policy)$", "bare document title line"),
    (r"^legal\s+info$", "navigation block"),
    (r"^archived\s+versions?$", "navigation block"),
]
BOILERPLATE_REGEX = [
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in BOILERPLATE_PATTERNS
]

# High-precision statutory concept signals mapped to DPDP 2023, DPDP Rules 2025, and IT Act 2000
STATUTORY_RELEVANCE_TERMS: dict[str, tuple[str, ...]] = {
    "consent_and_notice": (
        "consent", "opt-out", "opt out", "opt-in", "withdraw consent",
        "withdrawal of consent", "notice at collection", "specified purpose",
        "purpose of processing", "freely given", "unconditional consent"
    ),
    "security_safeguards": (
        "security measures", "encryption", "encrypted", "safeguard personal data",
        "unauthorized processing", "accidental disclosure", "loss of data",
        "technical and organizational", "access control", "pseudonymization",
        "tokenization", "reasonable security practices"
    ),
    "breach_notification": (
        "data breach", "security breach", "personal data breach", "notify the board",
        "incident response", "notify affected", "breach notification", "compromised data"
    ),
    "retention_and_erasure": (
        "erase your", "erasure", "retention period", "delete your personal data",
        "delete your information", "data retention", "no longer necessary",
        "right to be forgotten", "deletion of account", "storage limitation"
    ),
    "children_and_minors": (
        "child", "children", "minor", "parental consent", "lawful guardian",
        "verifiable consent", "tracking of children", "behavioural monitoring of children",
        "age gating", "under 18", "under 13", "safety of children"
    ),
    "data_principal_rights": (
        "right to access", "right to correction", "access your data",
        "correct inaccurate", "rectify", "data portability", "download your information",
        "copy of your personal", "exercise your rights"
    ),
    "grievance_and_dpo": (
        "grievance officer", "data protection officer", "dpo", "redressal",
        "grievance mechanism", "complaint", "file a complaint", "nodal contact",
        "respond within", "appellate"
    ),
    "cross_border_transfer": (
        "cross-border", "transfer outside india", "overseas transfer",
        "transfer of personal data", "international data transfer", "data localization",
        "countries outside india"
    ),
    "intermediary_due_diligence": (
        "intermediary", "rule 3", "grievance redressal mechanism", "takedown notice",
        "blocking order", "section 69a", "court order", "unlawful content",
        "due diligence", "remove within 24 hours", "remove within 36 hours",
        "government coordination", "cyber security incident"
    ),
}

NORMATIVE_TERMS = (
    "shall", "must", "may not", "will not", "required to", "prohibited",
    "responsible for ensuring", "subject to applicable law", "you have the right to",
    "in accordance with applicable law",
)

PRIVACY_DOC_KEYWORDS = (
    "privacy", "data", "protection", "security", "child", "dpa", "cookie"
)
COMMERCIAL_DOC_KEYWORDS = (
    "ad policy", "ad_policy", "advertising", "commercial", "payment",
    "billing", "bidding", "cloud acceptable use"
)


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path
    near_duplicate_threshold: float
    high_relevance_threshold: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage and compliance pre-filtering of social-media clauses.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.98,
        help="Jaccard threshold for near-duplicate identification.",
    )
    parser.add_argument(
        "--high-relevance-threshold",
        type=int,
        default=2,
        help="Minimum concept count for high-priority compliance matching.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


def display_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", canonical_text(text)))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def stitch_fragmented_clauses(df: pd.DataFrame) -> pd.DataFrame:
    """Stitches dangling list headers ending in ':' into the subsequent clause."""
    if "clause_text" not in df.columns:
        return df

    records = df.to_dict(orient="records")
    stitched: list[dict[str, Any]] = []
    i = 0
    while i < len(records):
        curr = records[i]
        curr_text = display_text(curr.get("clause_text", ""))
        doc = curr.get("source_document", "")

        # If clause ends with a colon and next clause exists in same document, merge
        if curr_text.endswith(":") and (i + 1 < len(records)):
            nxt = records[i + 1]
            if nxt.get("source_document", "") == doc:
                nxt_text = display_text(nxt.get("clause_text", ""))
                curr["clause_text"] = f"{curr_text} {nxt_text}"
                curr["stitched_with_next"] = True
                stitched.append(curr)
                i += 2
                continue

        curr["stitched_with_next"] = False
        stitched.append(curr)
        i += 1

    return pd.DataFrame(stitched)


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.casefold() == ".csv":
        dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            payload = payload["records"]
        elif isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON must be a list of records or contain a 'records' list")
        dataframe = pd.DataFrame(payload)
    else:
        raise ValueError("Input must be JSON or CSV")

    if "clause_text" not in dataframe.columns:
        raise ValueError("Input must contain the 'clause_text' column")
    if dataframe.empty:
        raise ValueError("Input contains no records")

    dataframe["clause_text"] = dataframe["clause_text"].fillna("").astype(str)
    return dataframe


def row_value(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    return "" if pd.isna(value) else str(value)


def boilerplate_signal(text: str) -> tuple[bool, str]:
    candidate = display_text(text)
    for pattern, reason in BOILERPLATE_REGEX:
        if pattern.fullmatch(candidate):
            return True, reason
    return False, ""


def document_domain(source_document: str) -> str:
    doc_lower = source_document.lower()
    if any(k in doc_lower for k in PRIVACY_DOC_KEYWORDS):
        return "data_privacy"
    if any(k in doc_lower for k in COMMERCIAL_DOC_KEYWORDS):
        return "commercial_advertising"
    return "platform_governance"


def statutory_concept_signal(text: str, section_heading: str) -> tuple[int, list[str], str, bool]:
    combined = canonical_text(f"{section_heading} {text}")
    categories: list[str] = []
    for category, terms in STATUTORY_RELEVANCE_TERMS.items():
        if any(term in combined for term in terms):
            categories.append(category)

    normative = any(term in combined for term in NORMATIVE_TERMS)
    score = len(categories)
    if normative and score > 0:
        score += 1

    if score >= 3:
        level = "high"
    elif score >= 1:
        level = "medium"
    else:
        level = "low"
    return score, categories, level, normative


def quality_flags(text: str, row: pd.Series) -> list[str]:
    flags: list[str] = []
    cleaned = display_text(text)
    if not cleaned:
        flags.append("empty_clause_text")
    if len(cleaned) < 12:
        flags.append("very_short_text")
    if len(cleaned) > 5000:
        flags.append("very_long_text")
    if "source_document" in row.index and not row_value(row, "source_document").strip():
        flags.append("missing_source_document")
    if "app_name" in row.index and not row_value(row, "app_name").strip():
        flags.append("missing_app_name")
    return flags


def exact_duplicate_groups(dataframe: pd.DataFrame) -> pd.Series:
    counts = dataframe["_canonical_hash"].value_counts()
    group_by_hash: dict[str, str] = {}
    next_group = 1
    result: list[str] = []
    for value in dataframe["_canonical_hash"]:
        if counts[value] == 1:
            result.append("")
            continue
        if value not in group_by_hash:
            group_by_hash[value] = f"EXACT-{next_group:05d}"
            next_group += 1
        result.append(group_by_hash[value])
    return pd.Series(result, index=dataframe.index, dtype="string")


def near_duplicate_groups(
    dataframe: pd.DataFrame,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    groups: list[dict[str, Any]] = []
    assignments = [""] * len(dataframe)
    similarities = [0.0] * len(dataframe)

    for position, value in enumerate(dataframe["_tokens"]):
        best_group: dict[str, Any] | None = None
        best_score = 0.0
        for group in groups:
            score = jaccard(value, group["representative"])
            if score >= threshold and score > best_score:
                best_group = group
                best_score = score

        if best_group is None:
            best_group = {
                "id": f"NEAR-{len(groups) + 1:05d}",
                "representative": value,
                "members": [],
            }
            groups.append(best_group)
            best_score = 1.0

        best_group["members"].append(position)
        similarities[position] = round(best_score, 6)

    for group in groups:
        if len(group["members"]) > 1:
            for position in group["members"]:
                assignments[position] = group["id"]

    return (
        pd.Series(assignments, index=dataframe.index, dtype="string"),
        pd.Series(similarities, index=dataframe.index, dtype="float64"),
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = Config(
        input_path=args.input,
        output_dir=args.output_dir,
        near_duplicate_threshold=args.near_duplicate_threshold,
        high_relevance_threshold=args.high_relevance_threshold,
    )
    run_timestamp = utc_now_iso()

    try:
        raw_df = load_input(config.input_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Preprocessing: Join fragmented introductory stems with next clause
    dataframe = stitch_fragmented_clauses(raw_df)
    original_columns = list(dataframe.columns)
    dataframe = dataframe.copy()
    dataframe["_canonical_hash"] = dataframe["clause_text"].map(sha256_text)
    dataframe["_tokens"] = dataframe["clause_text"].map(tokens)

    boilerplate_values: list[bool] = []
    boilerplate_reasons: list[str] = []
    relevance_scores: list[int] = []
    relevance_categories: list[list[str]] = []
    relevance_levels: list[str] = []
    normative_values: list[bool] = []
    quality_values: list[list[str]] = []
    domain_values: list[str] = []
    priorities: list[str] = []

    for _, row in dataframe.iterrows():
        text = display_text(row.get("clause_text", ""))
        heading = row_value(row, "section_heading")
        source_doc = row_value(row, "source_document")

        is_boilerplate, boilerplate_reason = boilerplate_signal(text)
        score, categories, level, normative = statutory_concept_signal(text, heading)
        flags = quality_flags(text, row)
        domain = document_domain(source_doc)

        boilerplate_values.append(is_boilerplate)
        boilerplate_reasons.append(boilerplate_reason)
        relevance_scores.append(score)
        relevance_categories.append(categories)
        relevance_levels.append(level)
        normative_values.append(normative)
        quality_values.append(flags)
        domain_values.append(domain)

        # Review priority heuristic
        if flags:
            priority = "quality_check"
        elif is_boilerplate:
            priority = "boilerplate_review"
        elif score >= config.high_relevance_threshold and domain == "data_privacy":
            priority = "high"
        elif score >= 1 or domain == "data_privacy":
            priority = "medium"
        else:
            priority = "low"
        priorities.append(priority)

    dataframe["likely_boilerplate"] = boilerplate_values
    dataframe["boilerplate_reason"] = boilerplate_reasons
    dataframe["document_domain"] = domain_values
    dataframe["relevance_score"] = relevance_scores
    dataframe["relevance_categories"] = [json.dumps(val, ensure_ascii=False) for val in relevance_categories]
    dataframe["relevance_level"] = relevance_levels
    dataframe["contains_normative_language"] = normative_values
    dataframe["quality_flags"] = [json.dumps(val, ensure_ascii=False) for val in quality_values]
    dataframe["review_priority"] = priorities
    dataframe["exact_duplicate_group"] = exact_duplicate_groups(dataframe)

    near_groups, near_similarity = near_duplicate_groups(
        dataframe,
        config.near_duplicate_threshold,
    )
    dataframe["near_duplicate_group"] = near_groups
    dataframe["near_duplicate_similarity"] = near_similarity

    dataframe["filter_decision"] = "RETAIN"
    dataframe["retention_reason"] = "recall_first_no_deletion"
    dataframe["eligible_for_taxonomy_matching"] = True
    dataframe["filter_run_timestamp_utc"] = run_timestamp

    summary = {
        "run_timestamp_utc": run_timestamp,
        "input_path": str(config.input_path),
        "total_records": int(len(dataframe)),
        "review_priority_counts": dataframe["review_priority"].value_counts().to_dict(),
        "document_domain_counts": dataframe["document_domain"].value_counts().to_dict(),
        "relevance_level_counts": dataframe["relevance_level"].value_counts().to_dict(),
        "stitched_fragment_count": int(dataframe.get("stitched_with_next", pd.Series(dtype=bool)).sum()),
    }

    print("=== Social-Media Clause Triage Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run complete; no output files were written.")
        return 0

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Define only the 10 essential audit and context columns
    essential_cols = [
        "clause_id", "app_name", "source_document", "source_file",
        "page_start", "page_end", "section_heading", "clause_text",
        "relevance_score", "relevance_level"
    ]

    # 2. Filter labeling candidates before pruning internal deduplication hash
    labeling_mask = (
        (dataframe["review_priority"].isin(["high", "medium"])) &
        (~dataframe["likely_boilerplate"]) &
        (~dataframe["document_domain"].isin(["commercial_advertising"])) &
        (dataframe["quality_flags"] == "[]")
    )
    labeling_candidates = dataframe[labeling_mask].drop_duplicates(subset=["_canonical_hash"]).copy()

    # 3. Retain only essential columns across both datasets
    output_dataframe = dataframe[[c for c in essential_cols if c in dataframe.columns]]
    labeling_dataframe = labeling_candidates[[c for c in essential_cols if c in labeling_candidates.columns]]

    # 4. Save only the two required clean CSV files
    main_csv = config.output_dir / "social_media_clauses_flagged.csv"
    for_labeling_csv = config.output_dir / "social_media_clauses_for_labeling.csv"

    output_dataframe.to_csv(main_csv, index=False)
    labeling_dataframe.to_csv(for_labeling_csv, index=False)

    print(f"\nMain Preserved Dataset: {main_csv} ({len(output_dataframe)} clauses, {len(output_dataframe.columns)} columns)")
    print(f"Curated Dataset for Compliance Labeling: {for_labeling_csv} ({len(labeling_dataframe)} clauses, {len(labeling_dataframe.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())