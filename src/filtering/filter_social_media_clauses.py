"""
Production-grade, recall-first triage for social-media policy clauses.

This is a NON-DESTRUCTIVE triage stage. It is intentionally not a hard
relevance filter because social-media platforms publish different and changing
sets of documents, including but not limited to:

- Privacy Policies
- Terms of Service
- Cookies Policies
- Data Processing Terms
- Data Security Terms
- Data Transfer Addenda
- Business, Merchant, and Payments Terms
- Messaging and Channels Guidelines
- Intellectual Property Policies
- Community or safety policies
- Future document types not known at implementation time

The script preserves every input record. It never deletes, excludes, or
automatically down-weights a clause based on filename, document type, absence
of a keyword, or an administrative/boilerplate signal.

It adds transparent metadata for downstream retrieval, annotation, and manual
review:

- quality flags;
- narrow boilerplate signals;
- broad document-agnostic concept signals;
- normative-language signals;
- review priority;
- exact and conservative near-duplicate groups;
- stable canonical text hash;
- machine-checkable retention fields.

The relevance signals are only prioritization hints. A clause with a relevance
score of zero remains eligible for taxonomy matching and human annotation.

Input
-----
Default:
    datasets/SocialMediaPolicies/social_media_clauses.json

JSON must be a list of records or an object containing a `records` list.
CSV is also supported. The input must contain `clause_text`.

Usage
-----
    python src/filtering/filter_social_media_clauses.py

Custom paths:
    python src/filtering/filter_social_media_clauses.py \
        --input datasets/SocialMediaPolicies/social_media_clauses.json \
        --output-dir datasets/SocialMediaPolicies/filtering

Dry run:
    python src/filtering/filter_social_media_clauses.py --dry-run

Install
-------
    pip install pandas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("datasets/SocialMediaPolicies/social_media_clauses.json")
DEFAULT_OUTPUT_DIR = Path("datasets/SocialMediaPolicies")

# These patterns match complete artifact blocks only. They do not match a
# phrase embedded inside a substantive legal paragraph.
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

# Broad concepts cover more than privacy-specific terminology. The terms are
# used as triage signals only and are deliberately not used as inclusion rules.
RELEVANCE_TERMS: dict[str, tuple[str, ...]] = {
    "data_and_information": (
        "personal data", "personal information", "user data", "your information",
        "information", "data", "identifier", "account", "device", "content",
        "message", "communication", "metadata",
    ),
    "collection_and_use": (
        "collect", "obtain", "receive", "use", "process", "purpose", "access",
        "store", "storage", "retain", "retention", "delete", "erase", "remove",
    ),
    "consent_and_choice": (
        "consent", "opt out", "opt-out", "opt in", "opt-in", "permission",
        "withdraw", "choice", "preference", "allow", "deny",
    ),
    "rights_and_requests": (
        "your rights", "access your", "correct", "correction", "rectification",
        "download your data", "request", "complaint", "grievance", "appeal",
    ),
    "security_and_incidents": (
        "security", "secure", "encrypt", "encryption", "protect", "safeguard",
        "confidential", "breach", "incident", "unauthorized", "integrity",
        "availability", "authentication",
    ),
    "children_and_age": (
        "child", "children", "minor", "parent", "guardian", "age", "under 13",
        "under 16", "under 18", "young person",
    ),
    "sharing_and_disclosure": (
        "share", "sharing", "third party", "third-party", "disclose", "disclosure",
        "affiliate", "partner", "vendor", "service provider", "recipient",
    ),
    "transfer_and_location": (
        "transfer", "transmit", "cross-border", "outside india", "overseas",
        "location", "country", "jurisdiction", "data center", "data centre",
    ),
    "governance_and_accountability": (
        "responsible", "responsibility", "officer", "contact us", "authority",
        "audit", "compliance", "policy", "terms", "notice", "controller",
        "processor", "fiduciary", "representative",
    ),
    "platform_rights_and_restrictions": (
        "suspend", "terminate", "restrict", "remove content", "moderation",
        "community", "guidelines", "intellectual property", "copyright",
        "license", "enforcement", "prohibited",
    ),
    "payments_and_business": (
        "payment", "transaction", "merchant", "business", "customer", "invoice",
        "billing", "financial", "purchase", "refund",
    ),
}

NORMATIVE_TERMS = (
    "shall", "must", "may", "may not", "will", "required", "prohibited",
    "responsible for", "liable", "unless", "except", "subject to", "provided that",
    "you agree", "we may", "we will",
)


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path
    near_duplicate_threshold: float
    high_relevance_threshold: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-destructive, recall-first triage of social-media clauses."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.98,
        help="Jaccard threshold for conservative near-duplicate groups. Duplicates remain retained.",
    )
    parser.add_argument(
        "--high-relevance-threshold",
        type=int,
        default=2,
        help="Minimum concept-category count for high-priority review.",
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
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


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
    if pd.isna(value):
        return ""
    return str(value)


def boilerplate_signal(text: str) -> tuple[bool, str]:
    candidate = display_text(text)
    for pattern, reason in BOILERPLATE_REGEX:
        if pattern.fullmatch(candidate):
            return True, reason
    return False, ""


def concept_signal(text: str, section_heading: str) -> tuple[int, list[str], str, bool]:
    combined = canonical_text(f"{section_heading} {text}")
    categories: list[str] = []
    for category, terms in RELEVANCE_TERMS.items():
        if any(term in combined for term in terms):
            categories.append(category)

    normative = any(term in combined for term in NORMATIVE_TERMS)
    score = len(categories)
    if normative:
        score += 1

    if score >= 4:
        level = "high"
    elif score >= 2:
        level = "medium"
    elif score == 1:
        level = "low"
    else:
        level = "unknown"
    return score, categories, level, normative


def quality_flags(text: str, row: pd.Series) -> list[str]:
    flags: list[str] = []
    cleaned = display_text(text)
    if not cleaned:
        flags.append("empty_clause_text")
    if len(cleaned) < 10:
        flags.append("very_short_text")
    if len(cleaned) > 5000:
        flags.append("very_long_text")
    if "source_document" in row.index and not row_value(row, "source_document").strip():
        flags.append("missing_source_document")
    if "source_file" in row.index and not row_value(row, "source_file").strip():
        flags.append("missing_source_file")
    if "app_name" in row.index and not row_value(row, "app_name").strip():
        flags.append("missing_app_name")
    if "page_start" in row.index and not row_value(row, "page_start").strip():
        flags.append("missing_page_start")
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
    """Create conservative duplicate metadata without removing records.

    This greedy implementation is suitable for a modest academic corpus. For
    very large corpora, replace it with MinHash/LSH, but preserve the output
    contract and the non-destructive behavior.
    """
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
        dataframe = load_input(config.input_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

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
    priorities: list[str] = []

    for _, row in dataframe.iterrows():
        text = display_text(row.get("clause_text", ""))
        heading = row_value(row, "section_heading")
        is_boilerplate, boilerplate_reason = boilerplate_signal(text)
        score, categories, level, normative = concept_signal(text, heading)
        flags = quality_flags(text, row)

        boilerplate_values.append(is_boilerplate)
        boilerplate_reasons.append(boilerplate_reason)
        relevance_scores.append(score)
        relevance_categories.append(categories)
        relevance_levels.append(level)
        normative_values.append(normative)
        quality_values.append(flags)

        # Priority is only a review-order hint. It does not affect eligibility.
        if flags:
            priority = "quality_check"
        elif is_boilerplate:
            priority = "boilerplate_review"
        elif score >= config.high_relevance_threshold:
            priority = "high"
        elif score == 1:
            priority = "medium"
        else:
            priority = "low"
        priorities.append(priority)

    dataframe["likely_boilerplate"] = boilerplate_values
    dataframe["boilerplate_reason"] = boilerplate_reasons
    dataframe["relevance_score"] = relevance_scores
    dataframe["relevance_categories"] = [json.dumps(value, ensure_ascii=False) for value in relevance_categories]
    dataframe["relevance_level"] = relevance_levels
    dataframe["contains_normative_language"] = normative_values
    dataframe["quality_flags"] = [json.dumps(value, ensure_ascii=False) for value in quality_values]
    dataframe["review_priority"] = priorities
    dataframe["exact_duplicate_group"] = exact_duplicate_groups(dataframe)

    near_groups, near_similarity = near_duplicate_groups(
        dataframe,
        config.near_duplicate_threshold,
    )
    dataframe["near_duplicate_group"] = near_groups
    dataframe["near_duplicate_similarity"] = near_similarity

    # Explicit retention contract. These fields prevent downstream code from
    # accidentally interpreting triage metadata as a hard filter.
    dataframe["filter_decision"] = "RETAIN"
    dataframe["retention_reason"] = "recall_first_no_deletion"
    dataframe["eligible_for_taxonomy_matching"] = True
    dataframe["eligible_for_human_annotation"] = True
    dataframe["filter_run_timestamp_utc"] = run_timestamp

    summary = {
        "run_timestamp_utc": run_timestamp,
        "input_path": str(config.input_path),
        "input_rows": int(len(dataframe)),
        "output_rows": int(len(dataframe)),
        "rows_deleted": 0,
        "rows_excluded": 0,
        "retention_contract": "Every input clause is retained in the main output",
        "document_types_used_as_hard_filters": [],
        "filename_used_as_hard_filter": False,
        "boilerplate_counts": dataframe["likely_boilerplate"].value_counts().to_dict(),
        "review_priority_counts": dataframe["review_priority"].value_counts().to_dict(),
        "relevance_level_counts": dataframe["relevance_level"].value_counts().to_dict(),
        "rows_with_quality_flags": int((dataframe["quality_flags"] != "[]").sum()),
        "rows_in_exact_duplicate_groups": int((dataframe["exact_duplicate_group"] != "").sum()),
        "rows_in_near_duplicate_groups": int((dataframe["near_duplicate_group"] != "").sum()),
        "near_duplicate_threshold": config.near_duplicate_threshold,
        "original_columns": original_columns,
        "added_columns": [column for column in dataframe.columns if column not in original_columns],
        "warning": (
            "Boilerplate and relevance fields are triage metadata only. "
            "They must not be used as destructive filters without human/legal review."
        ),
    }

    print("=== Recall-first social-media clause triage ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run complete; no output files were written.")
        return 0

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_dataframe = dataframe.drop(columns=["_tokens"], errors="ignore")

    main_json = config.output_dir / "social_media_clauses_flagged.json"
    main_csv = config.output_dir / "social_media_clauses_flagged.csv"
    review_json = config.output_dir / "social_media_clause_review_queue.json"
    duplicate_json = config.output_dir / "social_media_clause_duplicate_report.json"
    summary_json = config.output_dir / "social_media_filter_summary.json"

    write_json(main_json, output_dataframe.to_dict(orient="records"))
    output_dataframe.to_csv(main_csv, index=False)

    review_columns = [
        column for column in [
            "clause_id", "app_name", "source_document", "source_file", "page_start",
            "section_heading", "clause_text", "likely_boilerplate", "boilerplate_reason",
            "relevance_score", "relevance_categories", "relevance_level",
            "contains_normative_language", "quality_flags", "review_priority",
            "exact_duplicate_group", "near_duplicate_group", "filter_decision",
        ] if column in output_dataframe.columns
    ]
    review_dataframe = output_dataframe[review_columns].copy()
    review_dataframe = review_dataframe[
        (review_dataframe["review_priority"] != "low")
        | (review_dataframe["quality_flags"] != "[]")
    ]
    write_json(review_json, review_dataframe.to_dict(orient="records"))

    duplicate_columns = [
        column for column in [
            "clause_id", "app_name", "source_document", "source_file", "page_start",
            "clause_text", "_canonical_hash", "exact_duplicate_group",
            "near_duplicate_group", "near_duplicate_similarity",
        ] if column in output_dataframe.columns
    ]
    duplicate_dataframe = output_dataframe[duplicate_columns].copy()
    duplicate_dataframe = duplicate_dataframe[
        (duplicate_dataframe["exact_duplicate_group"] != "")
        | (duplicate_dataframe["near_duplicate_group"] != "")
    ]
    write_json(duplicate_json, duplicate_dataframe.to_dict(orient="records"))
    write_json(summary_json, summary)

    print("\nSaved:")
    for path in [main_json, main_csv, review_json, duplicate_json, summary_json]:
        print(f"  {path}")
    print(
        f"\nContract check: input rows = {len(dataframe)}, "
        f"output rows = {len(output_dataframe)}, deleted = 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
