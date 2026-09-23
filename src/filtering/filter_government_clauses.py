"""
Recall-first filtering and triage for government legal clauses.

Purpose
-------
This is NOT a destructive relevance filter. The project includes many kinds of
social-media governance documents, not only privacy policies. A legal clause
that appears administrative, institutional, technical, or unrelated today may
still map to a future platform policy document or to a legal-taxonomy entry.

Therefore this script:

1. Preserves every input clause in every output dataset.
2. Never deletes, filters out, or automatically down-weights a clause.
3. Adds transparent triage metadata for later retrieval, annotation, or review.
4. Detects exact/near duplicates without removing them.
5. Flags possible administrative material conservatively, but does not treat
   the flag as a final legal decision.
6. Creates a review queue for high-uncertainty or low-quality records.
7. Produces summary reports showing exactly what was retained and why.

The downstream taxonomy and semantic-matching stages must decide whether a
clause is substantively relevant. This script must run before those stages and
must not be used as a hard training-data filter.

Expected input
--------------
By default:
    datasets/GovernmentActs/government_clauses.json

The input may be JSON records or a CSV file. Each record should contain at
least `clause_text`; `source_act`, `source_file`, and `section_heading` are
recommended but optional.

Usage
-----
    python src/filtering/filter_government_clauses.py

Custom paths:
    python src/filtering/filter_government_clauses.py \
        --input datasets/GovernmentActs/government_clauses.json \
        --output-dir datasets/GovernmentActs/filtering \
        --similarity-threshold 0.98

The output keeps all input rows and adds fields such as:
    filter_decision = "RETAIN"
    retention_reason = "recall_first_no_deletion"
    administrative_signal
    administrative_reasons
    relevance_signal
    review_priority
    exact_duplicate_group
    near_duplicate_group
    quality_flags

Install:
    pip install pandas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_INPUT = Path("datasets/GovernmentActs/government_clauses.json")
DEFAULT_OUTPUT_DIR = Path("datasets/GovernmentActs")

# These patterns are deliberately narrow. They are signals for human review,
# never deletion rules. They avoid assuming that a clause must come from a
# privacy policy or that a document type is irrelevant.
ADMINISTRATIVE_SIGNALS: list[tuple[str, str, int]] = [
    (r"\bpension\b", "personnel pension language", 1),
    (r"\bgratuity\b", "personnel gratuity language", 1),
    (r"\bcasual leave\b|\bleave travel concession\b", "personnel leave language", 1),
    (r"\bpay matrix\b|\bsitting fee\b", "official compensation language", 1),
    (r"\bmedical assistance\b.*\b(officer|member|employee|staff)\b", "personnel benefit language", 1),
    (r"\b(salary|allowance|tenure|reappointment)\b.*\bchairperson\b", "institutional service-condition language", 1),
    (r"\b(board member|member of the board)\b.*\b(appointment|removal|tenure)\b", "institutional appointment language", 1),
    (r"\bcivil services?\b.*\b(pay|pension|conduct rules?)\b", "civil-service cross-reference", 1),
    (r"\bregistered no\b|\bgazette of india\b", "gazette front-matter signal", 1),
    (r"^\s*be it enacted by parliament\b", "enactment preamble signal", 1),
    (r"^\s*this act may be called\b.*\bcome into force\b", "short-title/commencement signal", 1),
    (r"\bjurisdiction of civil court\b|\bbar of jurisdiction\b", "procedural jurisdiction signal", 1),
]
ADMIN_REGEX = [
    (re.compile(pattern, re.IGNORECASE), reason, weight)
    for pattern, reason, weight in ADMINISTRATIVE_SIGNALS
]

# Broad, document-agnostic legal concepts. These are used only to identify
# clauses that deserve normal or high-priority review. A clause with no term is
# still retained and still eligible for semantic retrieval.
RELEVANCE_TERMS: dict[str, tuple[str, ...]] = {
    "data_and_information": (
        "personal data", "personal information", "sensitive", "information", "data",
        "record", "identifier", "account", "device", "communication",
    ),
    "collection_and_use": (
        "collect", "obtain", "receive", "use", "process", "purpose", "disclose",
        "share", "transfer", "access", "store", "retain", "delete", "erase",
    ),
    "rights_and_choice": (
        "consent", "withdraw", "opt out", "access", "correction", "rectification",
        "grievance", "complaint", "request", "notice", "choice", "permission",
    ),
    "security_and_incident": (
        "security", "safeguard", "protect", "confidential", "encryption", "breach",
        "incident", "unauthorized", "integrity", "availability",
    ),
    "children_and_vulnerable_users": (
        "child", "children", "minor", "parent", "guardian", "age", "under 18",
    ),
    "governance_and_accountability": (
        "fiduciary", "controller", "processor", "responsible", "officer", "audit",
        "compliance", "obligation", "appoint", "contact", "authority", "register",
    ),
    "territorial_and_cross_border": (
        "india", "indian", "outside india", "cross-border", "overseas", "foreign",
        "jurisdiction", "government", "state", "transfer",
    ),
}

HEADING_SIGNALS = (
    "schedule", "chapter", "section", "rule", "regulation", "powers", "duties",
    "obligation", "penalty", "offence", "procedure", "authority", "security",
    "data", "information", "privacy", "confidentiality", "disclosure",
)

CONTROL_WORDS = (
    "shall", "must", "may", "may not", "should", "prohibited", "required",
    "responsible", "liable", "unless", "except", "subject to", "provided that",
)


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path
    similarity_threshold: float
    high_priority_threshold: int
    retain_all: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recall-first, non-destructive triage of government legal clauses."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.98,
        help="Jaccard threshold for near-duplicate grouping; duplicates are retained.",
    )
    parser.add_argument(
        "--high-priority-threshold",
        type=int,
        default=2,
        help="Minimum relevance score for a high-priority review signal.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and analyze input without writing output files.",
    )
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


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def tokenize(text: str) -> set[str]:
    normalized = canonical_text(text)
    return set(re.findall(r"[a-z0-9]+", normalized))


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.casefold() == ".csv":
        dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                payload = payload["records"]
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of records or an object containing 'records'")
        dataframe = pd.DataFrame(payload)
    else:
        raise ValueError("Input must be a .json or .csv file")

    if "clause_text" not in dataframe.columns:
        raise ValueError("Input must contain a 'clause_text' column")
    if dataframe.empty:
        raise ValueError("Input contains zero records")

    # Preserve every original column. Normalize only missing clause text to an
    # empty string so that malformed rows can be retained and reviewed.
    dataframe["clause_text"] = dataframe["clause_text"].fillna("").astype(str)
    return dataframe


def safe_string(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return str(value)


def administrative_signal(clause_text: str, section_heading: str) -> tuple[str, int, list[str]]:
    score = 0
    reasons: list[str] = []
    for pattern, reason, weight in ADMIN_REGEX:
        if pattern.search(clause_text):
            score += weight
            reasons.append(reason)

    heading = canonical_text(section_heading)
    if heading and any(term in heading for term in ("salary", "allowance", "conditions of service", "staff", "pension")):
        # Heading evidence is weak and never sufficient by itself.
        score += 1
        reasons.append("administrative-sounding heading")

    if score == 0:
        return "none", 0, []
    if score == 1:
        return "weak", score, reasons
    return "possible", score, reasons


def relevance_signal(clause_text: str, section_heading: str) -> tuple[int, list[str], str]:
    combined = canonical_text(f"{section_heading} {clause_text}")
    matched_categories: list[str] = []
    score = 0

    for category, terms in RELEVANCE_TERMS.items():
        matched = [term for term in terms if term in combined]
        if matched:
            score += 1
            matched_categories.append(category)

    control_match = any(term in combined for term in CONTROL_WORDS)
    if control_match:
        score += 1
        matched_categories.append("normative_or_conditional_language")

    if score >= 4:
        level = "high"
    elif score >= 2:
        level = "medium"
    elif score == 1:
        level = "low"
    else:
        level = "unknown"
    return score, matched_categories, level


def quality_flags(clause_text: str, row: pd.Series) -> list[str]:
    flags: list[str] = []
    text = display_text(clause_text)
    if not text:
        flags.append("empty_clause_text")
    if len(text) < 10:
        flags.append("very_short_text")
    if len(text) > 5000:
        flags.append("very_long_text")
    if "page_start" in row.index and not str(row.get("page_start", "")).strip():
        flags.append("missing_page_start")
    if "source_file" in row.index and not str(row.get("source_file", "")).strip():
        flags.append("missing_source_file")
    if "source_act" in row.index and not str(row.get("source_act", "")).strip():
        flags.append("missing_source_act")
    return flags


def assign_exact_duplicate_groups(dataframe: pd.DataFrame) -> pd.Series:
    hashes = dataframe["_canonical_hash"]
    counts = hashes.value_counts()
    group_ids: dict[str, str] = {}
    next_group = 1
    output: list[str] = []
    for value in hashes:
        if counts[value] == 1:
            output.append("")
            continue
        if value not in group_ids:
            group_ids[value] = f"EXACT-{next_group:05d}"
            next_group += 1
        output.append(group_ids[value])
    return pd.Series(output, index=dataframe.index, dtype="string")


def assign_near_duplicate_groups(
    dataframe: pd.DataFrame,
    threshold: float,
) -> tuple[pd.Series, dict[str, float]]:
    """Group near duplicates with a conservative greedy comparison.

    This is intended for audit metadata, not deletion. For very large corpora,
    replace this with MinHash/LSH or an embedding index while retaining the
    same non-destructive contract.
    """
    groups: list[dict[str, Any]] = []
    assignments: list[str] = [""] * len(dataframe)
    similarities: dict[str, float] = {}

    for position, tokens in enumerate(dataframe["_tokens"]):
        assigned_group: dict[str, Any] | None = None
        best_similarity = 0.0
        for group in groups:
            similarity = jaccard_similarity(tokens, group["representative_tokens"])
            if similarity >= threshold and similarity > best_similarity:
                assigned_group = group
                best_similarity = similarity

        if assigned_group is None:
            group_id = f"NEAR-{len(groups) + 1:05d}"
            assigned_group = {
                "id": group_id,
                "representative_tokens": tokens,
                "members": [],
            }
            groups.append(assigned_group)
            best_similarity = 1.0

        assigned_group["members"].append(position)
        similarities[str(position)] = round(best_similarity, 6)

    # Assign the group ID to every member only after grouping is complete. A
    # singleton is not useful as a duplicate group, but remains in the
    # similarity metadata for reproducibility.
    for group in groups:
        if len(group["members"]) > 1:
            for position in group["members"]:
                assignments[position] = group["id"]
    return pd.Series(assignments, index=dataframe.index, dtype="string"), similarities


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = Config(
        input_path=args.input,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        high_priority_threshold=args.high_priority_threshold,
    )
    run_timestamp = utc_now_iso()

    try:
        dataframe = load_records(config.input_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    original_columns = list(dataframe.columns)
    dataframe = dataframe.copy()
    dataframe["_canonical_hash"] = dataframe["clause_text"].map(text_hash)
    dataframe["_tokens"] = dataframe["clause_text"].map(tokenize)

    administrative_values: list[str] = []
    administrative_scores: list[int] = []
    administrative_reasons: list[list[str]] = []
    relevance_scores: list[int] = []
    relevance_categories: list[list[str]] = []
    relevance_levels: list[str] = []
    quality_values: list[list[str]] = []
    priorities: list[str] = []

    for _, row in dataframe.iterrows():
        clause_text = display_text(row.get("clause_text", ""))
        heading = safe_string(row, "section_heading")
        admin_level, admin_score, admin_reasons = administrative_signal(clause_text, heading)
        rel_score, rel_categories, rel_level = relevance_signal(clause_text, heading)
        flags = quality_flags(clause_text, row)

        administrative_values.append(admin_level)
        administrative_scores.append(admin_score)
        administrative_reasons.append(admin_reasons)
        relevance_scores.append(rel_score)
        relevance_categories.append(rel_categories)
        relevance_levels.append(rel_level)
        quality_values.append(flags)

        # Priority is only a queue-management hint. It does not change model
        # eligibility or remove the record from any output.
        if flags:
            priority = "quality_review"
        elif rel_score >= config.high_priority_threshold:
            priority = "normal_or_high_relevance_review"
        elif admin_level == "possible":
            priority = "administrative_review_optional"
        else:
            priority = "standard_review"
        priorities.append(priority)

    dataframe["administrative_signal"] = administrative_values
    dataframe["administrative_signal_score"] = administrative_scores
    dataframe["administrative_reasons"] = [json.dumps(value, ensure_ascii=False) for value in administrative_reasons]
    dataframe["relevance_signal_score"] = relevance_scores
    dataframe["relevance_categories"] = [json.dumps(value, ensure_ascii=False) for value in relevance_categories]
    dataframe["relevance_signal_level"] = relevance_levels
    dataframe["quality_flags"] = [json.dumps(value, ensure_ascii=False) for value in quality_values]
    dataframe["review_priority"] = priorities

    dataframe["exact_duplicate_group"] = assign_exact_duplicate_groups(dataframe)
    near_groups, near_similarity = assign_near_duplicate_groups(
        dataframe,
        threshold=config.similarity_threshold,
    )
    dataframe["near_duplicate_group"] = near_groups
    dataframe["near_duplicate_similarity"] = [near_similarity[str(i)] for i in range(len(dataframe))]

    # These columns make the retention contract machine-checkable downstream.
    dataframe["filter_decision"] = "RETAIN"
    dataframe["retention_reason"] = "recall_first_no_deletion"
    dataframe["eligible_for_taxonomy_matching"] = True
    dataframe["eligible_for_human_annotation"] = True
    dataframe["filter_run_timestamp_utc"] = run_timestamp

    exact_duplicate_count = int((dataframe["exact_duplicate_group"] != "").sum())
    near_duplicate_count = int((dataframe["near_duplicate_group"] != "").sum())
    quality_flag_count = int((dataframe["quality_flags"] != "[]").sum())

    summary = {
        "run_timestamp_utc": run_timestamp,
        "input_path": str(config.input_path),
        "input_rows": int(len(dataframe)),
        "output_rows": int(len(dataframe)),
        "rows_deleted": 0,
        "rows_excluded": 0,
        "retention_contract": "Every input clause is retained in the main output",
        "administrative_signal_counts": dataframe["administrative_signal"].value_counts().to_dict(),
        "relevance_signal_counts": dataframe["relevance_signal_level"].value_counts().to_dict(),
        "review_priority_counts": dataframe["review_priority"].value_counts().to_dict(),
        "rows_with_quality_flags": quality_flag_count,
        "rows_in_exact_duplicate_groups": exact_duplicate_count,
        "rows_in_near_duplicate_groups": near_duplicate_count,
        "similarity_threshold": config.similarity_threshold,
        "original_columns": original_columns,
        "added_columns": [column for column in dataframe.columns if column not in original_columns],
        "warning": (
            "Administrative and relevance signals are triage metadata only. "
            "They must not be used as destructive filters without legal review."
        ),
    }

    print("=== Recall-first filtering summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    config.output_dir.mkdir(parents=True, exist_ok=True)

## Retain only the 10 essential audit and context columns
    essential_cols = [
        "clause_id", "source_act", "source_file", "page_start", "page_end",
        "section_heading", "clause_text", "relevance_signal_score",
        "relevance_signal_level", "quality_flags"
    ]
    output_dataframe = dataframe[[c for c in essential_cols if c in dataframe.columns]]
    output_records = output_dataframe.to_dict(orient="records")

    main_json = config.output_dir / "government_clauses_filtered.json"
    main_csv = config.output_dir / "government_clauses_filtered.csv"

    write_json(main_json, output_records)
    output_dataframe.to_csv(main_csv, index=False)

    print("\nSaved:")
    print(f"  {main_json}")
    print(f"  {main_csv}")
    print(
        f"\nContract check: input rows = {len(dataframe)}, "
        f"output rows = {len(output_dataframe)}, columns = {len(output_dataframe.columns)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
