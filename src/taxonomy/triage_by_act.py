"""
Production-grade, non-destructive taxonomy triage by source Act.

Purpose
-------
Split draft taxonomy candidates into per-Act review summaries and audit
provenance against the project's target composition.

This script does NOT finalize, delete, merge, or legally approve entries. It
creates an auditable review workspace. Every input taxonomy candidate appears
in exactly one primary output bucket, including unresolved and multi-Act
candidates.

The primary Act assignment is resolved from:
    taxonomy entry -> source_clause_ids -> filtered government clause dataset

Target ranges (Balanced Multi-Act Benchmark)
--------------------------------------------
DPDP_ACT_2023:             18-20
DPDP_RULES_2025:           14-16
IT_ACT_2000:                8-10
RTI_ACT_2005:               0-0
TRAI_ACT_1997:              0-0
AERA_ACT_2008:              0-0
DISASTER_MGMT_ACT_2005:     0-0
PATENTS_ACT_1970:           0-0

Input defaults
--------------
    datasets/GovernmentActs/government_clauses_filtered.csv
    corpus/taxonomy_generation/taxonomy_candidates_normalized.json

Output defaults
---------------
    corpus/taxonomy_generation/triage_by_act/

Run
---
python src/taxonomy/triage_by_act.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CLAUSES = Path("datasets/GovernmentActs/government_clauses_filtered.csv")
DEFAULT_TAXONOMY = Path("corpus/taxonomy_generation/taxonomy_candidates_normalized.json")
DEFAULT_OUTPUT = Path("corpus/taxonomy_generation/triage_by_act")

TARGET_COUNTS: dict[str, tuple[int, int]] = {
    "DPDP_ACT_2023": (18, 20),
    "DPDP_RULES_2025": (14, 16),
    "IT_ACT_2000": (8, 10),
    "RTI_ACT_2005": (0, 0),
    "TRAI_ACT_1997": (0, 0),
    "AERA_ACT_2008": (0, 0),
    "DISASTER_MGMT_ACT_2005": (0, 0),
    "PATENTS_ACT_1970": (0, 0),
}

ACT_NOTES: dict[str, str] = {
    "DPDP_ACT_2023": "Primary substantive law; should provide the majority of the final taxonomy.",
    "DPDP_RULES_2025": "Operational detail; should provide the second-largest substantive contribution.",
    "IT_ACT_2000": "Retain only genuinely policy-checkable confidentiality/security provisions; exclude surveillance/blocking content.",
    "RTI_ACT_2005": "Government transparency framework; likely reviewed and not applicable to private social-media policy unless a specific checkable connection is established.",
    "TRAI_ACT_1997": "Institutional telecom regulation; likely little or no user-policy-checkable content.",
    "AERA_ACT_2008": "Airport tariff regulation; likely little or no user-policy-checkable content.",
    "DISASTER_MGMT_ACT_2005": "Review for a narrow emergency/legitimate-use processing connection.",
    "PATENTS_ACT_1970": "Review only for a specific DPDP-connected legal-privilege or cross-reference obligation.",
}

GOVERNMENT_FACING_SIGNALS = {
    "interception": "interception/surveillance duty",
    "monitoring": "monitoring duty",
    "decryption": "decryption/technical-assistance duty",
    "traffic data": "traffic-data assistance duty",
    "technical assistance": "technical-assistance duty",
}

# Requirement-level signals for policy-checkability review. These are broader
# than privacy policy language because the project includes many policy types.
POLICY_CHECKABILITY_SIGNALS = {
    "personal data", "personal information", "data", "information", "consent",
    "notice", "purpose", "collect", "process", "use", "share", "disclose",
    "transfer", "retain", "retention", "delete", "erase", "security", "breach",
    "confidential", "child", "minor", "rights", "access", "correction", "grievance",
    "complaint", "contact", "account", "message", "content", "payment", "merchant",
    "business", "cookie", "device", "identifier", "storage", "encryption", "protect",
    "copyright", "license", "guidelines", "suspend", "terminate", "moderation",
}

# Strong signals for administrative or institutional text. These are flags only.
ADMINISTRATIVE_SIGNALS = {
    "pension", "gratuity", "sitting fee", "pay matrix", "conditions of service",
    "appellate tribunal", "chairperson", "board member", "gazette of india",
    "registered no", "appointment of officers", "staff of the authority",
}


@dataclass(frozen=True)
class Config:
    clauses_path: Path
    taxonomy_path: Path
    output_dir: Path
    duplicate_similarity_threshold: float
    fail_on_unresolved: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage draft taxonomy candidates by source Act.")
    parser.add_argument("--clauses", type=Path, default=DEFAULT_CLAUSES)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--duplicate-similarity-threshold",
        type=float,
        default=0.85,
        help="Token Jaccard threshold used only to flag possible duplicate requirements.",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Return exit code 2 if any taxonomy candidate has unresolved provenance.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def canonical_text(value: Any) -> str:
    text = normalize_text(value).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", canonical_text(value)))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def stable_id(value: Any, prefix: str) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_clauses(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clause file not found: {path}")
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    for required in ["clause_id", "source_act"]:
        if required not in dataframe.columns:
            raise ValueError(f"Clause CSV must contain {required}")
    dataframe["clause_id"] = dataframe["clause_id"].astype(str)
    dataframe["source_act"] = dataframe["source_act"].astype(str)
    duplicate_ids = dataframe[dataframe["clause_id"].duplicated(keep=False)]["clause_id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Clause CSV contains duplicate clause IDs: {sorted(set(duplicate_ids))[:10]}")
    return dataframe


def load_taxonomy(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    if not isinstance(payload, list):
        raise ValueError("Taxonomy JSON must be a list or contain a records list")
    return [dict(item) for item in payload if isinstance(item, dict)]


def parse_source_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        # Fallback for simple comma-separated provenance fields.
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(value).strip()]


def resolve_act(source_ids: list[str], clause_lookup: dict[str, dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    source_acts = sorted({str(clause_lookup[cid].get("source_act", "")).strip() for cid in source_ids if cid in clause_lookup})
    missing_ids = sorted(set(source_ids) - set(clause_lookup))
    source_acts = [act for act in source_acts if act]
    if len(source_acts) == 1:
        resolved = source_acts[0]
    elif len(source_acts) > 1:
        resolved = "MULTI_ACT__" + "__".join(source_acts)
    else:
        resolved = "UNRESOLVED"
    return resolved, source_acts, missing_ids


def target_status(act: str, count: int) -> str:
    target = TARGET_COUNTS.get(act)
    if target is None:
        return "NO_TARGET_DEFINED"
    low, high = target
    if count < low:
        return "BELOW_TARGET"
    if count > high:
        return "ABOVE_TARGET"
    return "WITHIN_TARGET"


def signal_matches(text: str, signals: dict[str, str] | set[str]) -> list[str]:
    lowered = canonical_text(text)
    if isinstance(signals, dict):
        return [reason for phrase, reason in signals.items() if phrase in lowered]
    return sorted(term for term in signals if term in lowered)


def review_flags(entry: dict[str, Any], resolved_act: str, source_acts: list[str], missing_ids: list[str]) -> dict[str, Any]:
    requirement = normalize_text(entry.get("requirement", ""))
    category = normalize_text(entry.get("category", ""))
    source = normalize_text(entry.get("source", ""))
    checkable_test = normalize_text(entry.get("checkable_test", ""))

    government_facing_reasons: list[str] = []
    if category == "Intermediary/Platform Liability":
        government_facing_reasons = signal_matches(requirement, GOVERNMENT_FACING_SIGNALS)

    checkability_hits = signal_matches(
        f"{requirement} {checkable_test} {source}",
        POLICY_CHECKABILITY_SIGNALS,
    )
    administrative_hits = signal_matches(requirement, ADMINISTRATIVE_SIGNALS)

    flags: list[str] = []
    if not requirement:
        flags.append("missing_requirement")
    if not category:
        flags.append("missing_category")
    if not source:
        flags.append("missing_source_citation")
    if not checkable_test:
        flags.append("missing_checkable_test")
    if missing_ids:
        flags.append("missing_source_clause_ids")
    if not source_acts:
        flags.append("no_resolved_source_act")
    if len(source_acts) > 1:
        flags.append("multi_act_provenance")
    if category == "Intermediary/Platform Liability" and government_facing_reasons:
        flags.append("likely_government_facing")
    if not checkability_hits:
        flags.append("low_policy_checkability_signal")
    if administrative_hits:
        flags.append("possible_institutional_or_administrative_content")

    if "likely_government_facing" in flags:
        recommended_action = "REVIEW_FOR_REMOVAL_OR_RECLASSIFICATION"
    elif "multi_act_provenance" in flags or "missing_source_clause_ids" in flags:
        recommended_action = "REVIEW_PROVENANCE"
    elif "low_policy_checkability_signal" in flags:
        recommended_action = "REVIEW_APPLICABILITY"
    elif "possible_institutional_or_administrative_content" in flags:
        recommended_action = "REVIEW_SCOPE"
    else:
        recommended_action = "LEGAL_REVIEW"

    return {
        "review_flags": flags,
        "likely_government_facing": bool(government_facing_reasons),
        "government_facing_reasons": government_facing_reasons,
        "policy_checkability_signal_count": len(checkability_hits),
        "policy_checkability_signal_terms": checkability_hits,
        "administrative_signal_terms": administrative_hits,
        "recommended_action": recommended_action,
        "resolved_act": resolved_act,
    }


def assign_requirement_duplicate_flags(entries: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    for entry in entries:
        entry["_requirement_tokens"] = tokens(entry.get("requirement", ""))
        entry["requirement_canonical_hash"] = hashlib.sha256(
            canonical_text(entry.get("requirement", "")).encode("utf-8")
        ).hexdigest()

    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        exact_groups[entry["requirement_canonical_hash"]].append(index)

    exact_group_counter = 0
    for indices in exact_groups.values():
        if len(indices) > 1:
            exact_group_counter += 1
            group_id = f"REQ-EXACT-{exact_group_counter:05d}"
            for index in indices:
                entries[index]["requirement_exact_duplicate_group"] = group_id
        else:
            entries[indices[0]]["requirement_exact_duplicate_group"] = ""

    # Greedy near-duplicate grouping within the same category. This flags
    # candidates for manual consolidation; it never merges or removes them.
    groups: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        best_group: dict[str, Any] | None = None
        best_score = 0.0
        for group in groups:
            if group["category"] != entry.get("category"):
                continue
            score = jaccard(entry["_requirement_tokens"], group["representative_tokens"])
            if score >= threshold and score > best_score:
                best_group = group
                best_score = score
        if best_group is None:
            best_group = {
                "id": f"REQ-NEAR-{len(groups) + 1:05d}",
                "category": entry.get("category", ""),
                "representative_tokens": entry["_requirement_tokens"],
                "members": [],
            }
            groups.append(best_group)
            best_score = 1.0
        best_group["members"].append(index)
        entry["requirement_near_duplicate_similarity"] = round(best_score, 6)

    for group in groups:
        group_id = group["id"] if len(group["members"]) > 1 else ""
        for index in group["members"]:
            entries[index]["requirement_near_duplicate_group"] = group_id

    for entry in entries:
        entry.pop("_requirement_tokens", None)
    return entries


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return cleaned or "UNNAMED"


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.duplicate_similarity_threshold <= 1.0:
        raise ValueError("--duplicate-similarity-threshold must be between 0 and 1")

    config = Config(
        clauses_path=args.clauses,
        taxonomy_path=args.taxonomy,
        output_dir=args.output_dir,
        duplicate_similarity_threshold=args.duplicate_similarity_threshold,
        fail_on_unresolved=args.fail_on_unresolved,
    )
    run_timestamp = utc_now_iso()

    clauses_df = load_clauses(config.clauses_path)
    entries = load_taxonomy(config.taxonomy_path)
    clause_lookup = clauses_df.set_index("clause_id", drop=False).to_dict(orient="index")

    if not entries:
        raise ValueError("Taxonomy input contains no candidate entries")

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved_entries: list[dict[str, Any]] = []
    invalid_provenance: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []

    for index, original_entry in enumerate(entries, start=1):
        entry = dict(original_entry)
        entry.setdefault("candidate_id", f"UNID-{index:06d}")
        source_ids = parse_source_ids(entry.get("source_clause_ids", []))
        resolved_act, source_acts, missing_ids = resolve_act(source_ids, clause_lookup)
        flags = review_flags(entry, resolved_act, source_acts, missing_ids)

        entry.update({
            "triage_candidate_id": entry["candidate_id"],
            "source_clause_ids": source_ids,
            "source_acts": source_acts,
            "missing_source_clause_ids": missing_ids,
            "source_provenance_status": (
                "RESOLVED_SINGLE_ACT" if len(source_acts) == 1 and not missing_ids
                else "RESOLVED_MULTI_ACT" if len(source_acts) > 1 and not missing_ids
                else "PARTIAL_PROVENANCE" if source_acts and missing_ids
                else "UNRESOLVED"
            ),
            "triage_run_timestamp_utc": run_timestamp,
            **flags,
        })

        if missing_ids or not source_acts:
            invalid_provenance.append(entry)
        if resolved_act == "UNRESOLVED":
            unresolved_entries.append(entry)
        by_bucket[resolved_act].append(entry)
        processed.append(entry)

    processed = assign_requirement_duplicate_flags(
        processed,
        threshold=config.duplicate_similarity_threshold,
    )

    # Rebuild buckets after duplicate metadata has been added.
    by_bucket = defaultdict(list)
    for entry in processed:
        by_bucket[entry["resolved_act"]].append(entry)

    # Add target status to every known Act bucket. Multi-Act buckets are not
    # silently counted as a single known Act.
    bucket_summaries: list[dict[str, Any]] = []
    for act in sorted(by_bucket):
        count = len(by_bucket[act])
        target = TARGET_COUNTS.get(act)
        bucket_summaries.append({
            "act": act,
            "count": count,
            "target_min": target[0] if target else None,
            "target_max": target[1] if target else None,
            "target_status": target_status(act, count),
            "rationale": ACT_NOTES.get(act, "No target configured; requires manual review."),
            "government_facing_count": sum(1 for item in by_bucket[act] if item.get("likely_government_facing")),
            "policy_checkability_review_count": sum(1 for item in by_bucket[act] if "low_policy_checkability_signal" in item.get("review_flags", [])),
        })

    known_counts = {
        act: len(by_bucket.get(act, []))
        for act in TARGET_COUNTS
    }
    total_known = sum(known_counts.values())
    dpdp_count = known_counts.get("DPDP_ACT_2023", 0) + known_counts.get("DPDP_RULES_2025", 0)
    dpdp_share = round(dpdp_count / total_known * 100, 3) if total_known else 0.0

    summary = {
        "run_timestamp_utc": run_timestamp,
        "clauses_input": str(config.clauses_path),
        "taxonomy_input": str(config.taxonomy_path),
        "taxonomy_candidate_count": len(entries),
        "processed_candidate_count": len(processed),
        "known_act_candidate_count": total_known,
        "unresolved_candidate_count": len(unresolved_entries),
        "partial_or_invalid_provenance_count": len(invalid_provenance),
        "multi_act_bucket_count": sum(1 for act in by_bucket if act.startswith("MULTI_ACT__")),
        "dpdp_act_plus_rules_count": dpdp_count,
        "dpdp_act_plus_rules_share_percent": dpdp_share,
        "target_composition_note": "Target ranges are review guidance, not automatic deletion rules.",
        "target_counts": {act: {"min": low, "max": high} for act, (low, high) in TARGET_COUNTS.items()},
        "bucket_summaries": bucket_summaries,
        "intermediary_platform_liability_total": sum(
            1 for item in processed if item.get("category") == "Intermediary/Platform Liability"
        ),
        "intermediary_likely_government_facing": sum(
            1 for item in processed if item.get("likely_government_facing")
        ),
        "exact_requirement_duplicate_rows": sum(
            1 for item in processed if item.get("requirement_exact_duplicate_group")
        ),
        "near_requirement_duplicate_rows": sum(
            1 for item in processed if item.get("requirement_near_duplicate_group")
        ),
        "taxonomy_status": "DRAFT_TRIAGE_PENDING_LEGAL_REVIEW",
        "non_destructive_contract": "Every input taxonomy candidate is retained in exactly one output bucket.",
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("Dry run complete; no files were written.")
        return 2 if config.fail_on_unresolved and unresolved_entries else 0

    config.output_dir.mkdir(parents=True, exist_ok=True)

# Write one file per resolved bucket, including MULTI_ACT and UNRESOLVED.
    for act, act_entries in sorted(by_bucket.items()):
        write_json(config.output_dir / f"{safe_filename(act)}.json", act_entries)

    # Explicit reports for review rather than relying on console output.
    write_json(config.output_dir / "UNRESOLVED.json", unresolved_entries)
    write_json(config.output_dir / "PROVENANCE_ISSUES.json", invalid_provenance)
    write_json(
        config.output_dir / "INTERMEDIARY_GOVERNMENT_FACING_REVIEW.json",
        [item for item in processed if item.get("likely_government_facing")],
    )
    write_json(
        config.output_dir / "LOW_POLICY_CHECKABILITY_REVIEW.json",
        [item for item in processed if "low_policy_checkability_signal" in item.get("review_flags", [])],
    )
    write_json(
        config.output_dir / "REQUIREMENT_DUPLICATE_REVIEW.json",
        [
            item for item in processed
            if item.get("requirement_exact_duplicate_group")
            or item.get("requirement_near_duplicate_group")
        ],
    )
    write_json(config.output_dir / "ALL_TRIAGED_CANDIDATES.json", processed)
    write_json(config.output_dir / "ACT_TARGET_SUMMARY.json", bucket_summaries)
    write_json(config.output_dir / "TRIAGE_RUN_SUMMARY.json", summary)

    # Human-readable CSV for sorting/filtering without changing the JSON source.
    flat_rows: list[dict[str, Any]] = []
    for item in processed:
        row = dict(item)
        for field in ["source_clause_ids", "source_acts", "missing_source_clause_ids", "review_flags", "government_facing_reasons", "policy_checkability_signal_terms", "administrative_signal_terms"]:
            if field in row and not isinstance(row[field], str):
                row[field] = json.dumps(row[field], ensure_ascii=False)
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(config.output_dir / "ALL_TRIAGED_CANDIDATES.csv", index=False)

    # Console table required for quick DPDP-first triage.
    print(f"Loaded {len(entries)} taxonomy candidates")
    print(f"Resolved clauses from {len(clauses_df)} cleaned government clauses")
    print(f"\n{'Act':35s} {'Count':>7s} {'Target':>10s} {'Status':>18s}")
    print("-" * 76)
    for item in bucket_summaries:
        target = (
            "n/a" if item["target_min"] is None
            else f"{item['target_min']}-{item['target_max']}"
        )
        print(f"{item['act'][:35]:35s} {item['count']:7d} {target:>10s} {item['target_status']:>18s}")

    print(f"\nDPDP Act + Rules: {dpdp_count} entries ({dpdp_share:.2f}% of known-Act entries)")
    print(
        "Intermediary/Platform Liability: "
        f"{summary['intermediary_platform_liability_total']} total; "
        f"{summary['intermediary_likely_government_facing']} likely government-facing"
    )
    print(f"\nPer-Act files and review reports written to: {config.output_dir}")
    print("No entries were deleted, merged, or legally approved.")

    if config.fail_on_unresolved and unresolved_entries:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
