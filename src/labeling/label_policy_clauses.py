"""
Resumable, cache-backed bootstrap labeling of social-media policy clauses.

Labels curated policy clauses against the finalized multi-act compliance taxonomy.
Uses targeted candidate gating to prevent context degradation and eliminate hallucinations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama
import pandas as pd

DEFAULT_CORPUS = Path("datasets/SocialMediaPolicies/social_media_clauses_for_labeling.csv")
DEFAULT_TAXONOMY = Path("corpus/dpdp_taxonomy_final.json")
DEFAULT_OUTPUT = Path("corpus/labeled_clauses_bootstrap.csv")
DEFAULT_CACHE_DIR = Path("corpus/label_cache")
DEFAULT_FAILURES = Path("corpus/labeled_clauses_failures.json")
DEFAULT_RUN_STATE = Path("corpus/labeled_clauses_run_state.json")

PROMPT_VERSION = "policy-labeling-v10-dynamic-gated"
MODEL_DEFAULT = "qwen3:8b"
BATCH_SIZE_DEFAULT = 1
RETRY_MAX_DEFAULT = 2
RETRY_BACKOFF_DEFAULT = 2.0

VERDICTS = {
    "Compliant": 1.0,
    "Partially Compliant": 0.5,
    "Non-Compliant": 0.0,
    "Not Addressed": 0.0,
}

OLLAMA_NUM_PREDICT = 2048
OLLAMA_TEMPERATURE = 0.0
OLLAMA_TIMEOUT_SECONDS = 180.0

# Keyword routing dictionary for high-precision candidate selection
CATEGORY_KEYWORD_MAP = {
    "Children/Vulnerable Groups": [
        "child", "children", "minor", "under 18", "parent", "guardian", "underage",
        "parental consent", "age limit", "age of majority", "detrimental to child"
    ],
    "Data Principal Rights": [
        "access", "download", "export", "portability", "correct", "rectif", "update your information",
        "copy of data", "review your data", "my activity", "takeout", "manage your info"
    ],
    "Data Retention & Erasure": [
        "delet", "eras", "retention", "retain", "retention period", "storage period", "wipe",
        "how long we keep", "remove your account", "deactivate"
    ],
    "Consent & Notice": [
        "consent", "withdraw", "revoke", "opt out", "opt-out", "notice", "purpose", "privacy policy",
        "agree to this", "your choices", "permission"
    ],
    "Grievance Redressal": [
        "grievance", "redress", "officer", "nodal", "complaint", "contact us", "dpo", "dispute",
        "data protection officer", "timeline", "india grievance"
    ],
    "Breach Notification": [
        "breach", "incident", "leak", "unauthorized disclosure", "compromise", "notify users",
        "security incident", "cert-in", "board notification"
    ],
    "Cross-Border Transfer": [
        "transfer", "cross-border", "overseas", "outside india", "international", "adequacy",
        "jurisdiction", "data transfer"
    ],
    "Intermediary/Platform Liability": [
        "takedown", "blocking", "order", "court order", "government direction", "prohibit",
        "unlawful", "due diligence", "remove content", "rule 3", "infringing"
    ],
    "Security Safeguards": [
        "security", "encrypt", "safeguard", "technical measures", "access control", "unauthorized access",
        "log", "logs", "monitor", "vulnerability", "audit", "password", "two-factor", "multi-factor"
    ],
    "Significant Data Fiduciary Obligations": [
        "fiduciary", "processor", "audit", "impact assessment", "dpia", "undertaking"
    ]
}


def build_label_json_schema(clause_ids: list[str], candidate_taxonomy_ids: set[str]) -> dict[str, Any]:
    valid_ids = ["NONE", *sorted(candidate_taxonomy_ids)]
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string", "enum": list(clause_ids)},
                "best_match_taxonomy_id": {"type": "string", "enum": valid_ids},
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "justification": {"type": "string"},
            },
            "required": ["clause_id", "best_match_taxonomy_id", "verdict", "justification"],
        },
    }


SYSTEM_PROMPT = """You are a senior data protection compliance auditor auditing privacy policies against Indian regulations (DPDP Act 2023, DPDP Rules 2025, IT Act 2000).

AUDIT PRINCIPLES:
1. SEMANTIC MATCH FIRST:
   - Match a rule ONLY if the clause directly concerns the legal subject matter of that rule (e.g., child consent, data deletion, access rights, grievance officer, encryption).
   - If a clause discusses general features, advertising settings, or terms of use without fulfilling the statutory test, return:
     "best_match_taxonomy_id": "NONE", "verdict": "Not Addressed".

2. STRICT NEGATIVE LOGIC (NO INVERTED LOGIC):
   - Stating that a platform DOES NOT do something (e.g., "We do not keep server logs") is NOT compliance with a rule requiring log retention. That is either "Not Addressed" or "Non-Compliant".

3. VERDICT DEFINITIONS:
   - "Compliant": The clause fully satisfies the statutory requirement with actionable mechanism, contact, or timelines.
   - "Partially Compliant": The clause addresses the statutory topic (e.g., permits data deletion or rights requests) but omits specific statutory details or timelines.
   - "Non-Compliant": The clause directly contradicts or waives statutory rights.
   - "Not Addressed": Used ONLY when "best_match_taxonomy_id" is "NONE".

Return ONLY a valid JSON array."""


@dataclass(frozen=True)
class Config:
    corpus: Path
    taxonomy: Path
    output: Path
    cache_dir: Path
    failures: Path
    run_state: Path
    model: str
    batch_size: int
    retry_max: int
    retry_backoff: float
    force_new_run: bool


class LabelingError(Exception):
    pass


class ValidationError(LabelingError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap labeling with dynamic candidate gating.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--run-state", type=Path, default=DEFAULT_RUN_STATE)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--retry-max", type=int, default=RETRY_MAX_DEFAULT)
    parser.add_argument("--retry-backoff", type=float, default=RETRY_BACKOFF_DEFAULT)
    parser.add_argument("--force-new-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_taxonomy(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]

    records = []
    seen = set()
    for item in payload:
        tid = normalize_text(item.get("taxonomy_id", ""))
        # Ignore removed/sovereign rules if still present
        if tid in {"DPDP_RUL-0225", "DPDP_ACT-0106", "DPDP_ACT-0124", "DPDP_ACT-0151", "DPDP_ACT-0152", "IT_ACT_2-0426", "IT_ACT_2-0314", "IT_ACT_2-0293"}:
            continue
        if tid and tid not in seen:
            seen.add(tid)
            records.append({
                "taxonomy_id": tid,
                "act": normalize_text(item.get("act", "")),
                "category": normalize_text(item.get("category", "")),
                "weight": normalize_text(item.get("weight", "")),
                "requirement": normalize_text(item.get("requirement", "")),
                "checkable_test": normalize_text(item.get("checkable_test", "")),
                "required_concepts": normalize_text(item.get("required_concepts", "")),
                "exclude_concepts": normalize_text(item.get("exclude_concepts", "")),
            })
    return records, file_sha256(path)


def load_corpus(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    df["clause_id"] = df["clause_id"].map(normalize_text)
    df["clause_text"] = df["clause_text"].map(normalize_text)
    if "source_document" not in df.columns:
        df["source_document"] = ""
    df["source_document"] = df["source_document"].map(normalize_text)
    return df, file_sha256(path)


def select_candidate_rules(clause_text: str, taxonomy: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dynamically route the top 4-8 most relevant taxonomy rules for this specific clause."""
    lower_text = clause_text.lower()
    scored_candidates = []

    for rule in taxonomy:
        score = 0
        cat = rule["category"]
        
        # Check category keywords
        keywords = CATEGORY_KEYWORD_MAP.get(cat, [])
        for kw in keywords:
            if kw in lower_text:
                score += 3

        # Check required concepts in taxonomy definition
        concepts = [c.strip().lower() for c in rule.get("required_concepts", "").split(";") if c.strip()]
        for c in concepts:
            if c in lower_text:
                score += 4

        # Check raw requirement words
        for token in rule["requirement"].lower().split():
            if len(token) > 4 and token in lower_text:
                score += 1

        if score > 0:
            scored_candidates.append((score, rule))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = [r for _, r in scored_candidates[:6]]

    # If no candidates triggered keyword match, provide 3 foundational rules (Notice, Rights, Safeguards)
    if not candidates:
        fallback_cats = {"Consent & Notice", "Data Principal Rights", "Security Safeguards"}
        candidates = [r for r in taxonomy if r["category"] in fallback_cats][:4]

    return candidates


def format_taxonomy(taxonomy: list[dict[str, Any]]) -> str:
    lines = []
    for item in taxonomy:
        lines.append(
            f"[{item['taxonomy_id']}] Category: {item['category']} | Act: {item['act']}\n"
            f"  Requirement: {item['requirement']}\n"
            f"  Checkable Test: {item['checkable_test']}"
        )
    return "\n\n".join(lines)


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_array(raw: str) -> Any:
    text = _strip_thinking_tags(raw)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    obj_start = text.find("{")
    if obj_start != -1:
        try:
            obj = json.loads(text[obj_start:])
            if isinstance(obj, dict):
                for val in obj.values():
                    if isinstance(val, list):
                        return val
                if "clause_id" in obj:
                    return [obj]
        except (ValueError, json.JSONDecodeError):
            pass
    raise ValidationError(f"Model response did not contain valid JSON array: {raw[:200]}")


def validate_results(
    raw_results: Any,
    batch: list[dict[str, Any]],
    candidate_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        raise ValidationError("Model response must be a JSON array")

    expected_ids = [row["clause_id"] for row in batch]
    expected_set = set(expected_ids)
    seen = []
    normalized = []

    for position, result in enumerate(raw_results, start=1):
        if not isinstance(result, dict):
            raise ValidationError(f"Result {position} is not an object")

        cid = normalize_text(result.get("clause_id", ""))
        tid = normalize_text(result.get("best_match_taxonomy_id", "NONE"))
        verdict = normalize_text(result.get("verdict", "Not Addressed"))
        justification = normalize_text(result.get("justification", ""))

        if cid not in expected_set or cid in seen:
            continue

        if tid not in candidate_ids:
            tid = "NONE"

        if tid == "NONE" or verdict == "Not Addressed":
            tid = "NONE"
            verdict = "Not Addressed"
            if not justification:
                justification = "The clause does not meaningfully address the listed taxonomy requirements."

        if justification and len(justification.split()) > 45:
            justification = " ".join(justification.split()[:45])

        seen.append(cid)
        normalized.append({
            "clause_id": cid,
            "best_match_taxonomy_id": tid,
            "verdict": verdict,
            "justification": justification,
            "label_needs_review": False,
        })

    # Fill any missed clause as Not Addressed
    for cid in expected_ids:
        if cid not in seen:
            normalized.append({
                "clause_id": cid,
                "best_match_taxonomy_id": "NONE",
                "verdict": "Not Addressed",
                "justification": "The clause does not meaningfully address the candidate requirements.",
                "label_needs_review": False,
            })

    return normalized


def call_model(
    clause_row: dict[str, Any],
    candidate_rules: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    candidate_ids = {r["taxonomy_id"] for r in candidate_rules}
    taxonomy_text = format_taxonomy(candidate_rules)
    batch = [clause_row]

    user_prompt = (
        f"CANDIDATE STATUTORY RULES:\n{taxonomy_text}\n\n"
        f"CLAUSE TO AUDIT:\n"
        f"[{clause_row['clause_id']}] Document: {clause_row.get('source_document', '')}\n"
        f"Text: \"{clause_row.get('clause_text', '')}\"\n\n"
        f"TASK:\n"
        f"1. Select the single best matching rule ID ONLY if this clause specifically regulates that subject matter.\n"
        f"2. If this clause does NOT implement or address any of the candidate rules, return 'NONE' and 'Not Addressed'.\n"
        f"3. Stating the platform DOES NOT do something required by law (e.g. not keeping logs) is NEVER Compliant.\n\n"
        f"Return ONLY a JSON array with: clause_id, best_match_taxonomy_id, verdict, justification."
    )

    client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
    response = client.chat(
        model=config.model,
        think=False,
        format=build_label_json_schema([clause_row["clause_id"]], candidate_ids),
        options={"num_predict": OLLAMA_NUM_PREDICT, "temperature": OLLAMA_TEMPERATURE},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response["message"]["content"]
    validated = validate_results(extract_json_array(raw), batch, candidate_ids)
    return validated[0]


def main() -> int:
    args = parse_args()
    config = Config(
        corpus=args.corpus,
        taxonomy=args.taxonomy,
        output=args.output,
        cache_dir=args.cache_dir,
        failures=args.failures,
        run_state=args.run_state,
        model=args.model,
        batch_size=args.batch_size,
        retry_max=args.retry_max,
        retry_backoff=args.retry_backoff,
        force_new_run=args.force_new_run,
    )

    corpus_df, corpus_hash = load_corpus(config.corpus)
    taxonomy, taxonomy_hash = load_taxonomy(config.taxonomy)
    taxonomy_lookup = {item["taxonomy_id"]: item for item in taxonomy}

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    run_fingerprint = sha256_json({
        "corpus_hash": corpus_hash,
        "taxonomy_hash": taxonomy_hash,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
    })
    run_id = f"run-{run_fingerprint[:12]}-{uuid.uuid4().hex[:8]}" if config.force_new_run else f"run-{run_fingerprint[:16]}"

    rows = corpus_df.to_dict(orient="records")
    all_labels = {}
    print(f"Starting dynamic-gated labeling run {run_id}")
    print(f"Corpus: {len(rows)} clauses | Active Taxonomy: {len(taxonomy)} benchmark rules")

    for idx, row in enumerate(rows, start=1):
        cid = row["clause_id"]
        candidates = select_candidate_rules(row["clause_text"], taxonomy)

        # Content cache key based on clause text and candidate set
        cache_identity = {
            "prompt_version": PROMPT_VERSION,
            "model": config.model,
            "clause_id": cid,
            "clause_text": row["clause_text"],
            "candidates": [c["taxonomy_id"] for c in candidates],
        }
        cache_key = sha256_json(cache_identity)
        c_path = config.cache_dir / f"clause_{cache_key}.json"

        if not config.force_new_run and c_path.exists():
            try:
                cached = load_json(c_path)
                all_labels[cid] = {**cached, "label_status": "CACHED_SUCCESS", "run_id": run_id}
                continue
            except Exception:
                pass

        label_res = None
        for attempt in range(1, config.retry_max + 1):
            try:
                label_res = call_model(row, candidates, config)
                break
            except Exception as exc:
                if attempt == config.retry_max:
                    print(f"Clause {cid} failed after {config.retry_max} attempts: {exc}", file=sys.stderr)
                time.sleep(config.retry_backoff * attempt)

        if label_res is not None:
            label_res["cache_key"] = cache_key
            label_res["model"] = config.model
            atomic_write_json(c_path, label_res)
            all_labels[cid] = {**label_res, "label_status": "NEW_SUCCESS", "run_id": run_id}
        else:
            all_labels[cid] = {
                "clause_id": cid,
                "best_match_taxonomy_id": "NONE",
                "verdict": "",
                "justification": "Model labeling failed during inference.",
                "label_needs_review": True,
                "label_status": "NEEDS_REVIEW",
                "cache_key": cache_key,
                "run_id": run_id,
                "model": config.model,
            }

        if idx % 25 == 0 or idx == len(rows):
            matched_so_far = sum(1 for v in all_labels.values() if v.get("best_match_taxonomy_id") not in {"NONE", ""})
            print(f"  Processed {idx}/{len(rows)} clauses... (Positive Matches: {matched_so_far})")

    label_df = pd.DataFrame(list(all_labels.values()))
    merged = corpus_df.merge(label_df, on="clause_id", how="left")

    merged["dpdp_category"] = merged["best_match_taxonomy_id"].map(
        lambda val: taxonomy_lookup.get(val, {}).get("category", "Not Applicable") if val else "Not Applicable"
    )
    merged["matched_act"] = merged["best_match_taxonomy_id"].map(
        lambda val: taxonomy_lookup.get(val, {}).get("act", "None") if val else "None"
    )
    merged["verdict_score"] = merged["verdict"].map(VERDICTS).fillna(0.0)

    output_cols = [
        "clause_id", "app_name", "source_document", "clause_text",
        "best_match_taxonomy_id", "dpdp_category", "matched_act", "verdict", "verdict_score",
        "justification", "label_status", "label_needs_review", "cache_key", "run_id", "model"
    ]
    output_cols = [c for c in output_cols if c in merged.columns]
    
    config.output.parent.mkdir(parents=True, exist_ok=True)
    merged[output_cols].to_csv(config.output, index=False)

    summary = {
        "run_id": run_id,
        "input_clause_count": len(corpus_df),
        "labeled_success_count": int((merged["label_status"].isin(["NEW_SUCCESS", "CACHED_SUCCESS"])).sum()),
        "matched_positive_count": int((merged["best_match_taxonomy_id"].isin(taxonomy_lookup)).sum()),
        "verdict_distribution": merged["verdict"].value_counts().to_dict(),
        "category_distribution": merged["dpdp_category"].value_counts().to_dict(),
    }
    atomic_write_json(config.run_state, summary)

    print("\n=== Labeling Run Complete ===")
    print(json.dumps(summary, indent=2))
    print(f"Output saved to: {config.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())