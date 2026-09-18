"""
Resumable, cache-backed bootstrap labeling of social-media policy clauses.

The model labels each policy clause against a finalized taxonomy. This is a
bootstrap dataset for human verification, not a legal determination.

Production guarantees
---------------------
1. Every input clause has a stable identity and is validated for uniqueness.
2. Every batch has a content-addressed cache key based on:
   - clause content and IDs;
   - taxonomy content;
   - prompt version;
   - model settings.
3. Completed batches are atomically written to individual cache files.
4. A process interruption loses at most the currently running model call.
5. Reruns reuse valid cache files and continue from the first missing batch.
6. Changed input, taxonomy, prompt, model, or settings automatically creates
   new cache keys instead of silently reusing stale labels.
7. Model output is strictly validated for:
   - exact expected clause IDs;
   - no duplicates;
   - valid taxonomy IDs;
   - valid verdicts;
   - non-empty justifications when a taxonomy match is present.
8. Invalid batches are retried and can be split into smaller batches.
9. Failed clauses are marked NEEDS_REVIEW; they are never silently converted
   to Not Addressed.
10. The output includes provenance, cache keys, model metadata, and run status.

Usage
-----
python label_policy_clauses_resumable.py \
  --corpus datasets/SocialMediaPolicies/social_media_clauses_flagged.csv \
  --taxonomy corpus/dpdp_taxonomy_final.json \
  --output corpus/labeled_clauses_bootstrap.csv \
  --cache-dir corpus/label_cache

Resume after interruption: run the same command again.
Start a new run explicitly:
  add --force-new-run

Dependencies
------------
pip install pandas ollama
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


DEFAULT_CORPUS = Path("datasets/SocialMediaPolicies/social_media_clauses_flagged.csv")
DEFAULT_TAXONOMY = Path("corpus/dpdp_taxonomy_final.json")
DEFAULT_OUTPUT = Path("corpus/labeled_clauses_bootstrap.csv")
DEFAULT_CACHE_DIR = Path("corpus/label_cache")
DEFAULT_FAILURES = Path("corpus/labeled_clauses_failures.json")
DEFAULT_RUN_STATE = Path("corpus/labeled_clauses_run_state.json")

PROMPT_VERSION = "policy-labeling-v7-reliable-batches"
MODEL_DEFAULT = "qwen3:8b"
BATCH_SIZE_DEFAULT = 6
RETRY_MAX_DEFAULT = 2
RETRY_BACKOFF_DEFAULT = 2.0

VERDICTS = {
    "Compliant": 1.0,
    "Partially Compliant": 0.5,
    "Non-Compliant": 0.0,
    "Not Addressed": 0.0,
}

OLLAMA_NUM_PREDICT = 4096
OLLAMA_TEMPERATURE = 0.1
OLLAMA_TIMEOUT_SECONDS = 180.0


def build_label_json_schema(clause_ids: list[str], taxonomy_ids: set[str]) -> dict[str, Any]:
    """Constrain Ollama output to this batch's clause IDs and real taxonomy IDs."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "clause_id": {"type": "string", "enum": list(clause_ids)},
                "best_match_taxonomy_id": {
                    "type": "string",
                    "enum": ["NONE", *sorted(taxonomy_ids)],
                },
                "verdict": {
                    "type": "string",
                    "enum": list(VERDICTS),
                },
                "justification": {"type": "string"},
            },
            "required": ["clause_id", "best_match_taxonomy_id", "verdict", "justification"],
        },
    }


SYSTEM_PROMPT = """You are labeling social-media policy clauses against a finalized legal compliance taxonomy.

For every input clause, choose the single taxonomy requirement it most meaningfully addresses,
or use NONE if no listed requirement is meaningfully addressed.

STRICT SEMANTIC MATCHING RULES:
- Match the legal concept, not an isolated keyword. The word "security" alone is never sufficient.
- A security/safeguards requirement matches only clauses about protecting personal data, personal-data confidentiality/integrity/availability, access controls for personal data, encryption, breach prevention/response, or comparable data-protection safeguards.
- Do NOT match platform, application, account, infrastructure, anti-circumvention, abuse-prevention, content-integrity, copyright/IP, watermark, legal-notice, branding, advertising, UI, payment, billing, subscription, or general operational clauses to a personal-data security requirement unless the clause expressly concerns personal data.
- A clause about removing watermarks/labels/legal or proprietary notices is not a data-security match.
- A clause about circumventing platform security features is not a personal-data protection match.
- If the clause does not expressly or unambiguously concern the same legal subject and obligation as a taxonomy requirement, choose NONE.
- When uncertain between a weak taxonomy match and NONE, choose NONE.

Verdict definitions:
- Compliant: the clause clearly satisfies the matched requirement.
- Partially Compliant: the clause addresses the topic but is vague, incomplete, or missing a material element.
- Non-Compliant: the clause explicitly contradicts the matched requirement. Do not use this merely because detail is missing.
- Not Addressed: the clause does not meaningfully address a taxonomy requirement. If the match is NONE, verdict must be Not Addressed.

Calibration:
- Vague-but-present language is Partially Compliant.
- Absence of a topic is Not Addressed, not Non-Compliant.
- Do not infer facts not stated in the clause.
- Use only taxonomy IDs supplied in the TAXONOMY block. Copy the ID exactly; never abbreviate.
- Never leave verdict blank. If nothing matches, verdict must be Not Addressed and best_match_taxonomy_id must be NONE.
- Treat each taxonomy item's Required concepts and Exclude fields as hard semantic gates, not suggestions.
- A clause must satisfy the required concepts and must not fall within the exclusions before it can be matched.
- If the clause fits an exclusion or fails a required concept, choose NONE even if a keyword overlaps.
- Return exactly one result for every supplied clause_id, with no extras.
- Return ONLY a JSON array and no markdown.

Each result must have exactly these keys:
clause_id, best_match_taxonomy_id, verdict, justification

Justification must be one concise sentence of at most 30 words. For NONE/Not Addressed, explain that no listed requirement is meaningfully addressed.
"""


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
    split_failed_batches: bool
    max_split_depth: int


class LabelingError(Exception):
    """Expected error for model/API/validation failures."""


class ValidationError(LabelingError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable policy-clause labeling with content-addressed caching.")
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
    parser.add_argument("--no-split-failed-batches", action="store_true")
    parser.add_argument("--max-split-depth", type=int, default=3)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
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
    if not isinstance(payload, list) or not payload:
        raise ValueError("Taxonomy JSON must be a non-empty list or contain a non-empty records list")

    records: list[dict[str, Any]] = []
    taxonomy_ids: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Taxonomy item {index} is not an object")
        taxonomy_id = normalize_text(item.get("taxonomy_id", ""))
        requirement = normalize_text(item.get("requirement", ""))
        if not taxonomy_id or not requirement:
            raise ValueError(f"Taxonomy item {index} requires taxonomy_id and requirement")
        if taxonomy_id in taxonomy_ids:
            raise ValueError(f"Duplicate taxonomy_id: {taxonomy_id}")
        taxonomy_ids.add(taxonomy_id)
        records.append({
            "taxonomy_id": taxonomy_id,
            "category": normalize_text(item.get("category", "")),
            "weight": normalize_text(item.get("weight", "")),
            "requirement": requirement,
            "checkable_test": normalize_text(item.get("checkable_test", "")),
            "required_concepts": normalize_text(item.get("required_concepts", "")),
            "exclude_concepts": normalize_text(item.get("exclude_concepts", "")),
            "positive_examples": normalize_text(item.get("positive_examples", "")),
            "negative_examples": normalize_text(item.get("negative_examples", "")),
        })
    return records, file_sha256(path)


def load_corpus(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    if "clause_id" not in dataframe.columns or "clause_text" not in dataframe.columns:
        raise ValueError("Corpus must contain clause_id and clause_text columns")
    dataframe["clause_id"] = dataframe["clause_id"].map(normalize_text)
    dataframe["clause_text"] = dataframe["clause_text"].map(normalize_text)
    if dataframe["clause_id"].eq("").any():
        raise ValueError("Corpus contains blank clause_id values")
    duplicates = dataframe[dataframe["clause_id"].duplicated(keep=False)]["clause_id"].tolist()
    if duplicates:
        raise ValueError(f"Corpus contains duplicate clause_id values: {sorted(set(duplicates))[:20]}")
    return dataframe, file_sha256(path)


def format_taxonomy(taxonomy: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in taxonomy:
        lines.append(
            f"[{item['taxonomy_id']}] ({item['category']}, weight {item['weight']}) {item['requirement']}"
        )
        required = item.get("required_concepts", "")
        exclude = item.get("exclude_concepts", "")
        if required:
            lines.append(f"  Required: {required}")
        if exclude:
            lines.append(f"  Exclude: {exclude}")
    return "\n".join(lines)


def batch_cache_key(
    taxonomy_hash: str,
    batch: list[dict[str, Any]],
    config: Config,
) -> str:
    identity = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_hash": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "taxonomy_hash": taxonomy_hash,
        "model": config.model,
        "batch_size": config.batch_size,
        "ollama_options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
            "format": "json_schema_enum_v1",
        },
        "clauses": [
            {"clause_id": row["clause_id"], "clause_text": row.get("clause_text", "")}
            for row in batch
        ],
    }
    return sha256_json(identity)


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"batch_{key}.json"


def format_batch(batch: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{row['clause_id']}] {row.get('clause_text', '')}" for row in batch)


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. qwen3)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _find_balanced_array(text: str, start: int) -> list[Any]:
    """Parse a balanced JSON array from *text* beginning at index *start*."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValidationError("Unterminated JSON array in model response")


def _extract_complete_objects(text: str) -> list[dict[str, Any]]:
    """Salvage complete JSON objects from a truncated or wrapped response."""
    objects: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            break
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        cursor = end + 1
    return objects


def extract_json_array(raw: str) -> Any:
    # Strip thinking-model tags before any other processing.
    text = _strip_thinking_tags(raw)

    # Fast path: entire response is already a JSON array.
    if text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Second fast path: look for the first '[' and parse a balanced array.
    start = text.find("[")
    if start != -1:
        try:
            return _find_balanced_array(text, start)
        except (ValidationError, json.JSONDecodeError):
            pass

    salvaged = _extract_complete_objects(text)
    if salvaged:
        return salvaged

    # Fallback: model may have wrapped a single object (not an array).
    obj_start = text.find("{")
    if obj_start != -1:
        try:
            obj = json.loads(text[obj_start:])
            if isinstance(obj, dict):
                arrays = [value for value in obj.values() if isinstance(value, list)]
                if len(arrays) == 1:
                    return arrays[0]
                if "clause_id" in obj:
                    return [obj]
        except (ValueError, json.JSONDecodeError):
            pass

    raise ValidationError("Model response did not contain valid JSON array")


def _first_present(result: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in result and result[key] not in (None, ""):
            return result[key]
    return ""


def resolve_taxonomy_id(raw_id: str, taxonomy_ids: set[str]) -> str:
    """Map a model taxonomy ID onto an allowed ID when the match is unambiguous."""
    taxonomy_id = normalize_text(raw_id)
    if not taxonomy_id or taxonomy_id.upper() == "NONE":
        return "NONE"
    if taxonomy_id in taxonomy_ids:
        return taxonomy_id
    lowered = {item.lower(): item for item in taxonomy_ids}
    if taxonomy_id.lower() in lowered:
        return lowered[taxonomy_id.lower()]
    prefix_hits = [item for item in taxonomy_ids if item.startswith(taxonomy_id)]
    if len(prefix_hits) == 1:
        return prefix_hits[0]
    return taxonomy_id


def normalize_raw_result(result: dict[str, Any], taxonomy_ids: set[str]) -> dict[str, Any]:
    """Map common model-shape drift onto the required label keys."""
    matches = result.get("matches")
    if isinstance(matches, list) and matches:
        first = matches[0] if isinstance(matches[0], dict) else {}
        merged = {**first, **{k: v for k, v in result.items() if k != "matches"}}
        result = merged
    elif matches == [] and not result.get("best_match_taxonomy_id") and not result.get("verdict"):
        result = {
            **result,
            "best_match_taxonomy_id": "NONE",
            "verdict": "Not Addressed",
            "justification": result.get("justification")
            or "No listed requirement is meaningfully addressed.",
        }

    taxonomy_id = resolve_taxonomy_id(
        str(_first_present(result, ("best_match_taxonomy_id", "taxonomy_id", "match_id", "requirement_id"))),
        taxonomy_ids,
    )
    verdict = normalize_text(_first_present(result, ("verdict", "compliance", "label", "status")))
    aliases = {
        "not addressed": "Not Addressed",
        "not_addressed": "Not Addressed",
        "none": "Not Addressed",
        "n/a": "Not Addressed",
        "na": "Not Addressed",
        "compliant": "Compliant",
        "partially compliant": "Partially Compliant",
        "partial": "Partially Compliant",
        "non-compliant": "Non-Compliant",
        "noncompliant": "Non-Compliant",
    }
    verdict = aliases.get(verdict.lower(), verdict) if verdict else verdict
    justification = result.get("justification", result.get("reason", ""))
    if taxonomy_id == "NONE":
        if not verdict:
            verdict = "Not Addressed"
        if not normalize_text(justification):
            justification = "No listed requirement is meaningfully addressed."
    return {
        "clause_id": result.get("clause_id", ""),
        "best_match_taxonomy_id": taxonomy_id,
        "verdict": verdict,
        "justification": justification,
    }


def validate_results(
    raw_results: Any,
    batch: list[dict[str, Any]],
    taxonomy_ids: set[str],
) -> list[dict[str, str]]:
    if not isinstance(raw_results, list):
        raise ValidationError("Model response must be a JSON array")

    expected_ids = [row["clause_id"] for row in batch]
    expected_set = set(expected_ids)
    seen: list[str] = []
    normalized: list[dict[str, str]] = []
    allowed_verdicts = set(VERDICTS)

    for position, result in enumerate(raw_results, start=1):
        if not isinstance(result, dict):
            raise ValidationError(f"Result {position} is not an object")
        result = normalize_raw_result(result, taxonomy_ids)
        clause_id = normalize_text(result.get("clause_id", ""))
        taxonomy_id = resolve_taxonomy_id(result.get("best_match_taxonomy_id", ""), taxonomy_ids)
        verdict = normalize_text(result.get("verdict", ""))
        justification = normalize_text(result.get("justification", ""))
        if clause_id not in expected_set:
            raise ValidationError(f"Unexpected clause_id: {clause_id}")
        if clause_id in seen:
            raise ValidationError(f"Duplicate result for clause_id: {clause_id}")
        if taxonomy_id != "NONE" and taxonomy_id not in taxonomy_ids:
            raise ValidationError(f"Unknown taxonomy_id {taxonomy_id} for {clause_id}")
        if verdict not in allowed_verdicts:
            raise ValidationError(f"Invalid verdict {verdict!r} for {clause_id}")
        if taxonomy_id == "NONE" and verdict != "Not Addressed":
            raise ValidationError(f"NONE match must have Not Addressed verdict for {clause_id}")
        if not justification:
            raise ValidationError(f"Missing justification for {clause_id}")
        if len(justification.split()) > 40:
            raise ValidationError(f"Justification is too long for {clause_id}")
        seen.append(clause_id)
        normalized.append({
            "clause_id": clause_id,
            "best_match_taxonomy_id": taxonomy_id,
            "verdict": verdict,
            "justification": justification,
        })

    if set(seen) != expected_set or len(seen) != len(expected_ids):
        missing = sorted(expected_set - set(seen))
        extra_count = len(seen) - len(set(seen))
        raise ValidationError(f"Batch result coverage mismatch; missing={missing}, duplicate_count={extra_count}")

    # Enforce the most important semantic invariant at the output boundary:
    # a NONE match must always be Not Addressed. Other semantic checks remain
    # human-reviewable because they require the clause and taxonomy meaning.

    # Stable order makes cache files deterministic and output diffs reviewable.
    order = {clause_id: index for index, clause_id in enumerate(expected_ids)}
    return sorted(normalized, key=lambda item: order[item["clause_id"]])


def call_model(taxonomy_text: str, batch: list[dict[str, Any]], config: Config, taxonomy_ids: set[str]) -> list[dict[str, str]]:
    user_message = (
        f"TAXONOMY:\n{taxonomy_text}\n\n"
        f"CLAUSES:\n{format_batch(batch)}\n\n"
        "Return ONLY a JSON array containing exactly one object per input clause_id. "
        "Each object must include clause_id, best_match_taxonomy_id, verdict, and justification. "
        "Copy best_match_taxonomy_id exactly from the TAXONOMY block, or use NONE. "
        "verdict must be one of: Compliant, Partially Compliant, Non-Compliant, Not Addressed. "
        'If nothing matches, use best_match_taxonomy_id="NONE" and verdict="Not Addressed".'
    )
    clause_ids = [row["clause_id"] for row in batch]
    try:
        client = ollama.Client(timeout=OLLAMA_TIMEOUT_SECONDS)
        response = client.chat(
            model=config.model,
            think=False,
            # Enumerate allowed IDs/verdicts so the grammar cannot emit blank
            # verdicts, truncated taxonomy IDs, or a "matches" wrapper.
            format=build_label_json_schema(clause_ids, taxonomy_ids),
            options={"num_predict": OLLAMA_NUM_PREDICT, "temperature": OLLAMA_TEMPERATURE},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise LabelingError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc
    try:
        raw = response["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise LabelingError(f"Malformed Ollama response: {exc}") from exc
    return validate_results(extract_json_array(raw), batch, taxonomy_ids)


def read_valid_cache(
    path: Path,
    expected_key: str,
    batch: list[dict[str, Any]],
    taxonomy_ids: set[str],
) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
        if payload.get("cache_key") != expected_key:
            return None
        results = validate_results(payload.get("labels"), batch, taxonomy_ids)
        return results
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError):
        # A corrupt or incomplete cache is ignored and safely regenerated.
        return None


def write_cache(
    path: Path,
    key: str,
    batch: list[dict[str, Any]],
    labels: list[dict[str, str]],
    config: Config,
) -> None:
    payload = {
        "cache_key": key,
        "created_at_utc": utc_now(),
        "prompt_version": PROMPT_VERSION,
        "model": config.model,
        "clause_ids": [row["clause_id"] for row in batch],
        "labels": labels,
        "status": "SUCCESS",
    }
    atomic_write_json(path, payload)


def label_with_retries(
    taxonomy_text: str,
    batch: list[dict[str, Any]],
    config: Config,
    taxonomy_ids: set[str],
    batch_label: str,
    retry_max: int | None = None,
) -> tuple[list[dict[str, str]] | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    attempts = retry_max if retry_max is not None else config.retry_max
    for attempt in range(1, attempts + 1):
        try:
            labels = call_model(taxonomy_text, batch, config, taxonomy_ids)
            return labels, errors
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep the run alive; record exact failure
            error = {
                "batch": batch_label,
                "attempt": attempt,
                "clause_ids": [row["clause_id"] for row in batch],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_utc": utc_now(),
            }
            errors.append(error)
            print(f"  {batch_label}: attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if attempt < attempts:
                time.sleep(config.retry_backoff * attempt)
    return None, errors


def atomic_write_output(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.retry_max < 1 or args.max_split_depth < 0:
        raise ValueError("batch-size, retry-max must be positive and max-split-depth cannot be negative")

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
        split_failed_batches=not args.no_split_failed_batches,
        max_split_depth=args.max_split_depth,
    )

    corpus_df, corpus_hash = load_corpus(config.corpus)
    taxonomy, taxonomy_hash = load_taxonomy(config.taxonomy)
    taxonomy_ids = {item["taxonomy_id"] for item in taxonomy}
    taxonomy_lookup = {item["taxonomy_id"]: item for item in taxonomy}
    taxonomy_text = format_taxonomy(taxonomy)

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    run_fingerprint = sha256_json({
        "corpus_hash": corpus_hash,
        "taxonomy_hash": taxonomy_hash,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
        "batch_size": config.batch_size,
    })
    run_id = f"run-{run_fingerprint[:16]}"

    if config.force_new_run:
        # A new run ID is useful for an intentional fresh attempt, but old
        # content-addressed cache files remain available for inspection.
        run_id = f"run-{run_fingerprint[:12]}-{uuid.uuid4().hex[:8]}"

    rows = corpus_df.to_dict(orient="records")
    all_labels: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    cache_hits = 0
    model_calls = 0
    batches_total = 0

    def process_batch(batch: list[dict[str, Any]], logical_name: str, depth: int = 0) -> None:
        nonlocal cache_hits, model_calls, batches_total
        if not batch:
            return
        batches_total += 1
        key = batch_cache_key(taxonomy_hash, batch, config)
        path = cache_path(config.cache_dir, key)
        cached = None if config.force_new_run else read_valid_cache(path, key, batch, taxonomy_ids)
        if cached is not None:
            cache_hits += 1
            for label in cached:
                all_labels[label["clause_id"]] = {
                    **label,
                    "label_status": "CACHED_SUCCESS",
                    "cache_key": key,
                    "run_id": run_id,
                    "model": config.model,
                }
            print(f"  {logical_name}: cache hit ({len(batch)} clauses)")
            return

        # Only fast-fail batches larger than the configured batch size;
        # standard-sized batches (up to batch_size) get the full retry budget.
        retries = 1 if (config.split_failed_batches and len(batch) > config.batch_size) else config.retry_max
        labels, errors = label_with_retries(
            taxonomy_text, batch, config, taxonomy_ids, logical_name, retry_max=retries
        )
        model_calls += 1
        if labels is not None:
            write_cache(path, key, batch, labels, config)
            for label in labels:
                all_labels[label["clause_id"]] = {
                    **label,
                    "label_status": "NEW_SUCCESS",
                    "cache_key": key,
                    "run_id": run_id,
                    "model": config.model,
                }
            print(f"  {logical_name}: labeled and cached ({len(batch)} clauses)")
            return

        # A malformed large response can often be repaired by splitting the
        # batch. Each child gets a different content-addressed key.
        if config.split_failed_batches and len(batch) > 1 and depth < config.max_split_depth:
            midpoint = len(batch) // 2
            print(f"  {logical_name}: splitting failed batch into {len(batch[:midpoint])}+{len(batch[midpoint:])}")
            process_batch(batch[:midpoint], f"{logical_name}.a", depth + 1)
            process_batch(batch[midpoint:], f"{logical_name}.b", depth + 1)
            return

        failures.extend(errors)
        for row in batch:
            all_labels[row["clause_id"]] = {
                "clause_id": row["clause_id"],
                "best_match_taxonomy_id": "",
                "verdict": "",
                "justification": "",
                "label_status": "NEEDS_REVIEW",
                "cache_key": key,
                "run_id": run_id,
                "model": config.model,
                "failure_reason": "; ".join(error["error"] for error in errors[-2:]),
            }
        print(f"  {logical_name}: FAILED; {len(batch)} clauses require review", file=sys.stderr)

    # Stable order is critical for predictable restart behavior.
    for start in range(0, len(rows), config.batch_size):
        batch = rows[start:start + config.batch_size]
        process_batch(batch, f"batch-{start // config.batch_size + 1}")

    label_rows = list(all_labels.values())
    label_df = pd.DataFrame(label_rows)
    merged = corpus_df.merge(label_df, on="clause_id", how="left", validate="one_to_one", suffixes=("", "_label"))

    # Do not turn failed labels into Not Addressed. Only model-produced verdicts
    # receive a verdict score. Failed records remain visibly incomplete.
    merged["best_match_taxonomy_id"] = merged["best_match_taxonomy_id"].fillna("")
    merged["verdict"] = merged["verdict"].fillna("")
    merged["justification"] = merged["justification"].fillna("")
    merged["label_status"] = merged["label_status"].fillna("NEEDS_REVIEW")
    merged["dpdp_category"] = merged["best_match_taxonomy_id"].map(
        lambda value: taxonomy_lookup.get(value, {}).get("category", "Not Applicable") if value else "Not Applicable"
    )
    merged["verdict_score"] = merged["verdict"].map(VERDICTS)
    merged["labeling_run_id"] = run_id
    merged["corpus_sha256"] = corpus_hash
    merged["taxonomy_sha256"] = taxonomy_hash
    merged["prompt_version"] = PROMPT_VERSION

    output_columns = [
        "clause_id", "app_name", "source_document", "clause_text",
        "best_match_taxonomy_id", "dpdp_category", "verdict", "verdict_score",
        "justification", "label_status", "cache_key", "run_id", "model",
        "failure_reason", "labeling_run_id", "corpus_sha256", "taxonomy_sha256",
        "prompt_version",
    ]
    output_columns = [column for column in output_columns if column in merged.columns]
    atomic_write_output(merged[output_columns], config.output)

    state = {
        "run_id": run_id,
        "completed_at_utc": utc_now(),
        "corpus": str(config.corpus),
        "corpus_sha256": corpus_hash,
        "taxonomy": str(config.taxonomy),
        "taxonomy_sha256": taxonomy_hash,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
        "input_clause_count": len(corpus_df),
        "labeled_success_count": int((merged["label_status"].isin(["NEW_SUCCESS", "CACHED_SUCCESS"])).sum()),
        "needs_review_count": int((merged["label_status"] == "NEEDS_REVIEW").sum()),
        "cache_hits": cache_hits,
        "cache_bypassed": config.force_new_run,
        "model_calls": model_calls,
        "batches_processed_including_splits": batches_total,
        "failure_event_count": len(failures),
        "status": "COMPLETE_WITH_REVIEW_ITEMS" if failures else "COMPLETE",
    }
    atomic_write_json(config.run_state, state)
    atomic_write_json(config.failures, failures)

    print("\n=== Resumable labeling summary ===")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print(f"Saved labels: {config.output}")
    print(f"Cache directory: {config.cache_dir}")
    print(f"Failure log: {config.failures}")
    print("Only labels with label_status NEW_SUCCESS or CACHED_SUCCESS are model-generated.")
    print("NEEDS_REVIEW rows must be resolved before training.")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
