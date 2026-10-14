"""
Production-grade taxonomy candidate generation from government legal clauses.

This script generates TAXONOMY CANDIDATES. It does not create a final legal
compliance taxonomy automatically. Every candidate must be reviewed and
approved by a qualified legal researcher before it is used for annotation or
model training.

Design goals
------------
1. Preserve legal provenance for every candidate and every source clause.
2. Never silently lose clauses because of duplicate collapse or model errors.
3. Keep LLM-generated drafts separate from human-approved taxonomy entries.
4. Validate model output against a strict schema and allowed categories.
5. Retry failed batches and cache successful results for resumability.
6. Record unmapped clauses, invalid model records, and batch errors.
7. Produce conservative semantic merge candidates for manual review only.
8. Support all policy document types downstream; the taxonomy is not limited
   to privacy-policy terminology.

Input
-----
The input should be the output of the recall-first government-clause filter:

    datasets/GovernmentActs/government_clauses_filtered.json

The script also accepts government_clauses.json and CSV input. Required field:
    clause_text

Recommended fields:
    clause_id, source_act, source_file, page_start, section_heading,
    exact_duplicate_group, relevance_signal_level or relevance_level,
    administrative_signal

Outputs
-------
By default, under corpus/taxonomy_generation/:

    taxonomy_candidates_raw.json
    taxonomy_candidates_normalized.json
    taxonomy_merge_candidates.json
    taxonomy_unmapped_clauses.json
    taxonomy_invalid_model_records.json
    taxonomy_batch_manifest.json
    taxonomy_run_summary.json
    cache/*.json

The normalized output is still a DRAFT. It includes `human_review_status` set to
`PENDING` for every candidate.

Install
-------
    pip install pandas ollama sentence-transformers

The Ollama server and model must be available locally, for example:
    ollama serve
    ollama pull qwen3:8b

Run
---
    python src/taxonomy/generate_taxonomy.py \
        --input datasets/GovernmentActs/government_clauses_filtered.json \
        --output-dir corpus/taxonomy_generation \
        --model qwen3:8b

Resume an interrupted run using the same command. Successful batches are
loaded from the cache and are not sent to the model again.

To force regeneration:
    python src/taxonomy/generate_taxonomy.py ... --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import ollama
except ImportError:  # Provide a clear runtime message rather than import failure noise.
    ollama = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


DEFAULT_INPUT = Path("datasets/GovernmentActs/government_clauses_filtered.json")
DEFAULT_OUTPUT_DIR = Path("corpus/taxonomy_generation")
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_RETRIES = 3
DEFAULT_MERGE_THRESHOLD = 0.85

ALLOWED_CATEGORIES = {
    "Consent & Notice",
    "Data Retention & Erasure",
    "Security Safeguards",
    "Breach Notification",
    "Cross-Border Transfer",
    "Children/Vulnerable Groups",
    "Data Principal Rights",
    "Grievance Redressal",
    "Significant Data Fiduciary Obligations",
    "Intermediary/Platform Liability",
    "Confidentiality & Misuse of Data",
}

# This is intentionally a controlled list. If the legal team adds a category,
# it should be added explicitly and the taxonomy version should be incremented.
ALLOWED_WEIGHTS = {1, 2, 3}

# Regex that matches values the model should never use as a legal citation.
# Filenames (anything ending in a known document extension) are forbidden by
# citation rules 8-10 in SYSTEM_PROMPT and are automatically rejected here.
_FILENAME_SOURCE_RE = re.compile(
    r"\.(?:pdf|json|csv|txt|docx?|xlsx?|xml|html?)\s*$",
    re.IGNORECASE,
)

# Threshold for considering two requirements in the same category near-identical.
# Entries below this cosine similarity are treated as distinct obligations and kept.
SAME_CATEGORY_DUPLICATE_THRESHOLD = 0.85

# Lazy-loaded embedding model for intra-batch duplicate detection.
# Loaded once on first use, not per batch, to avoid repeated download overhead.
_DEDUP_MODEL: Any = None


def _get_dedup_model() -> Any:
    """Return the sentence-transformer used for same-category duplicate detection.

    Falls back gracefully if sentence-transformers is not installed;
    callers must handle a None return value.
    """
    global _DEDUP_MODEL
    if _DEDUP_MODEL is None and SentenceTransformer is not None:
        _DEDUP_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _DEDUP_MODEL

SYSTEM_PROMPT = """You are assisting a legal researcher in drafting a candidate taxonomy
of independently checkable obligations from Indian data-protection and allied
legal materials. The candidate taxonomy will be used to audit social-media
governance documents of many possible types, including Privacy Policies, Terms
of Service, Cookies Policies, Data Processing Terms, Data Security Terms, Data
Transfer Addenda, Business/Merchant Terms, Payments Policies, Messaging and
Channels Guidelines, Intellectual Property Policies, and future document types.

Use only the supplied legal clause text. Do not invent a legal obligation or
citation. Do not infer an organization's actual internal practice.

Granularity rules:
1. Produce one independently checkable requirement per entry.
2. Merge clauses only when they state the same obligation, not merely a related topic.
3. Keep notice, consent, withdrawal, purpose, retention, erasure, security,
   breach communication, rights, grievance, children's data, transfer, and
   accountability duties separate when they have different tests.
4. Exclude institutional/personnel administration only when it cannot be
   checked in any social-media governance document. When uncertain, do not
   exclude it; return it for human review.
5. COMPOUND STATUTORY CLAUSE RULE:
   - If an input legal clause establishes multiple distinct, independently checkable obligations across different areas (e.g., both technical security safeguards and mandatory breach notification, or both notice content requirements and consent withdrawal mechanisms), you MAY generate a separate candidate entry for each distinct obligation.
   - Do NOT duplicate the same obligation under two different categories.
   - Every generated entry must have its own independently verifiable checkable_test and appropriate category.
   - If a clause contains no independently checkable compliance obligation, leave it unmapped (do not produce an entry for it).
6. Use exactly one category from the allowed category list below.
7. Weight must be 1, 2, or 3, where 3 is a core obligation and 1 is procedural.

CITATION RULES -- these are enforced, not optional:
8. The "source" field MUST cite a specific section, rule, or schedule
   number -- e.g. "DPDP Act 2023, Section 8(1)", "DPDP Rules 2025, Rule 8(2)",
   "IT Act 2000, Section 72". NEVER cite a filename (e.g. "DPDP_2025.pdf",
   "government_clauses.json", "DPDP_2023.pdf") -- a filename is not a legal
   citation and will be automatically rejected.
9. If the input clause text does not contain a visible section/rule number,
   look for one within the same clause_text block -- numbers often appear at
   the start of a clause (e.g. "8. The Consent Manager shall..." means
   Section or Rule 8). If truly no number is visible anywhere in the clause,
   set "source" to the Act name only (e.g. "DPDP Rules 2025") and set
   "citation_confidence" to "low". Do NOT substitute a filename under any
   circumstance.
10. If the citation is genuinely unavailable, use an empty string for
    "source" and mark citation_confidence as "missing" -- never a filename.

Allowed categories:
- Consent & Notice
- Data Retention & Erasure
- Security Safeguards
- Breach Notification
- Cross-Border Transfer
- Children/Vulnerable Groups
- Data Principal Rights
- Grievance Redressal
- Significant Data Fiduciary Obligations
- Intermediary/Platform Liability
- Confidentiality & Misuse of Data

Return ONLY a JSON array. Each object must contain exactly these keys:
category, requirement, source, weight, source_clause_ids, checkable_test,
applicability, assessability, citation_confidence, notes.
"""


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_dir: Path
    model: str
    embedding_model: str
    batch_size: int
    max_retries: int
    retry_delay: float
    merge_threshold: float
    temperature: float
    num_predict: int
    force: bool


@dataclass
class BatchResult:
    batch_id: str
    status: str
    input_clause_ids: list[str]
    raw_response: str
    parsed_entries: list[dict[str, Any]]
    invalid_records: list[dict[str, Any]]
    error: str | None
    attempts: int
    model: str
    created_at: str
    # Records of multi-mapped clause decisions for audit (added after initial release;
    # default_factory keeps old cached JSON files compatible via setdefault below).
    multi_mapped_records: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate auditable taxonomy candidates from legal clauses.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--merge-threshold", type=float, default=DEFAULT_MERGE_THRESHOLD)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--force", action="store_true", help="Ignore existing batch cache and regenerate all batches.")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
            raise ValueError("JSON input must be a list or contain a records list")
        dataframe = pd.DataFrame(payload)
    else:
        raise ValueError("Input must be .json or .csv")

    if "clause_text" not in dataframe.columns:
        raise ValueError("Input must contain clause_text")
    if dataframe.empty:
        raise ValueError("Input contains no clauses")

    dataframe = dataframe.copy()
    dataframe["clause_text"] = dataframe["clause_text"].fillna("").astype(str)
    if "clause_id" not in dataframe.columns:
        dataframe["clause_id"] = [f"INPUT-{index + 1:07d}" for index in range(len(dataframe))]
    dataframe["clause_id"] = dataframe["clause_id"].astype(str)

    duplicated_ids = dataframe[dataframe["clause_id"].duplicated(keep=False)]["clause_id"].tolist()
    if duplicated_ids:
        raise ValueError(f"Input contains duplicate clause_id values: {sorted(set(duplicated_ids))[:10]}")
    return dataframe


def get_signal_level(row: pd.Series) -> str:
    for column in ("relevance_signal_level", "relevance_level"):
        if column in row.index and str(row.get(column, "")).strip():
            return str(row.get(column)).strip().casefold()
    return "unknown"


def get_admin_signal(row: pd.Series) -> str:
    value = row.get("administrative_signal", "none")
    if pd.isna(value):
        return "none"
    return str(value).strip().casefold() or "none"


def deduplicate_representatives(dataframe: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Collapse exact duplicate groups only for model input, never for provenance."""
    representatives: list[dict[str, Any]] = []
    rep_lookup: dict[str, list[str]] = {}
    seen_groups: set[str] = set()

    for _, row in dataframe.iterrows():
        clause_id = str(row["clause_id"])
        group = str(row.get("exact_duplicate_group", "") or "").strip()
        if group and group in seen_groups:
            continue

        if group:
            member_ids = dataframe.loc[
                dataframe["exact_duplicate_group"].astype(str) == group,
                "clause_id",
            ].astype(str).tolist()
            seen_groups.add(group)
        else:
            member_ids = [clause_id]

        rep_lookup[clause_id] = member_ids
        representatives.append({
            "representative_clause_id": clause_id,
            "clause_text": str(row["clause_text"]),
            "source_act": str(row.get("source_act", "")),
            "source_file": str(row.get("source_file", "")),
            "page_start": str(row.get("page_start", "")),
            "section_heading": str(row.get("section_heading", "")),
            "relevance_signal_level": get_signal_level(row),
            "administrative_signal": get_admin_signal(row),
            "group_members": member_ids,
        })

    return representatives, rep_lookup


def order_representatives(representatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}

    def key(item: dict[str, Any]) -> tuple[int, int, str]:
        signal = item.get("relevance_signal_level", "unknown")
        admin = 1 if item.get("administrative_signal") == "possible" else 0
        return (order.get(signal, 3), admin, item["representative_clause_id"])

    return sorted(representatives, key=key)


def batch_id_for(batch: list[dict[str, Any]], config: Config) -> str:
    payload = {
        "model": config.model,
        "batch_size": config.batch_size,
        "system_prompt_hash": stable_hash(SYSTEM_PROMPT),
        "clauses": [
            {
                "id": item["representative_clause_id"],
                "text": item["clause_text"],
            }
            for item in batch
        ],
    }
    return stable_hash(payload)[:24]


def build_user_message(batch: list[dict[str, Any]]) -> str:
    lines = [
        "Analyze the following legal clauses. Return one candidate per independently checkable obligation.",
        "Do not create candidates for clauses that are not policy-checkable. If none are checkable, return [].",
        "Each source_clause_ids list may contain only one representative clause ID from this batch.",
        "\nCLAUSES:",
    ]
    for item in batch:
        lines.append(
            f"[{item['representative_clause_id']}] "
            f"ACT={item.get('source_act', '')} "
            # SOURCE/filename intentionally omitted: the model copies filenames
            # into the 'source' field, violating citation rules 8-10. The Act
            # name is sufficient context; section numbers come from clause text.
            f"PAGE={item.get('page_start', '')} "
            f"SECTION={item.get('section_heading', '')}\n"
            f"{item['clause_text']}"
        )
    lines.append("\nReturn only the JSON array.")
    return "\n".join(lines)


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
    raise ValueError("Unterminated JSON array in model response")


def extract_json_array(text: str) -> list[Any]:
    # Strip thinking-model tags before any other processing.
    text = _strip_thinking_tags(text)

    # Fast path: entire response is already a JSON array.
    if text.startswith("[") and text.endswith("]"):
        return json.loads(text)

    # Second fast path: look for the first '[' and parse a balanced array.
    start = text.find("[")
    if start != -1:
        try:
            return _find_balanced_array(text, start)
        except (ValueError, json.JSONDecodeError):
            pass

    # Fallback: model may have wrapped the array inside a JSON object
    # (common when Ollama's format="json" constraint is active or the model
    # decides to wrap for structural reasons, e.g. {"items": [...]} ).
    obj_start = text.find("{")
    if obj_start != -1:
        # Find the first '[' inside the object.
        array_start = text.find("[", obj_start)
        if array_start != -1:
            try:
                return _find_balanced_array(text, array_start)
            except (ValueError, json.JSONDecodeError):
                pass
        # Try to parse the whole object and pull any list-valued key.
        try:
            obj = json.loads(text[obj_start:])
            if isinstance(obj, dict):
                for value in obj.values():
                    if isinstance(value, list):
                        return value
        except (ValueError, json.JSONDecodeError):
            pass

    raise ValueError("Model response contains no JSON array")


def validate_entry(entry: Any, allowed_ids: set[str]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(entry, dict):
        return None, "entry is not an object"

    required = {
        "category", "requirement", "source", "weight", "source_clause_ids",
        "checkable_test", "applicability", "assessability",
        "citation_confidence", "notes",
    }
    missing = sorted(required - set(entry))
    if missing:
        return None, f"missing fields: {missing}"

    category = str(entry.get("category", "")).strip()
    requirement = str(entry.get("requirement", "")).strip()
    source = str(entry.get("source", "")).strip()
    checkable_test = str(entry.get("checkable_test", "")).strip()
    applicability = str(entry.get("applicability", "")).strip()
    assessability = str(entry.get("assessability", "")).strip()
    citation_confidence = str(entry.get("citation_confidence", "")).strip().casefold()
    notes = str(entry.get("notes", "")).strip()

    if category not in ALLOWED_CATEGORIES:
        return None, f"invalid category: {category!r}"
    if not requirement or len(requirement) < 15:
        return None, "requirement is empty or too short"
    if len(requirement) > 1000:
        return None, "requirement is unusually long"
    if not checkable_test:
        return None, "checkable_test is empty"
    if not applicability:
        return None, "applicability is empty"
    if not assessability:
        return None, "assessability is empty"
    if citation_confidence not in {"high", "medium", "low", "missing", "unknown"}:
        return None, f"invalid citation_confidence: {citation_confidence!r}"

    # Enforce citation rules 8-10: filenames are never valid legal citations.
    if source and _FILENAME_SOURCE_RE.search(source):
        return None, (
            f"source looks like a filename, not a legal citation: {source!r}. "
            "Use the Act/Rule name and section number (e.g. 'DPDP Act 2023, Section 8(1)')."
        )

    try:
        weight = int(entry.get("weight"))
    except (TypeError, ValueError):
        return None, "weight is not an integer"
    if weight not in ALLOWED_WEIGHTS:
        return None, f"weight must be one of {sorted(ALLOWED_WEIGHTS)}"

    source_ids = entry.get("source_clause_ids")
    if not isinstance(source_ids, list) or not source_ids:
        return None, "source_clause_ids must be a non-empty list"
    source_ids = [str(value).strip() for value in source_ids]
    if any(not value for value in source_ids):
        return None, "source_clause_ids contains an empty ID"
    unknown_ids = sorted(set(source_ids) - allowed_ids)
    if unknown_ids:
        return None, f"source_clause_ids contains IDs outside this batch: {unknown_ids}"
    if len(set(source_ids)) != len(source_ids):
        return None, "source_clause_ids contains duplicates"

    normalized = {
        "category": category,
        "requirement": requirement,
        "source": source,
        "weight": weight,
        "source_clause_ids": sorted(set(source_ids)),
        "checkable_test": checkable_test,
        "applicability": applicability,
        "assessability": assessability,
        "citation_confidence": citation_confidence,
        "notes": notes,
        "human_review_status": "PENDING",
    }
    return normalized, None


def call_model(batch: list[dict[str, Any]], batch_number: int, total: int, config: Config) -> BatchResult:
    if ollama is None:
        raise RuntimeError("ollama package is not installed; run pip install ollama")

    allowed_ids = {item["representative_clause_id"] for item in batch}
    batch_id = batch_id_for(batch, config)
    user_message = build_user_message(batch)
    raw_response = ""
    last_error: str | None = None

    print(f"  batch {batch_number}/{total}: {len(batch)} clauses, id={batch_id}")
    for attempt in range(1, config.max_retries + 1):
        try:
            response = ollama.chat(
                model=config.model,
                think=False,
                # Do NOT pass format="json": Ollama's JSON-object grammar
                # conflicts with the JSON *array* the prompt requests and
                # causes the model to wrap output in an object like
                # {"items": [...]}, breaking extract_json_array.
                options={
                    "num_predict": config.num_predict,
                    "temperature": config.temperature,
                },
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw_response = str(response["message"]["content"]).strip()
            parsed = extract_json_array(raw_response)
            valid: list[dict[str, Any]] = []
            invalid: list[dict[str, Any]] = []
            for position, entry in enumerate(parsed):
                normalized, error = validate_entry(entry, allowed_ids)
                if normalized is None:
                    invalid.append({
                        "position": position,
                        "entry": entry,
                        "error": error,
                    })
                else:
                    valid.append(normalized)
            # --- Embedding-aware conflict resolution for multi-mapped clause IDs ---
            # A clause_id appearing in >1 valid entry may represent:
            #   (a) Entries in DIFFERENT categories  → distinct obligations from a
            #       compound source clause → KEEP ALL (log for audit).
            #   (b) Entries in the SAME category     → likely paraphrase duplicates
            #       → use cosine similarity to keep only semantically distinct ones.
            # This replaces the former "discard ALL" logic that silently lost
            # substantive content from every compound clause in the corpus.
            source_to_positions: dict[str, list[int]] = defaultdict(list)
            for position, entry in enumerate(valid):
                for source_id in entry["source_clause_ids"]:
                    source_to_positions[source_id].append(position)

            multi_mapped_log: list[dict[str, Any]] = []
            discard_positions: set[int] = set()

            for source_id, positions in source_to_positions.items():
                if len(positions) <= 1:
                    continue

                groups_by_category: dict[str, list[int]] = defaultdict(list)
                for pos in positions:
                    groups_by_category[valid[pos]["category"]].append(pos)

                kept: list[int] = []
                discarded: list[int] = []

                for category, cat_positions in groups_by_category.items():
                    if len(cat_positions) == 1:
                        kept.append(cat_positions[0])
                        continue
                    # Same category, multiple entries — use embedding similarity
                    # to distinguish genuine duplicates from distinct obligations.
                    embed_model = _get_dedup_model()
                    if embed_model is None:
                        # sentence-transformers not available: keep first, discard rest
                        kept.append(cat_positions[0])
                        discarded.extend(cat_positions[1:])
                        continue
                    texts = [valid[p]["requirement"] for p in cat_positions]
                    embeddings = embed_model.encode(
                        texts, convert_to_tensor=True, normalize_embeddings=True
                    )
                    local_kept = [cat_positions[0]]
                    for i in range(1, len(cat_positions)):
                        is_dup = any(
                            float(
                                util.cos_sim(
                                    embeddings[i],
                                    embeddings[cat_positions.index(k)],
                                ).item()
                            ) >= SAME_CATEGORY_DUPLICATE_THRESHOLD
                            for k in local_kept
                        )
                        if is_dup:
                            discarded.append(cat_positions[i])
                        else:
                            local_kept.append(cat_positions[i])
                    kept.extend(local_kept)

                discard_positions.update(discarded)
                decision = (
                    "KEPT_ALL_DISTINCT_CATEGORIES"
                    if len(groups_by_category) == len(positions)
                    else "KEPT_FIRST_PER_CATEGORY_OR_DEDUPED_BY_EMBEDDING"
                )
                multi_mapped_log.append({
                    "batch_id": batch_id,
                    "source_clause_id": source_id,
                    "total_entries": len(positions),
                    "distinct_categories": len(groups_by_category),
                    "categories": list(groups_by_category.keys()),
                    "decision": decision,
                    "kept_positions": sorted(kept),
                    "discarded_positions": sorted(discarded),
                    "entries_summary": [
                        {
                            "position": pos,
                            "category": valid[pos]["category"],
                            "requirement": valid[pos]["requirement"][:120],
                        }
                        for pos in positions
                    ],
                })

            if discard_positions:
                invalid.extend(
                    {
                        "position": pos,
                        "entry": valid[pos],
                        "error": (
                            "discarded as same-category, semantically similar duplicate "
                            "of another entry sharing this source clause"
                        ),
                    }
                    for pos in sorted(discard_positions)
                )
                valid = [
                    entry
                    for position, entry in enumerate(valid)
                    if position not in discard_positions
                ]

            return BatchResult(
                batch_id=batch_id,
                status="SUCCESS" if not invalid else "PARTIAL_VALIDATION",
                input_clause_ids=sorted(allowed_ids),
                raw_response=raw_response,
                parsed_entries=valid,
                invalid_records=invalid,
                error=None,
                attempts=attempt,
                model=config.model,
                created_at=utc_now_iso(),
                multi_mapped_records=multi_mapped_log,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"    attempt {attempt}/{config.max_retries} failed: {last_error}", file=sys.stderr)
            if attempt < config.max_retries:
                time.sleep(config.retry_delay * attempt)

    return BatchResult(
        batch_id=batch_id,
        status="FAILED",
        input_clause_ids=sorted(allowed_ids),
        raw_response=raw_response,
        parsed_entries=[],
        invalid_records=[],
        error=last_error or "unknown model error",
        attempts=config.max_retries,
        model=config.model,
        created_at=utc_now_iso(),
        multi_mapped_records=[],
    )


def expand_source_ids(
    entries: list[dict[str, Any]],
    rep_lookup: dict[str, list[str]],
) -> list[dict[str, Any]]:
    expanded_entries: list[dict[str, Any]] = []
    for entry in entries:
        expanded = []
        representative_ids = entry.get("source_clause_ids", [])
        for clause_id in representative_ids:
            expanded.extend(rep_lookup.get(clause_id, [clause_id]))
        copy = dict(entry)
        copy["representative_source_clause_ids"] = sorted(set(representative_ids))
        copy["source_clause_ids"] = sorted(set(expanded))
        expanded_entries.append(copy)
    return expanded_entries


def deduplicate_entries(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove exact duplicate model outputs while preserving provenance."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = stable_hash({
            "category": entry["category"],
            "requirement": canonical_text(entry["requirement"]),
            "source": canonical_text(entry["source"]),
        })
        grouped[key].append(entry)

    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for key, group in grouped.items():
        base = dict(group[0])
        all_ids = sorted({cid for item in group for cid in item.get("source_clause_ids", [])})
        base["source_clause_ids"] = all_ids
        base["candidate_id"] = f"TC-{len(unique) + 1:06d}"
        base["duplicate_model_output_count"] = len(group)
        unique.append(base)
        if len(group) > 1:
            duplicates.append({
                "deduplication_key": key,
                "kept_candidate": base,
                "duplicate_entries": group[1:],
            })
    return unique, duplicates


def find_merge_candidates(entries: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    if len(entries) < 2:
        return []
    if SentenceTransformer is None or util is None:
        print("sentence-transformers unavailable; merge candidates not generated", file=sys.stderr)
        return []

    print("Embedding candidate requirements for manual merge review...")
    model = SentenceTransformer(config.embedding_model)
    texts = [entry["requirement"] for entry in entries]
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    candidates: list[dict[str, Any]] = []
    for left, right in combinations(range(len(entries)), 2):
        # Compare only within the same category. Similarity across categories
        # is useful for analysis but not a safe merge suggestion.
        if entries[left]["category"] != entries[right]["category"]:
            continue
        score = float(util.cos_sim(embeddings[left], embeddings[right]).item())
        if score >= config.merge_threshold:
            candidates.append({
                "similarity": round(score, 6),
                "candidate_1": entries[left],
                "candidate_2": entries[right],
                "merge_decision": "PENDING_HUMAN_REVIEW",
                "reason": "semantic similarity within the same legal category; no automatic merge performed",
            })
    candidates.sort(key=lambda item: -item["similarity"])
    return candidates


def load_cached_result(path: Path, expected_batch_id: str, config: Config) -> BatchResult | None:
    if config.force or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Backward-compat: older cache files predate the multi_mapped_records field.
        payload.setdefault("multi_mapped_records", [])
        result = BatchResult(**payload)
        if result.batch_id != expected_batch_id:
            return None
        if result.status == "FAILED":
            return None
        return result
    except Exception:
        return None


def batch_manifest_entry(result: BatchResult, batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batch_id": result.batch_id,
        "status": result.status,
        "attempts": result.attempts,
        "input_clause_ids": result.input_clause_ids,
        "input_count": len(batch),
        "valid_entry_count": len(result.parsed_entries),
        "invalid_entry_count": len(result.invalid_records),
        "error": result.error,
        "created_at": result.created_at,
        "model": result.model,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive", file=sys.stderr)
        return 1
    if not 0.0 <= args.merge_threshold <= 1.0:
        print("ERROR: --merge-threshold must be between 0 and 1", file=sys.stderr)
        return 1

    config = Config(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        max_retries=max(1, args.max_retries),
        retry_delay=max(0.0, args.retry_delay),
        merge_threshold=args.merge_threshold,
        temperature=args.temperature,
        num_predict=args.num_predict,
        force=args.force,
    )

    try:
        dataframe = load_input(config.input_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    representatives, rep_lookup = deduplicate_representatives(dataframe)
    ordered = order_representatives(representatives)
    batches = [ordered[index:index + config.batch_size] for index in range(0, len(ordered), config.batch_size)]
    cache_dir = config.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(dataframe)} government clauses")
    print(f"Model representatives: {len(representatives)}")
    print(f"Batches: {len(batches)}")

    all_entries: list[dict[str, Any]] = []
    all_invalid: list[dict[str, Any]] = []
    all_multi_mapped: list[dict[str, Any]] = []
    batch_manifest: list[dict[str, Any]] = []
    failed_batches: list[str] = []

    for number, batch in enumerate(batches, start=1):
        expected_batch_id = batch_id_for(batch, config)
        cache_path = cache_dir / f"batch_{expected_batch_id}.json"
        result = load_cached_result(cache_path, expected_batch_id, config)
        if result is not None:
            print(f"  cached batch {number}/{len(batches)}: {expected_batch_id}")
        else:
            result = call_model(batch, number, len(batches), config)
            write_json(cache_path, asdict(result))

        all_entries.extend(result.parsed_entries)
        all_invalid.extend([
            {
                "batch_id": result.batch_id,
                "input_clause_ids": result.input_clause_ids,
                **invalid,
            }
            for invalid in result.invalid_records
        ])
        all_multi_mapped.extend(result.multi_mapped_records)
        batch_manifest.append(batch_manifest_entry(result, batch))
        if result.status == "FAILED":
            failed_batches.append(result.batch_id)

    expanded_entries = expand_source_ids(all_entries, rep_lookup)
    normalized_entries, duplicate_outputs = deduplicate_entries(expanded_entries)

    # Build a source coverage report. It is expected that many legal clauses
    # are unmapped, but they must be explicitly listed rather than disappearing.
    mapped_ids = {
        clause_id
        for entry in normalized_entries
        for clause_id in entry.get("source_clause_ids", [])
    }
    all_input_ids = set(dataframe["clause_id"].astype(str))
    unmapped_ids = sorted(all_input_ids - mapped_ids)
    clause_lookup = dataframe.set_index("clause_id", drop=False).to_dict(orient="index")
    unmapped = [clause_lookup[clause_id] for clause_id in unmapped_ids]

    merge_candidates = find_merge_candidates(normalized_entries, config)

    summary = {
        "run_timestamp_utc": utc_now_iso(),
        "input_path": str(config.input_path),
        "input_clause_count": len(dataframe),
        "representative_clause_count": len(representatives),
        "batch_count": len(batches),
        "successful_or_partial_batches": sum(1 for item in batch_manifest if item["status"] != "FAILED"),
        "failed_batch_count": len(failed_batches),
        "raw_model_entry_count": len(all_entries),
        "normalized_candidate_count": len(normalized_entries),
        "invalid_model_record_count": len(all_invalid),
        "duplicate_model_output_group_count": len(duplicate_outputs),
        "unmapped_input_clause_count": len(unmapped),
        "merge_candidate_pair_count": len(merge_candidates),
        "mapped_input_clause_count": len(mapped_ids),
        "coverage_percent": round((len(mapped_ids) / len(all_input_ids) * 100), 3) if all_input_ids else 0.0,
        "model": config.model,
        "embedding_model": config.embedding_model,
        "taxonomy_status": "DRAFT_PENDING_HUMAN_LEGAL_REVIEW",
        "warning": "No candidate is legally approved or automatically merged by this script.",
    }

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "taxonomy_candidates_normalized.json", normalized_entries)
    write_json(output_dir / "taxonomy_run_summary.json", summary)
    print("\n=== Taxonomy-generation summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutputs written to: {output_dir}")

    if failed_batches:
        print("\nWARNING: failed batches remain. Re-run without --force to retry them.", file=sys.stderr)
        return 2
    if all_invalid:
        print("\nWARNING: some model records failed validation. Inspect taxonomy_invalid_model_records.json.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
