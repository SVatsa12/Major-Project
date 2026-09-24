"""
Resumable, cache-backed bootstrap labeling of social-media policy clauses.

Labels curated policy clauses against the finalized multi-act compliance taxonomy.
Uses targeted candidate gating to prevent context degradation and eliminate hallucinations.

v11 changes
-----------
* CATEGORY_KEYWORD_MAP expanded with platform-native phrasing for IT Act categories.
* Children/Vulnerable Groups gating: bare child/parent/minor keywords require at least
  one regulatory-context companion word to score, preventing YouTube Kids section inflation.
* Per-category candidate cap (max 2 rules per category) to prevent category monopoly.
* Semantic embedding fallback replaces the hardcoded 3-category static fallback.
* --platform CLI flag for targeted re-labeling of specific platforms with merge-back.
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

import numpy as np
import ollama
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:  # graceful degradation if library not installed
    _ST_AVAILABLE = False

DEFAULT_CORPUS = Path("datasets/SocialMediaPolicies/social_media_clauses_for_labeling.csv")
DEFAULT_TAXONOMY = Path("corpus/dpdp_taxonomy_final.json")
DEFAULT_OUTPUT = Path("corpus/labeled_clauses_bootstrap.csv")
DEFAULT_CACHE_DIR = Path("corpus/label_cache")
DEFAULT_FAILURES = Path("corpus/labeled_clauses_failures.json")
DEFAULT_RUN_STATE = Path("corpus/labeled_clauses_run_state.json")

PROMPT_VERSION = "policy-labeling-v11-debiased-semantic"
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

# ---------------------------------------------------------------------------
# Keyword routing dictionary for high-precision candidate selection.
# v11: expanded with platform-native phrasing so IT Act and Cross-Border rules
# fire on real policy language instead of regulatory jargon.
# ---------------------------------------------------------------------------
CATEGORY_KEYWORD_MAP: dict[str, list[str]] = {
    # --- Regulatory-gated child keywords -------------------------------------------
    # Bare words like "child", "parent", "minor" appear in generic content descriptions
    # (e.g., YouTube Kids show blurbs). Only multi-word compound phrases carry genuine
    # regulatory signal and are listed here. The select_candidate_rules() function
    # additionally requires at least one REGULATORY_CHILD_CONTEXT word to be present
    # in the clause before awarding category-keyword points for this category.
    "Children/Vulnerable Groups": [
        "parental consent", "guardian consent", "verifiable parental consent",
        "age verification", "age limit", "age of majority", "under 18",
        "underage", "child protection", "detrimental to child",
        "minor's data", "data of children", "child safety", "child account",
        "family link", "supervised experience", "children's privacy",
        "protect minors", "guardian permission", "tracking of children",
    ],
    "Data Principal Rights": [
        "access", "download", "export", "portability", "correct", "rectif",
        "update your information", "copy of data", "review your data",
        "my activity", "takeout", "manage your info", "right to erasure",
        "data subject request", "right to access", "right to correct",
        "right to object", "withdraw consent",
    ],
    "Data Retention & Erasure": [
        "delet", "eras", "retention", "retain", "retention period",
        "storage period", "wipe", "how long we keep", "remove your account",
        "deactivate",
        # Platform-native phrasing:
        "keep your information", "delete your account", "stored until",
        "duration of your account", "retention schedule", "how long we retain",
        "no longer necessary", "kept for", "purge", "data lifecycle",
    ],
    "Consent & Notice": [
        "consent", "withdraw", "revoke", "opt out", "opt-out", "notice",
        "purpose", "privacy policy", "agree to this", "your choices", "permission",
        "lawful basis", "legal basis", "processing purpose", "specific purpose",
    ],
    "Grievance Redressal": [
        "grievance", "redress", "officer", "nodal", "complaint", "contact us",
        "dpo", "dispute", "data protection officer", "timeline", "india grievance",
    ],
    "Breach Notification": [
        "breach", "incident", "leak", "unauthorized disclosure", "notify users",
        "cert-in", "board notification",
        # Platform-native phrasing:
        "security incident", "compromise", "incident response", "data leak",
        "compromised account", "account compromise", "security breach",
        "notify affected", "security event",
    ],
    "Cross-Border Transfer": [
        "transfer", "cross-border", "overseas", "outside india", "international",
        "adequacy", "jurisdiction", "data transfer",
        # Platform-native phrasing:
        "global infrastructure", "servers located", "internationally",
        "transfer your data", "outside your country", "other countries",
        "data centres outside", "stored globally", "processed in",
    ],
    "Intermediary/Platform Liability": [
        "takedown", "blocking", "government direction", "prohibit",
        "unlawful", "due diligence", "remove content", "rule 3", "infringing",
        # Platform-native phrasing for law-enforcement compliance clauses:
        "law enforcement", "law enforcement request", "legal process",
        "government request", "government agency", "court order",
        "valid legal request", "subpoena", "statutory obligation",
        "public authority", "legal demand", "national security",
        "regulatory authority", "compelled by law", "legal obligation",
    ],
    "Security Safeguards": [
        "security", "encrypt", "safeguard", "technical measures",
        "access control", "unauthorized access", "log", "logs", "monitor",
        "vulnerability", "password", "two-factor", "multi-factor",
        "penetration test", "security review", "ssl", "tls", "at rest",
        "in transit", "pseudonymis",
    ],
    "Significant Data Fiduciary Obligations": [
        "fiduciary", "processor", "impact assessment", "dpia", "undertaking",
        "significant data fiduciary", "data protection impact",
    ],
}

# Regulatory-context words required to score bare child/parent/minor keywords.
# A clause containing "child" or "parent" MUST also contain at least one of these
# to earn category-keyword points for Children/Vulnerable Groups.  This prevents
# generic video-platform content descriptions from inflating the category score.
_CHILD_REGULATORY_CONTEXT: frozenset[str] = frozenset({
    "consent", "age", "guardian", "protection", "minor", "underage",
    "parental", "tracking", "safety", "restrict", "limit", "verify",
    "account", "data", "permission", "supervised",
})


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
    platform_filter: frozenset[str]  # empty = all platforms


class LabelingError(Exception):
    pass


class ValidationError(LabelingError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap labeling with dynamic candidate gating (v11).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
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
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        metavar="NAME[,NAME...]",
        help=(
            "Comma-separated list of platform app_name values to re-label.\n"
            "  Example: --platform Meta,Youtube,Telegram\n"
            "Only clauses belonging to these platforms are processed; results\n"
            "are merged back into --output (bootstrap CSV) without duplicating\n"
            "clause_id rows that already exist for other platforms."
        ),
    )
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


# ---------------------------------------------------------------------------
# Candidate selection constants
# ---------------------------------------------------------------------------

# Minimum aggregate score a rule must reach to be promoted as a keyword candidate.
# Score of 4 = one required_concept hit (4 pts) or ≥2 category keyword hits (3 pts each).
MIN_SCORE_THRESHOLD = 4

# Hard ceiling on how many rules from the same category can enter the candidate
# set from keyword scoring alone.  Prevents a single keyword-rich category
# (e.g. Children/Vulnerable Groups) from monopolising all 6 slots.
MAX_CANDIDATES_PER_CATEGORY = 2

# Total candidates passed to the LLM prompt.  Must stay small enough to fit
# comfortably inside qwen3:8b's context without degrading output quality.
MAX_TOTAL_CANDIDATES = 6

# Minimum candidates we want before triggering the semantic embedding fallback.
_MIN_CANDIDATES_BEFORE_FALLBACK = 3

# ---------------------------------------------------------------------------
# Module-level taxonomy embedding cache (populated once on first call)
# ---------------------------------------------------------------------------
_taxonomy_embedder: "SentenceTransformer | None" = None
_taxonomy_embed_cache: dict[str, np.ndarray] = {}  # taxonomy_id -> embedding vector


def _get_embedder() -> "SentenceTransformer | None":
    """Lazy-load the sentence-transformer model exactly once."""
    global _taxonomy_embedder
    if not _ST_AVAILABLE:
        return None
    if _taxonomy_embedder is None:
        try:
            _taxonomy_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Could not load SentenceTransformer: {exc}", file=sys.stderr)
            return None
    return _taxonomy_embedder


def _embed_taxonomy(taxonomy: list[dict[str, Any]]) -> None:
    """Pre-compute and cache embeddings for every taxonomy rule (requirement + checkable_test)."""
    embedder = _get_embedder()
    if embedder is None:
        return
    uncached = [r for r in taxonomy if r["taxonomy_id"] not in _taxonomy_embed_cache]
    if not uncached:
        return
    texts = [
        f"{r['requirement']} {r['checkable_test']}".strip()
        for r in uncached
    ]
    try:
        vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for rule, vec in zip(uncached, vectors):
            _taxonomy_embed_cache[rule["taxonomy_id"]] = vec
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Embedding pre-computation failed: {exc}", file=sys.stderr)


def _semantic_top_k(
    clause_text: str,
    taxonomy: list[dict[str, Any]],
    exclude_ids: set[str],
    k: int,
) -> list[dict[str, Any]]:
    """
    Return up to *k* taxonomy rules whose requirement+checkable_test embedding
    is most similar to *clause_text*, skipping any rule_id already in *exclude_ids*.
    Returns an empty list if sentence-transformers is unavailable.
    """
    embedder = _get_embedder()
    if embedder is None or not _taxonomy_embed_cache:
        return []

    try:
        clause_vec = embedder.encode(clause_text, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for rule in taxonomy:
        rid = rule["taxonomy_id"]
        if rid in exclude_ids:
            continue
        tax_vec = _taxonomy_embed_cache.get(rid)
        if tax_vec is None:
            continue
        sim = float(np.dot(clause_vec, tax_vec))  # both normalised → cosine similarity
        scored.append((sim, rule))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def select_candidate_rules(
    clause_text: str,
    taxonomy: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Dynamically route the top-relevant taxonomy rules for this clause.

    Algorithm
    ---------
    1. Score every rule via keyword matching (with child-keyword regulatory gating).
    2. Apply per-category cap (MAX_CANDIDATES_PER_CATEGORY) to prevent monopoly.
    3. Take the top MAX_TOTAL_CANDIDATES by score.
    4. If fewer than _MIN_CANDIDATES_BEFORE_FALLBACK rules cleared the threshold,
       fill remaining slots using semantic (cosine) similarity against cached
       taxonomy embeddings.  This replaces the old hardcoded 3-category static
       fallback which permanently starved IT Act and Cross-Border rules.
    """
    lower_text = clause_text.lower()
    tokens_in_clause: frozenset[str] = frozenset(lower_text.split())

    # ── Detect whether the clause has any regulatory child-protection context ──
    # Bare words like "child", "parent", "minor" appear in generic content
    # (video titles, app descriptions).  We only award category-keyword points for
    # Children/Vulnerable Groups when at least one regulatory-context word is also
    # present — making the gating compound rather than single-keyword.
    _bare_child_words: frozenset[str] = frozenset({"child", "children", "parent", "parents", "minor", "minors"})
    _clause_has_child_regulatory_context: bool = bool(
        _bare_child_words & tokens_in_clause
        and _CHILD_REGULATORY_CONTEXT & tokens_in_clause
    )

    # ── Step 1: keyword + concept scoring ────────────────────────────────────
    # category -> [(score, rule), ...] — tracked separately to enforce per-cat cap
    category_buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for rule in taxonomy:
        score = 0
        cat = rule["category"]

        # Category keyword points (3 pts per hit)
        if cat == "Children/Vulnerable Groups":
            # Only award keyword points when regulatory context is present in clause.
            if _clause_has_child_regulatory_context:
                for kw in CATEGORY_KEYWORD_MAP.get(cat, []):
                    if kw in lower_text:
                        score += 3
        else:
            for kw in CATEGORY_KEYWORD_MAP.get(cat, []):
                if kw in lower_text:
                    score += 3

        # Required-concept points (4 pts per hit — strongest signal)
        for concept in (
            c.strip().lower()
            for c in rule.get("required_concepts", "").split(";")
            if c.strip()
        ):
            if concept in lower_text:
                score += 4

        # Raw requirement token points (1 pt per token >4 chars — tiebreaker)
        for token in rule["requirement"].lower().split():
            if len(token) > 4 and token in lower_text:
                score += 1

        if score >= MIN_SCORE_THRESHOLD:
            category_buckets.setdefault(cat, []).append((score, rule))

    # ── Step 2: per-category cap — sort within category, keep top-2 ─────────
    scored_candidates: list[tuple[int, dict[str, Any]]] = []
    for cat, entries in category_buckets.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        scored_candidates.extend(entries[:MAX_CANDIDATES_PER_CATEGORY])

    # ── Step 3: global top-N ─────────────────────────────────────────────────
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    candidates: list[dict[str, Any]] = [r for _, r in scored_candidates[:MAX_TOTAL_CANDIDATES]]

    # ── Step 4: semantic embedding fallback ──────────────────────────────────
    # Triggered whenever keyword scoring returned fewer than the minimum threshold.
    # Fills remaining slots with the highest-cosine-similarity taxonomy rules,
    # skipping rules already in the keyword-selected candidate set.
    if len(candidates) < _MIN_CANDIDATES_BEFORE_FALLBACK:
        already_selected = {r["taxonomy_id"] for r in candidates}
        needed = MAX_TOTAL_CANDIDATES - len(candidates)
        semantic_extras = _semantic_top_k(clause_text, taxonomy, already_selected, k=needed)
        if semantic_extras:
            candidates.extend(semantic_extras)
        elif not candidates:
            # Last resort: if both keyword AND semantic fallbacks yield nothing
            # (e.g. sentence-transformers not installed), use a minimal breadth set
            # that covers the most common positive categories across the corpus.
            fallback_cats = {
                "Consent & Notice", "Data Principal Rights",
                "Security Safeguards", "Data Retention & Erasure",
            }
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

    # Parse --platform filter into a normalised frozenset (case-insensitive)
    platform_filter: frozenset[str] = frozenset()
    if args.platform:
        platform_filter = frozenset(
            p.strip().lower() for p in args.platform.split(",") if p.strip()
        )

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
        platform_filter=platform_filter,
    )

    corpus_df, corpus_hash = load_corpus(config.corpus)
    taxonomy, taxonomy_hash = load_taxonomy(config.taxonomy)
    taxonomy_lookup = {item["taxonomy_id"]: item for item in taxonomy}

    # Pre-compute taxonomy embeddings now (once) so select_candidate_rules()
    # can use them without re-encoding per-clause.
    _embed_taxonomy(taxonomy)
    if _ST_AVAILABLE and _taxonomy_embed_cache:
        print(f"[INFO] Taxonomy embeddings cached for {len(_taxonomy_embed_cache)} rules.")
    elif not _ST_AVAILABLE:
        print("[WARN] sentence-transformers not available — semantic fallback disabled.")

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    run_fingerprint = sha256_json({
        "corpus_hash": corpus_hash,
        "taxonomy_hash": taxonomy_hash,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
    })
    run_id = (
        f"run-{run_fingerprint[:12]}-{uuid.uuid4().hex[:8]}"
        if config.force_new_run
        else f"run-{run_fingerprint[:16]}"
    )

    # ── Apply platform filter ────────────────────────────────────────────────
    # If --platform is supplied, only re-label clauses from those platforms.
    # All other clauses are carried forward unchanged from the existing output
    # file (if it exists) to allow cheap targeted re-runs.
    if config.platform_filter:
        filtered_names = sorted(config.platform_filter)
        if "app_name" not in corpus_df.columns:
            print(
                "[WARN] --platform supplied but corpus has no 'app_name' column; "
                "ignoring filter and processing all clauses.",
                file=sys.stderr,
            )
            work_df = corpus_df
        else:
            app_name_lower = corpus_df["app_name"].str.lower()
            mask = app_name_lower.isin(config.platform_filter)
            work_df = corpus_df[mask].copy()
            skipped = corpus_df[~mask].copy()
            print(
                f"[INFO] Platform filter active: {filtered_names}\n"
                f"       Processing {len(work_df)} clauses | "
                f"Skipping {len(skipped)} clauses from other platforms."
            )
    else:
        work_df = corpus_df
        skipped = pd.DataFrame()  # empty; no rows to carry forward

    rows = work_df.to_dict(orient="records")
    all_labels: dict[str, dict[str, Any]] = {}
    print(f"Starting labeling run {run_id}  (prompt_version={PROMPT_VERSION})")
    print(f"Active clauses: {len(rows)} | Taxonomy rules: {len(taxonomy)}")

    for idx, row in enumerate(rows, start=1):
        cid = row["clause_id"]
        candidates = select_candidate_rules(row["clause_text"], taxonomy)

        # Content cache key — includes candidate set so keyword-map changes
        # correctly invalidate stale cache entries from v10.
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
                    print(
                        f"Clause {cid} failed after {config.retry_max} attempts: {exc}",
                        file=sys.stderr,
                    )
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
            matched_so_far = sum(
                1 for v in all_labels.values()
                if v.get("best_match_taxonomy_id") not in {"NONE", ""}
            )
            print(f"  Processed {idx}/{len(rows)} clauses... (Positive Matches: {matched_so_far})")

    # ── Build the freshly-labeled slice ─────────────────────────────────────
    label_df = pd.DataFrame(list(all_labels.values()))
    merged = work_df.merge(label_df, on="clause_id", how="left")

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
        "justification", "label_status", "label_needs_review", "cache_key", "run_id", "model",
    ]
    output_cols = [c for c in output_cols if c in merged.columns]

    # ── Merge-back: combine re-labeled rows with unchanged platform rows ──────
    # Strategy: new labels always win over stale rows for the same clause_id.
    # Rows from platforms not in the filter are preserved as-is.
    config.output.parent.mkdir(parents=True, exist_ok=True)

    if config.platform_filter and not skipped.empty and config.output.exists():
        try:
            existing_output = pd.read_csv(config.output, dtype=str, keep_default_na=False)
            # Keep only the rows for platforms NOT in this run's filter
            if "app_name" in existing_output.columns:
                existing_other = existing_output[
                    ~existing_output["app_name"].str.lower().isin(config.platform_filter)
                ].copy()
            else:
                existing_other = existing_output.copy()
            # Concatenate: other-platform rows first, then freshly-labeled rows
            final_df = pd.concat(
                [existing_other, merged[output_cols]],
                ignore_index=True,
            )
            # Final deduplication guard: keep the last occurrence (new labels win)
            if "clause_id" in final_df.columns:
                final_df = final_df.drop_duplicates(subset="clause_id", keep="last")
            print(
                f"[INFO] Merge-back complete: "
                f"{len(existing_other)} existing rows + "
                f"{len(merged)} re-labeled rows = "
                f"{len(final_df)} total rows in output."
            )
        except Exception as exc:
            print(
                f"[WARN] Could not read existing output for merge-back ({exc}). "
                "Writing re-labeled rows only.",
                file=sys.stderr,
            )
            final_df = merged[output_cols]
    else:
        # Full run (no platform filter) or first run — just write all rows.
        final_df = merged[output_cols]

    final_df.to_csv(config.output, index=False)

    summary = {
        "run_id": run_id,
        "prompt_version": PROMPT_VERSION,
        "platform_filter": sorted(config.platform_filter) if config.platform_filter else "ALL",
        "input_clause_count": len(work_df),
        "labeled_success_count": int(
            (merged["label_status"].isin(["NEW_SUCCESS", "CACHED_SUCCESS"])).sum()
        ),
        "matched_positive_count": int(
            (merged["best_match_taxonomy_id"].isin(taxonomy_lookup)).sum()
        ),
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