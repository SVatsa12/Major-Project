"""
FINAL TAXONOMY CONSOLIDATION ENGINE - v2

Purpose
-------
Convert the draft 633-candidate taxonomy into a smaller, auditable,
project-specific taxonomy for social-media governance-document auditing.

IMPORTANT DESIGN PRINCIPLES
----------------------------
1. No arbitrary filtering.
2. No blind target-count truncation.
3. Every merge must have an auditable reason.
4. Every omission must have an auditable reason.
5. Provenance from the 2,609 cleaned government clauses is preserved.
6. Requirement meaning AND checkable-test meaning are compared.
7. Same-Act evidence is required for automatic merges.
8. Ambiguous cases go to REVIEW.
9. Target counts are project-design guidance, not legal truth.
10. No candidate is legally "approved" by this script.

Input
-----
- taxonomy_candidates_normalized.json
- government_clauses_cleaned(1).csv

Output
------
- FINAL_TAXONOMY_PROVISIONAL.json
- FINAL_TAXONOMY_PROVISIONAL.csv
- ALL_CONSOLIDATED_REQUIREMENTS.json
- MERGE_AUDIT.json
- DECISION_LOG.json
- OMITTED_CANDIDATES_REVIEW.json
- SCOPE_REVIEW_REQUIRED.json
- TARGET_REDUCTION_REVIEW.json
- PROVENANCE_REVIEW.json
- CONSOLIDATION_SUMMARY.json
- per_act/*.json

The final taxonomy remains PENDING_HUMAN_REVIEW.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.cwd()

DEFAULT_TAXONOMY = (
    PROJECT_ROOT
    / "corpus"
    / "taxonomy_generation"
    / "taxonomy_candidates_normalized.json"
)

DEFAULT_CLAUSES = (
    PROJECT_ROOT
    / "datasets"
    / "GovernmentActs"
    / "government_clauses_cleaned.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "corpus"
    / "taxonomy_generation"
    / "final_consolidation_v2"
)


# Project-specified target ranges.
#
# IMPORTANT:
# These are NOT used as arbitrary deletion rules.
# They are applied only after semantic/legal consolidation.
ACT_TARGETS = {
    "AERA_ACT_2008": (0, 1),
    "DPDP_ACT_2023": (15, 20),
    "DPDP_RULES_2025": (10, 15),
    "IT_ACT_2000": (2, 3),
    "RTI_ACT_2005": (0, 1),
    "TRAI_ACT_1997": (0, 1),
}


# ------------------------------------------------------------
# Similarity thresholds
# ------------------------------------------------------------
#
# Automatic merging is intentionally conservative.
#
# EXACT_DUPLICATE:
#     Same normalized requirement + same Act.
#
# STRONG_LEGAL_MERGE:
#     Same Act + strong semantic similarity + compatible legal
#     action/object + supporting provenance.
#
# POSSIBLE_MERGE:
#     Strong enough to warrant human review but not automatic merge.
#

AUTO_MERGE_SEMANTIC = 0.88
AUTO_MERGE_LEXICAL = 0.72

REVIEW_SEMANTIC = 0.76
REVIEW_LEXICAL = 0.55

# Exact/near-exact legal anchor threshold.
LEGAL_ANCHOR_BONUS = 0.10


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

USE_SENTENCE_TRANSFORMER = True
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_sentence_model = None

try:
    import numpy as np
except ImportError:
    np = None


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


# ============================================================
# BASIC UTILITIES
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))

    text = (
        text
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    return re.sub(r"\s+", " ", text).strip()


def canonical_text(value: Any) -> str:
    text = normalize_text(value).casefold()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", canonical_text(value)))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        canonical_text(left),
        canonical_text(right),
    ).ratio()


def stable_id(value: Any, prefix: str) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return (
        f"{prefix}-"
        f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def safe_filename(value: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )

    return cleaned.strip("_.") or "UNNAMED"


# ============================================================
# ACT NORMALIZATION
# ============================================================

def normalize_act_name(value: Any) -> str:
    text = canonical_text(value)

    if not text:
        return "UNRESOLVED"

    replacements = {
        "dpdp act 2023": "DPDP_ACT_2023",
        "digital personal data protection act 2023": "DPDP_ACT_2023",

        "dpdp rules 2025": "DPDP_RULES_2025",
        "digital personal data protection rules 2025": "DPDP_RULES_2025",

        "it act 2000": "IT_ACT_2000",
        "information technology act 2000": "IT_ACT_2000",

        "rti act 2005": "RTI_ACT_2005",
        "right to information act 2005": "RTI_ACT_2005",

        "trai act 1997": "TRAI_ACT_1997",
        "telecom regulatory authority of india act 1997": "TRAI_ACT_1997",

        "aera act 2008": "AERA_ACT_2008",
        "airports economic regulatory authority act 2008": "AERA_ACT_2008",

        "disaster management act": "DISASTER_MANAGEMENT",
        "disaster management act 2005": "DISASTER_MANAGEMENT",

        "patents act": "PATENTS_ACT",
        "patents act 1970": "PATENTS_ACT",
    }

    if text in replacements:
        return replacements[text]

    # Conservative substring handling.
    if "dpdp" in text and "rules" in text:
        return "DPDP_RULES_2025"

    if "dpdp" in text and "act" in text:
        return "DPDP_ACT_2023"

    if "information technology act" in text or "it act" in text:
        return "IT_ACT_2000"

    if "right to information" in text or "rti act" in text:
        return "RTI_ACT_2005"

    if "trai act" in text or "telecom regulatory authority" in text:
        return "TRAI_ACT_1997"

    if "aera act" in text or "airports economic regulatory" in text:
        return "AERA_ACT_2008"

    return normalize_text(value)


# ============================================================
# INPUT LOADING
# ============================================================

def load_taxonomy(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found:\n{path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            payload = payload["records"]
        else:
            raise ValueError(
                "Taxonomy JSON is a dictionary but does not contain "
                "a 'records' list."
            )

    if not isinstance(payload, list):
        raise ValueError(
            "Taxonomy JSON must be a list or contain a records list."
        )

    return [
        dict(item)
        for item in payload
        if isinstance(item, dict)
    ]


def load_clauses(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Government clause CSV not found:\n{path}"
        )

    # pandas is useful for arbitrary CSV schemas.
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError(
            "pandas is required. Install with:\n"
            "pip install pandas"
        )

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    if "clause_id" not in df.columns:
        raise ValueError(
            "Government clause CSV must contain 'clause_id'."
        )

    if "source_act" not in df.columns:
        raise ValueError(
            "Government clause CSV must contain 'source_act'."
        )

    df["clause_id"] = df["clause_id"].astype(str)
    df["source_act"] = df["source_act"].astype(str)

    duplicates = (
        df[df["clause_id"].duplicated(keep=False)]
        ["clause_id"]
        .tolist()
    )

    if duplicates:
        raise ValueError(
            "Duplicate clause IDs found: "
            f"{sorted(set(duplicates))[:20]}"
        )

    return df


def parse_source_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]

        except json.JSONDecodeError:
            pass

        return [
            part.strip()
            for part in text.split(",")
            if part.strip()
        ]

    return [str(value).strip()]


# ============================================================
# CLAUSE TEXT / LEGAL ANCHORS
# ============================================================

CLAUSE_TEXT_COLUMNS = [
    "clause_text",
    "text",
    "clause",
    "content",
    "provision",
    "description",
    "requirement",
    "source_text",
]

SECTION_COLUMNS = [
    "section",
    "section_number",
    "section_no",
    "rule",
    "rule_number",
    "rule_no",
    "sub_rule",
    "sub_section",
    "heading",
    "title",
]


def first_nonempty(row: dict[str, Any], columns: list[str]) -> str:
    for column in columns:
        if column in row:
            value = normalize_text(row[column])

            if value:
                return value

    return ""


def clause_text(row: dict[str, Any]) -> str:
    pieces = []

    for column in CLAUSE_TEXT_COLUMNS:
        if column in row:
            value = normalize_text(row[column])

            if value:
                pieces.append(value)

    # If no obvious text column exists, use the whole row.
    if not pieces:
        pieces = [
            normalize_text(value)
            for key, value in row.items()
            if key != "clause_id"
            and normalize_text(value)
        ]

    return " ".join(pieces)


def legal_anchor(row: dict[str, Any]) -> str:
    pieces = []

    for column in SECTION_COLUMNS:
        if column in row:
            value = normalize_text(row[column])

            if value:
                pieces.append(value)

    if not pieces:
        return ""

    return " | ".join(pieces)


def resolve_clause_records(
    source_ids: list[str],
    clause_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:

    records = []
    missing = []

    for clause_id in source_ids:
        row = clause_lookup.get(clause_id)

        if row is None:
            missing.append(clause_id)
            continue

        records.append(row)

    return records, missing


# ============================================================
# SOURCE / PROVENANCE
# ============================================================

def source_acts_for_records(
    records: list[dict[str, Any]]
) -> list[str]:

    acts = {
        normalize_act_name(
            record.get("source_act", "")
        )
        for record in records
    }

    return sorted(
        act
        for act in acts
        if act and act != "UNRESOLVED"
    )


def resolve_candidate_act(
    source_records: list[dict[str, Any]]
) -> str:

    acts = source_acts_for_records(source_records)

    if len(acts) == 1:
        return acts[0]

    if len(acts) > 1:
        return "MULTI_ACT__" + "__".join(acts)

    return "UNRESOLVED"


def provenance_status(
    source_ids: list[str],
    records: list[dict[str, Any]],
    missing: list[str],
) -> str:

    acts = source_acts_for_records(records)

    if not source_ids:
        return "NO_SOURCE_CLAUSES"

    if not acts:
        return "UNRESOLVED"

    if missing:
        return "PARTIAL_PROVENANCE"

    if len(acts) == 1:
        return "RESOLVED_SINGLE_ACT"

    return "RESOLVED_MULTI_ACT"


# ============================================================
# PROJECT SCOPE LOGIC
# ============================================================

# These are deliberately narrow.
#
# They are NOT general "government" filters.
#
# The project is auditing governance documents of social-media
# platforms, so authority-specific administrative duties are
# normally not document requirements for the platform itself.

HIGH_CONFIDENCE_OUT_OF_SCOPE = [
    (
        re.compile(
            r"\bcertifying authority\b"
            r"|\bcontroller of certifying authorities\b"
            r"|\blicence certifying authority\b"
            r"|\bcertifying authorities shall\b",
            re.I,
        ),
        "IT certifying-authority-specific obligation",
    ),

    (
        re.compile(
            r"\bpublic information officer\b"
            r"|\bcentral public information officer\b"
            r"|\bstate public information officer\b"
            r"|\bfirst appellate authority\b",
            re.I,
        ),
        "RTI public-authority procedural obligation",
    ),

    (
        re.compile(
            r"\bairport operator\b"
            r"|\bmajor airport\b"
            r"|\baerodrome\b"
            r"|\bairport tariff\b",
            re.I,
        ),
        "AERA/airport-specific obligation",
    ),

    (
        re.compile(
            r"\bpatent\b"
            r"|\bpatentee\b"
            r"|\bpatent application\b"
            r"|\bcontroller general of patents\b",
            re.I,
        ),
        "Patent-specific obligation",
    ),
]


# Strong government-facing concepts.
# These only create REVIEW flags.
GOVERNMENT_FACING_SIGNALS = [
    "central government",
    "state government",
    "government may",
    "government shall",
    "authority may",
    "authority shall",
    "official gazette",
    "authorized agency",
    "authorised agency",
    "law enforcement",
    "furnish information",
    "technical assistance",
    "interception",
    "decryption",
    "blocking order",
    "block access",
]


# Positive project relevance signals.
PROJECT_RELEVANCE_SIGNALS = [
    "personal data",
    "personal information",
    "data principal",
    "data fiduciary",
    "consent",
    "notice",
    "purpose",
    "collect",
    "collection",
    "process",
    "processing",
    "use",
    "share",
    "sharing",
    "disclose",
    "disclosure",
    "transfer",
    "retain",
    "retention",
    "delete",
    "deletion",
    "erase",
    "erasure",
    "security",
    "breach",
    "children",
    "child",
    "minor",
    "rights",
    "access",
    "correction",
    "grievance",
    "complaint",
    "contact",
    "cookie",
    "cookies",
    "device",
    "identifier",
    "storage",
    "encryption",
    "protect",
    "privacy",
    "copyright",
    "license",
    "licence",
    "terms",
    "content",
    "moderation",
    "suspend",
    "terminate",
    "account",
]


def find_signals(
    text: str,
    signals: list[str],
) -> list[str]:

    lowered = canonical_text(text)

    return sorted(
        signal
        for signal in signals
        if canonical_text(signal) in lowered
    )


def scope_analysis(
    entry: dict[str, Any],
    act: str,
    source_records: list[dict[str, Any]],
) -> dict[str, Any]:

    requirement = normalize_text(
        entry.get("requirement", "")
    )

    checkable = normalize_text(
        entry.get("checkable_test", "")
    )

    category = normalize_text(
        entry.get("category", "")
    )

    source = normalize_text(
        entry.get("source", "")
    )

    source_text = " ".join(
        clause_text(row)
        for row in source_records
    )

    combined = " ".join(
        [
            requirement,
            checkable,
            category,
            source,
            source_text,
        ]
    )

    out_of_scope_reasons = []

    for pattern, reason in HIGH_CONFIDENCE_OUT_OF_SCOPE:
        if pattern.search(combined):
            out_of_scope_reasons.append(reason)

    government_signals = find_signals(
        combined,
        GOVERNMENT_FACING_SIGNALS,
    )

    relevance_signals = find_signals(
        combined,
        PROJECT_RELEVANCE_SIGNALS,
    )

    # Do not automatically omit merely because the word
    # "government" occurs.
    #
    # For example, a government restriction on data transfers
    # could still matter to a platform privacy policy.
    if out_of_scope_reasons:
        scope_decision = "OMIT_REVIEW"
    elif government_signals and not relevance_signals:
        scope_decision = "SCOPE_REVIEW"
    elif relevance_signals:
        scope_decision = "PROJECT_RELEVANT"
    else:
        scope_decision = "SCOPE_REVIEW"

    return {
        "scope_decision": scope_decision,
        "out_of_scope_reasons": out_of_scope_reasons,
        "government_facing_signals": government_signals,
        "project_relevance_signals": relevance_signals,
    }


# ============================================================
# LEGAL ACTION / OBJECT EXTRACTION
# ============================================================

CORE_ACTIONS = {
    "obtain",
    "provide",
    "inform",
    "notify",
    "publish",
    "disclose",
    "share",
    "collect",
    "process",
    "use",
    "transfer",
    "retain",
    "delete",
    "erase",
    "secure",
    "protect",
    "appoint",
    "conduct",
    "assess",
    "report",
    "respond",
    "allow",
    "enable",
    "permit",
    "restrict",
    "prohibit",
    "verify",
    "authenticate",
    "register",
    "maintain",
    "review",
    "grievance",
    "complaint",
    "correct",
    "access",
    "suspend",
    "terminate",
}


def extract_actions(text: str) -> set[str]:
    lowered = canonical_text(text)

    return {
        action
        for action in CORE_ACTIONS
        if re.search(
            rf"\b{re.escape(action)}\w*\b",
            lowered,
        )
    }


OBJECT_GROUPS = {
    "consent": [
        "consent",
        "verifiable consent",
        "withdraw consent",
    ],

    "notice": [
        "notice",
        "information",
        "purpose",
        "processing purpose",
    ],

    "rights": [
        "data principal",
        "right",
        "access",
        "correction",
        "erasure",
        "grievance",
        "complaint",
    ],

    "children": [
        "child",
        "children",
        "minor",
        "guardian",
    ],

    "security": [
        "security",
        "safeguard",
        "encryption",
        "protect",
        "breach",
    ],

    "retention": [
        "retention",
        "retain",
        "delete",
        "deletion",
        "erase",
        "erasure",
    ],

    "transfer": [
        "transfer",
        "cross-border",
        "outside india",
    ],

    "cookies": [
        "cookie",
        "cookies",
        "tracking",
        "identifier",
        "device",
    ],

    "content": [
        "content",
        "moderation",
        "copyright",
        "license",
        "licence",
    ],

    "government": [
        "government",
        "authority",
        "authorized agency",
        "law enforcement",
    ],
}


def extract_object_groups(text: str) -> set[str]:
    lowered = canonical_text(text)

    found = set()

    for group, phrases in OBJECT_GROUPS.items():
        if any(
            canonical_text(phrase) in lowered
            for phrase in phrases
        ):
            found.add(group)

    return found


def semantic_compatibility(
    left_text: str,
    right_text: str,
) -> tuple[bool, list[str]]:

    left_actions = extract_actions(left_text)
    right_actions = extract_actions(right_text)

    left_objects = extract_object_groups(left_text)
    right_objects = extract_object_groups(right_text)

    reasons = []

    action_overlap = left_actions & right_actions
    object_overlap = left_objects & right_objects

    # If both have identifiable actions and there is no overlap,
    # merging is dangerous.
    if left_actions and right_actions and not action_overlap:
        reasons.append(
            "different_core_actions"
        )

        return False, reasons

    # If both identify legal objects and have no object overlap,
    # merging is also dangerous.
    if left_objects and right_objects and not object_overlap:
        reasons.append(
            "different_legal_objects"
        )

        return False, reasons

    if action_overlap:
        reasons.append(
            "shared_core_action:" +
            ",".join(sorted(action_overlap))
        )

    if object_overlap:
        reasons.append(
            "shared_legal_object:" +
            ",".join(sorted(object_overlap))
        )

    return True, reasons


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_sentence_model():
    global _sentence_model

    if not USE_SENTENCE_TRANSFORMER:
        return None

    if _sentence_model is not None:
        return _sentence_model

    try:
        from sentence_transformers import SentenceTransformer

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        _sentence_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        return _sentence_model

    except Exception as exc:
        print(
            "\nWARNING: sentence-transformers unavailable."
        )
        print(
            "Falling back to TF-IDF similarity."
        )
        print(
            f"Reason: {exc}\n"
        )

        return None


def build_comparison_text(entry: dict[str, Any]) -> str:

    requirement = normalize_text(
        entry.get("requirement", "")
    )

    checkable = normalize_text(
        entry.get("checkable_test", "")
    )

    category = normalize_text(
        entry.get("category", "")
    )

    applicability = normalize_text(
        entry.get("applicability", "")
    )

    return " ".join(
        [
            f"Requirement: {requirement}",
            f"Check: {checkable}",
            f"Category: {category}",
            f"Applicability: {applicability}",
        ]
    )


def cosine_vector_similarity(
    left,
    right,
) -> float:

    if np is None:
        return 0.0

    left = np.asarray(left)
    right = np.asarray(right)

    denominator = (
        np.linalg.norm(left)
        * np.linalg.norm(right)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(left, right) / denominator
    )


class SimilarityEngine:

    def __init__(self, entries: list[dict[str, Any]]):

        self.entries = entries

        self.model = load_sentence_model()

        self.embeddings = {}

        self.vectorizer = None
        self.tfidf_matrix = None
        self.index_by_id = {}

        for index, entry in enumerate(entries):
            candidate_id = entry["candidate_id"]
            self.index_by_id[candidate_id] = index

        self._build()

    def _build(self):

        texts = [
            build_comparison_text(entry)
            for entry in self.entries
        ]

        if self.model is not None:

            try:
                matrix = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )

                for index, entry in enumerate(self.entries):
                    self.embeddings[
                        entry["candidate_id"]
                    ] = matrix[index]

                return

            except Exception as exc:
                print(
                    "WARNING: embedding generation failed."
                )
                print(
                    f"Reason: {exc}"
                )

        # TF-IDF fallback.
        if TfidfVectorizer is not None:

            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )

            self.tfidf_matrix = (
                self.vectorizer.fit_transform(texts)
            )

    def semantic_similarity(
        self,
        left_id: str,
        right_id: str,
    ) -> float:

        if (
            left_id in self.embeddings
            and right_id in self.embeddings
        ):
            return cosine_vector_similarity(
                self.embeddings[left_id],
                self.embeddings[right_id],
            )

        if (
            self.tfidf_matrix is not None
            and left_id in self.index_by_id
            and right_id in self.index_by_id
        ):

            left_index = self.index_by_id[left_id]
            right_index = self.index_by_id[right_id]

            score = cosine_similarity(
                self.tfidf_matrix[left_index],
                self.tfidf_matrix[right_index],
            )

            return float(score[0][0])

        return 0.0


# ============================================================
# LEGAL ANCHOR EXTRACTION
# ============================================================

def candidate_legal_anchors(
    entry: dict[str, Any],
    source_records: list[dict[str, Any]],
) -> set[str]:

    anchors = set()

    source_text = normalize_text(
        entry.get("source", "")
    )

    # Candidate source often contains:
    #
    # DPDP Act 2023, Section 9(1)
    # DPDP Rules 2025, Rule 14(1)
    #
    # Capture these.
    matches = re.findall(
        r"\b(?:section|rule)\s*"
        r"[A-Za-z0-9()./-]+",
        source_text,
        flags=re.I,
    )

    for match in matches:
        anchors.add(
            canonical_text(match)
        )

    for row in source_records:

        anchor = legal_anchor(row)

        if anchor:
            anchors.add(
                canonical_text(anchor)
            )

    return {
        anchor
        for anchor in anchors
        if anchor
    }


# ============================================================
# CANDIDATE PREPARATION
# ============================================================

def prepare_candidates(
    raw_entries: list[dict[str, Any]],
    clause_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:

    prepared = []

    for index, original in enumerate(
        raw_entries,
        start=1,
    ):

        entry = dict(original)

        candidate_id = normalize_text(
            entry.get(
                "candidate_id",
                f"TC-AUTO-{index:06d}",
            )
        )

        source_ids = parse_source_ids(
            entry.get("source_clause_ids", [])
        )

        records, missing = resolve_clause_records(
            source_ids,
            clause_lookup,
        )

        acts = source_acts_for_records(records)

        resolved_act = resolve_candidate_act(
            records
        )

        scope = scope_analysis(
            entry,
            resolved_act,
            records,
        )

        requirement = normalize_text(
            entry.get("requirement", "")
        )

        checkable = normalize_text(
            entry.get("checkable_test", "")
        )

        combined_text = (
            f"{requirement} {checkable}"
        )

        prepared_entry = {
            **entry,

            "candidate_id": candidate_id,

            "source_clause_ids": source_ids,

            "resolved_act": resolved_act,

            "source_acts": acts,

            "missing_source_clause_ids": missing,

            "source_provenance_status": (
                provenance_status(
                    source_ids,
                    records,
                    missing,
                )
            ),

            "source_legal_anchors": sorted(
                candidate_legal_anchors(
                    entry,
                    records,
                )
            ),

            "source_clause_context": [
                {
                    "clause_id": row.get("clause_id", ""),
                    "source_act": row.get("source_act", ""),
                    "legal_anchor": legal_anchor(row),
                    "clause_text": clause_text(row),
                }
                for row in records
            ],

            "scope": scope,

            "_comparison_text": (
                combined_text
            ),

            "_actions": sorted(
                extract_actions(
                    combined_text
                )
            ),

            "_objects": sorted(
                extract_object_groups(
                    combined_text
                )
            ),
        }

        prepared.append(
            prepared_entry
        )

    return prepared


# ============================================================
# DUPLICATE / MERGE DECISION
# ============================================================

@dataclass
class PairDecision:
    decision: str
    reason: str
    semantic_similarity: float
    lexical_similarity: float
    requirement_similarity: float
    checkable_similarity: float
    same_act: bool
    same_category: bool
    shared_legal_anchor: bool
    compatibility: bool
    compatibility_reasons: list[str]


def compare_candidates(
    left: dict[str, Any],
    right: dict[str, Any],
    similarity_engine: SimilarityEngine,
) -> PairDecision:

    left_act = left.get("resolved_act", "")
    right_act = right.get("resolved_act", "")

    same_act = (
        left_act == right_act
        and left_act not in {
            "UNRESOLVED",
        }
        and not left_act.startswith("MULTI_ACT__")
    )

    left_category = canonical_text(
        left.get("category", "")
    )

    right_category = canonical_text(
        right.get("category", "")
    )

    same_category = (
        left_category
        == right_category
    )

    left_req = normalize_text(
        left.get("requirement", "")
    )

    right_req = normalize_text(
        right.get("requirement", "")
    )

    left_check = normalize_text(
        left.get("checkable_test", "")
    )

    right_check = normalize_text(
        right.get("checkable_test", "")
    )

    requirement_similarity = (
        sequence_similarity(
            left_req,
            right_req,
        )
    )

    checkable_similarity = (
        sequence_similarity(
            left_check,
            right_check,
        )
    )

    lexical_similarity = (
        0.65 * jaccard(
            tokens(left_req),
            tokens(right_req),
        )
        +
        0.35 * jaccard(
            tokens(left_check),
            tokens(right_check),
        )
    )

    semantic_similarity = (
        similarity_engine.semantic_similarity(
            left["candidate_id"],
            right["candidate_id"],
        )
    )

    left_anchors = set(
        left.get(
            "source_legal_anchors",
            [],
        )
    )

    right_anchors = set(
        right.get(
            "source_legal_anchors",
            [],
        )
    )

    shared_anchor = bool(
        left_anchors & right_anchors
    )

    combined_left = (
        f"{left_req} {left_check}"
    )

    combined_right = (
        f"{right_req} {right_check}"
    )

    compatible, compatibility_reasons = (
        semantic_compatibility(
            combined_left,
            combined_right,
        )
    )

    # --------------------------------------------------------
    # Rule 1: exact duplicate
    # --------------------------------------------------------

    if (
        same_act
        and canonical_text(left_req)
        == canonical_text(right_req)
        and canonical_text(left_check)
        == canonical_text(right_check)
    ):
        return PairDecision(
            decision="AUTO_MERGE",
            reason="Exact duplicate requirement and checkable test within same Act.",
            semantic_similarity=semantic_similarity,
            lexical_similarity=lexical_similarity,
            requirement_similarity=requirement_similarity,
            checkable_similarity=checkable_similarity,
            same_act=same_act,
            same_category=same_category,
            shared_legal_anchor=shared_anchor,
            compatibility=True,
            compatibility_reasons=[
                "exact_requirement_match",
                "exact_checkable_test_match",
            ],
        )

    # --------------------------------------------------------
    # Rule 2: different Acts -> never auto merge
    # --------------------------------------------------------

    if not same_act:

        if (
            semantic_similarity >= REVIEW_SEMANTIC
            and lexical_similarity >= REVIEW_LEXICAL
        ):
            return PairDecision(
                decision="REVIEW",
                reason=(
                    "High textual/semantic similarity but candidates "
                    "belong to different Acts; legal obligations must "
                    "not be merged automatically."
                ),
                semantic_similarity=semantic_similarity,
                lexical_similarity=lexical_similarity,
                requirement_similarity=requirement_similarity,
                checkable_similarity=checkable_similarity,
                same_act=False,
                same_category=same_category,
                shared_legal_anchor=shared_anchor,
                compatibility=compatible,
                compatibility_reasons=compatibility_reasons,
            )

        return PairDecision(
            decision="KEEP_SEPARATE",
            reason="Different legal Acts.",
            semantic_similarity=semantic_similarity,
            lexical_similarity=lexical_similarity,
            requirement_similarity=requirement_similarity,
            checkable_similarity=checkable_similarity,
            same_act=False,
            same_category=same_category,
            shared_legal_anchor=shared_anchor,
            compatibility=compatible,
            compatibility_reasons=compatibility_reasons,
        )

    # --------------------------------------------------------
    # Rule 3: incompatible legal action/object
    # --------------------------------------------------------

    if not compatible:

        return PairDecision(
            decision="KEEP_SEPARATE",
            reason=(
                "Requirements identify different legal actions "
                "or legal objects."
            ),
            semantic_similarity=semantic_similarity,
            lexical_similarity=lexical_similarity,
            requirement_similarity=requirement_similarity,
            checkable_similarity=checkable_similarity,
            same_act=same_act,
            same_category=same_category,
            shared_legal_anchor=shared_anchor,
            compatibility=False,
            compatibility_reasons=compatibility_reasons,
        )

    # --------------------------------------------------------
    # Rule 4: same legal anchor
    #
    # This is powerful evidence, but still requires semantic
    # compatibility.
    # --------------------------------------------------------

    if shared_anchor:

        if (
            semantic_similarity >= 0.82
            and lexical_similarity >= 0.62
        ):
            return PairDecision(
                decision="AUTO_MERGE",
                reason=(
                    "Same Act, shared legal/source anchor, and "
                    "strongly compatible requirement meaning."
                ),
                semantic_similarity=semantic_similarity,
                lexical_similarity=lexical_similarity,
                requirement_similarity=requirement_similarity,
                checkable_similarity=checkable_similarity,
                same_act=same_act,
                same_category=same_category,
                shared_legal_anchor=True,
                compatibility=True,
                compatibility_reasons=compatibility_reasons,
            )

        if (
            semantic_similarity >= REVIEW_SEMANTIC
            and lexical_similarity >= REVIEW_LEXICAL
        ):
            return PairDecision(
                decision="REVIEW",
                reason=(
                    "Same legal/source anchor but insufficient "
                    "evidence for automatic consolidation."
                ),
                semantic_similarity=semantic_similarity,
                lexical_similarity=lexical_similarity,
                requirement_similarity=requirement_similarity,
                checkable_similarity=checkable_similarity,
                same_act=same_act,
                same_category=same_category,
                shared_legal_anchor=True,
                compatibility=True,
                compatibility_reasons=compatibility_reasons,
            )

    # --------------------------------------------------------
    # Rule 5: different legal anchors
    #
    # Different sections can sometimes encode overlapping
    # requirements, but we should NOT automatically merge them.
    # --------------------------------------------------------

    if (
        semantic_similarity >= AUTO_MERGE_SEMANTIC
        and lexical_similarity >= AUTO_MERGE_LEXICAL
        and same_category
    ):

        return PairDecision(
            decision="REVIEW",
            reason=(
                "Very high semantic similarity, but legal anchors "
                "differ; human review required before merging."
            ),
            semantic_similarity=semantic_similarity,
            lexical_similarity=lexical_similarity,
            requirement_similarity=requirement_similarity,
            checkable_similarity=checkable_similarity,
            same_act=same_act,
            same_category=same_category,
            shared_legal_anchor=False,
            compatibility=True,
            compatibility_reasons=compatibility_reasons,
        )

    # --------------------------------------------------------
    # Rule 6: possible overlap
    # --------------------------------------------------------

    if (
        semantic_similarity >= REVIEW_SEMANTIC
        and lexical_similarity >= REVIEW_LEXICAL
        and same_category
    ):

        return PairDecision(
            decision="REVIEW",
            reason=(
                "Possible substantive overlap within same Act/category."
            ),
            semantic_similarity=semantic_similarity,
            lexical_similarity=lexical_similarity,
            requirement_similarity=requirement_similarity,
            checkable_similarity=checkable_similarity,
            same_act=same_act,
            same_category=same_category,
            shared_legal_anchor=False,
            compatibility=True,
            compatibility_reasons=compatibility_reasons,
        )

    return PairDecision(
        decision="KEEP_SEPARATE",
        reason="Insufficient evidence of substantive duplication.",
        semantic_similarity=semantic_similarity,
        lexical_similarity=lexical_similarity,
        requirement_similarity=requirement_similarity,
        checkable_similarity=checkable_similarity,
        same_act=same_act,
        same_category=same_category,
        shared_legal_anchor=shared_anchor,
        compatibility=compatible,
        compatibility_reasons=compatibility_reasons,
    )


# ============================================================
# CANONICAL REPRESENTATIVE SELECTION
# ============================================================

def numeric_assessability(value: Any) -> float:
    text = canonical_text(value)

    mapping = {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0,
        "3": 3.0,
        "2": 2.0,
        "1": 1.0,
    }

    if text in mapping:
        return mapping[text]

    try:
        return float(text)
    except ValueError:
        return 0.0


def numeric_confidence(value: Any) -> float:
    text = canonical_text(value)

    mapping = {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0,
    }

    return mapping.get(
        text,
        0.0,
    )


def candidate_priority(
    entry: dict[str, Any]
) -> float:

    try:
        weight = float(
            entry.get("weight", 0)
        )
    except (TypeError, ValueError):
        weight = 0.0

    assessability = (
        numeric_assessability(
            entry.get("assessability", "")
        )
    )

    confidence = (
        numeric_confidence(
            entry.get(
                "citation_confidence",
                "",
            )
        )
    )

    requirement_length = len(
        normalize_text(
            entry.get("requirement", "")
        )
    )

    checkable = len(
        normalize_text(
            entry.get("checkable_test", "")
        )
    )

    return (
        weight * 4.0
        + assessability * 2.0
        + confidence * 2.0
        + min(requirement_length / 200.0, 1.0)
        + min(checkable / 200.0, 1.0)
    )


def choose_canonical(
    members: list[dict[str, Any]]
) -> dict[str, Any]:

    return max(
        members,
        key=candidate_priority,
    )


# ============================================================
# UNION-FIND FOR AUTO MERGES
# ============================================================

class UnionFind:

    def __init__(self, ids: list[str]):

        self.parent = {
            item: item
            for item in ids
        }

        self.rank = {
            item: 0
            for item in ids
        }

    def find(self, item: str) -> str:

        if self.parent[item] != item:
            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(
        self,
        left: str,
        right: str,
    ):

        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root

        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root

        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


# ============================================================
# AUTO-CONSOLIDATION
# ============================================================

def consolidate_candidates(
    entries: list[dict[str, Any]],
    similarity_engine: SimilarityEngine,
):

    by_act_category = defaultdict(list)

    for entry in entries:

        act = entry.get(
            "resolved_act",
            "UNRESOLVED",
        )

        category = canonical_text(
            entry.get("category", "")
        )

        by_act_category[
            (act, category)
        ].append(entry)

    union_find = UnionFind(
        [
            entry["candidate_id"]
            for entry in entries
        ]
    )

    pair_audit = []

    # Compare candidates only within the same Act and category
    # for automatic consolidation.
    #
    # This prevents unrelated Acts/categories from being
    # collapsed simply because they use similar legal vocabulary.
    for (
        bucket_key,
        bucket,
    ) in by_act_category.items():

        if len(bucket) < 2:
            continue

        for i in range(len(bucket)):

            for j in range(
                i + 1,
                len(bucket),
            ):

                left = bucket[i]
                right = bucket[j]

                decision = compare_candidates(
                    left,
                    right,
                    similarity_engine,
                )

                pair_audit.append(
                    {
                        "left_candidate_id":
                            left["candidate_id"],

                        "right_candidate_id":
                            right["candidate_id"],

                        "act":
                            bucket_key[0],

                        "category":
                            bucket_key[1],

                        "decision":
                            decision.decision,

                        "reason":
                            decision.reason,

                        "semantic_similarity":
                            round(
                                decision.semantic_similarity,
                                6,
                            ),

                        "lexical_similarity":
                            round(
                                decision.lexical_similarity,
                                6,
                            ),

                        "requirement_similarity":
                            round(
                                decision.requirement_similarity,
                                6,
                            ),

                        "checkable_similarity":
                            round(
                                decision.checkable_similarity,
                                6,
                            ),

                        "same_act":
                            decision.same_act,

                        "same_category":
                            decision.same_category,

                        "shared_legal_anchor":
                            decision.shared_legal_anchor,

                        "compatibility":
                            decision.compatibility,

                        "compatibility_reasons":
                            decision.compatibility_reasons,
                    }
                )

                if (
                    decision.decision
                    == "AUTO_MERGE"
                ):

                    union_find.union(
                        left["candidate_id"],
                        right["candidate_id"],
                    )

    # --------------------------------------------------------
    # Build merge groups.
    # --------------------------------------------------------

    groups = defaultdict(list)

    for entry in entries:

        root = union_find.find(
            entry["candidate_id"]
        )

        groups[root].append(entry)

    consolidated = []

    merge_audit = []

    for group_index, members in enumerate(
        groups.values(),
        start=1,
    ):

        canonical = choose_canonical(
            members
        )

        member_ids = [
            item["candidate_id"]
            for item in members
        ]

        all_source_ids = sorted(
            {
                source_id
                for item in members
                for source_id in item.get(
                    "source_clause_ids",
                    [],
                )
            }
        )

        all_anchors = sorted(
            {
                anchor
                for item in members
                for anchor in item.get(
                    "source_legal_anchors",
                    [],
                )
            }
        )

        if len(members) == 1:

            consolidation_action = (
                "KEEP"
            )

            reason = (
                "No sufficiently strong duplicate "
                "evidence was found."
            )

        else:

            consolidation_action = (
                "MERGED"
            )

            reason = (
                "Candidates were consolidated because "
                "the automatic merge rules found "
                "substantive duplication within the "
                "same Act/category."
            )

        consolidated_record = {
            "taxonomy_id": (
                f"{canonical['resolved_act'][:8]}"
                f"-{group_index:04d}"
            ),

            "act": canonical.get(
                "resolved_act",
                "UNRESOLVED",
            ),

            "category": canonical.get(
                "category",
                "",
            ),

            "requirement": canonical.get(
                "requirement",
                "",
            ),

            "checkable_test": canonical.get(
                "checkable_test",
                "",
            ),

            "applicability": canonical.get(
                "applicability",
                "",
            ),

            "source": canonical.get(
                "source",
                "",
            ),

            "weight": canonical.get(
                "weight",
                "",
            ),

            "assessability": canonical.get(
                "assessability",
                "",
            ),

            "citation_confidence": canonical.get(
                "citation_confidence",
                "",
            ),

            "source_candidate_ids":
                sorted(member_ids),

            "source_clause_ids":
                all_source_ids,

            "source_legal_anchors":
                all_anchors,

            "source_provenance_status":
                (
                    "RESOLVED"
                    if all_source_ids
                    else "UNRESOLVED"
                ),

            "consolidation_action":
                consolidation_action,

            "consolidation_reason":
                reason,

            "canonical_candidate_id":
                canonical["candidate_id"],

            "review_status":
                "PENDING_HUMAN_REVIEW",

            "candidate_count":
                len(members),
        }

        consolidated.append(
            consolidated_record
        )

        if len(members) > 1:

            merge_audit.append(
                {
                    "merge_group_id":
                        consolidated_record[
                            "taxonomy_id"
                        ],

                    "canonical_candidate_id":
                        canonical[
                            "candidate_id"
                        ],

                    "merged_candidate_ids":
                        sorted(member_ids),

                    "source_clause_ids":
                        all_source_ids,

                    "reason":
                        reason,

                    "review_required":
                        True,
                }
            )

    return (
        consolidated,
        pair_audit,
        merge_audit,
    )


# ============================================================
# HUMAN REVIEW / SCOPE DECISIONS
# ============================================================

def build_scope_reviews(
    entries: list[dict[str, Any]]
):

    omitted_review = []
    scope_review = []
    provenance_review = []

    for entry in entries:

        scope = entry["scope"]

        if (
            scope["scope_decision"]
            == "OMIT_REVIEW"
        ):

            omitted_review.append(
                {
                    "candidate_id":
                        entry["candidate_id"],

                    "act":
                        entry.get(
                            "resolved_act",
                            "",
                        ),

                    "category":
                        entry.get(
                            "category",
                            "",
                        ),

                    "requirement":
                        entry.get(
                            "requirement",
                            "",
                        ),

                    "source_clause_ids":
                        entry.get(
                            "source_clause_ids",
                            [],
                        ),

                    "reasons":
                        scope[
                            "out_of_scope_reasons"
                        ],

                    "decision":
                        "OMIT_REVIEW",

                    "human_review_required":
                        True,
                }
            )

        elif (
            scope["scope_decision"]
            == "SCOPE_REVIEW"
        ):

            scope_review.append(
                {
                    "candidate_id":
                        entry["candidate_id"],

                    "act":
                        entry.get(
                            "resolved_act",
                            "",
                        ),

                    "category":
                        entry.get(
                            "category",
                            "",
                        ),

                    "requirement":
                        entry.get(
                            "requirement",
                            "",
                        ),

                    "source_clause_ids":
                        entry.get(
                            "source_clause_ids",
                            [],
                        ),

                    "government_facing_signals":
                        scope[
                            "government_facing_signals"
                        ],

                    "project_relevance_signals":
                        scope[
                            "project_relevance_signals"
                        ],

                    "decision":
                        "SCOPE_REVIEW",

                    "human_review_required":
                        True,
                }
            )

        if (
            entry.get(
                "source_provenance_status"
            )
            not in {
                "RESOLVED_SINGLE_ACT",
                "RESOLVED_MULTI_ACT",
            }
        ):

            provenance_review.append(
                {
                    "candidate_id":
                        entry["candidate_id"],

                    "source_clause_ids":
                        entry.get(
                            "source_clause_ids",
                            [],
                        ),

                    "missing_source_clause_ids":
                        entry.get(
                            "missing_source_clause_ids",
                            [],
                        ),

                    "source_provenance_status":
                        entry.get(
                            "source_provenance_status",
                            "",
                        ),

                    "requirement":
                        entry.get(
                            "requirement",
                            "",
                        ),
                }
            )

    return (
        omitted_review,
        scope_review,
        provenance_review,
    )


# ============================================================
# TARGET RANGE HANDLING
# ============================================================

def target_status(
    act: str,
    count: int,
) -> str:

    target = ACT_TARGETS.get(act)

    if target is None:
        return "NO_TARGET_DEFINED"

    low, high = target

    if count < low:
        return "BELOW_TARGET"

    if count > high:
        return "ABOVE_TARGET"

    return "WITHIN_TARGET"


def rank_for_target_reduction(
    record: dict[str, Any]
) -> float:

    try:
        weight = float(
            record.get("weight", 0)
        )
    except (TypeError, ValueError):
        weight = 0.0

    assessability = numeric_assessability(
        record.get(
            "assessability",
            "",
        )
    )

    confidence = numeric_confidence(
        record.get(
            "citation_confidence",
            "",
        )
    )

    source_count = len(
        record.get(
            "source_candidate_ids",
            [],
        )
    )

    category = canonical_text(
        record.get(
            "category",
            "",
        )
    )

    category_bonus = 0.5 if category else 0.0

    return (
        weight * 4.0
        + assessability * 2.0
        + confidence * 2.0
        + min(source_count, 5) * 0.5
        + category_bonus
    )


def apply_target_pressure(
    consolidated: list[dict[str, Any]]
):

    by_act = defaultdict(list)

    for record in consolidated:
        by_act[
            record.get(
                "act",
                "UNRESOLVED",
            )
        ].append(record)

    final_selection = []
    target_review = []

    for act, records in by_act.items():

        target = ACT_TARGETS.get(act)

        if target is None:

            for record in records:
                final_selection.append(record)

            continue

        low, high = target

        # ----------------------------------------------------
        # Below target:
        # Never invent requirements.
        # ----------------------------------------------------

        if len(records) <= high:

            for record in records:
                final_selection.append(record)

            continue

        # ----------------------------------------------------
        # Above target:
        #
        # We must reduce the taxonomy for project design,
        # but this is NOT treated as legal deletion.
        #
        # We preserve all displaced records in
        # TARGET_REDUCTION_REVIEW.
        # ----------------------------------------------------

        ranked = sorted(
            records,
            key=rank_for_target_reduction,
            reverse=True,
        )

        selected = []

        # First pass:
        # maximize category coverage.
        categories_seen = set()

        for record in ranked:

            category = canonical_text(
                record.get(
                    "category",
                    "",
                )
            )

            if (
                category
                and category not in categories_seen
                and len(selected) < high
            ):

                selected.append(record)
                categories_seen.add(category)

        # Second pass:
        # fill remaining slots by priority.
        for record in ranked:

            if (
                record in selected
                or len(selected) >= high
            ):
                continue

            selected.append(record)

        selected_ids = {
            record["taxonomy_id"]
            for record in selected
        }

        for record in selected:
            final_selection.append(record)

        for record in records:

            if (
                record["taxonomy_id"]
                not in selected_ids
            ):

                target_review.append(
                    {
                        "taxonomy_id":
                            record["taxonomy_id"],

                        "act":
                            act,

                        "category":
                            record.get(
                                "category",
                                "",
                            ),

                        "requirement":
                            record.get(
                                "requirement",
                                "",
                            ),

                        "source_candidate_ids":
                            record.get(
                                "source_candidate_ids",
                                [],
                            ),

                        "reason":
                            (
                                "Act exceeds the project target "
                                "maximum after consolidation. "
                                "Record retained for human/project "
                                "design review rather than deleted."
                            ),

                        "target_min":
                            low,

                        "target_max":
                            high,

                        "priority_score":
                            round(
                                rank_for_target_reduction(
                                    record
                                ),
                                6,
                            ),

                        "decision":
                            "TARGET_REDUCTION_REVIEW",

                        "human_review_required":
                            True,
                    }
                )

    return (
        final_selection,
        target_review,
    )


# ============================================================
# FINAL RECORD CLEANING
# ============================================================

def clean_internal_fields(
    record: dict[str, Any]
) -> dict[str, Any]:

    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


# ============================================================
# CSV OUTPUT
# ============================================================

def write_csv(
    path: Path,
    records: list[dict[str, Any]],
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not records:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    # Collect all keys.
    fields = []

    for record in records:
        for key in record.keys():
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:

            row = {}

            for field in fields:

                value = record.get(
                    field,
                    "",
                )

                if isinstance(
                    value,
                    (list, dict),
                ):

                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                    )

                row[field] = value

            writer.writerow(row)


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    raw_entries,
    prepared,
    consolidated,
    final_taxonomy,
    pair_audit,
    merge_audit,
    omitted_review,
    scope_review,
    target_review,
    provenance_review,
):

    act_before = Counter(
        entry.get(
            "resolved_act",
            "UNRESOLVED",
        )
        for entry in prepared
    )

    act_after = Counter(
        record.get(
            "act",
            "UNRESOLVED",
        )
        for record in final_taxonomy
    )

    merge_count = sum(
        1
        for record in consolidated
        if record.get(
            "consolidation_action"
        ) == "MERGED"
    )

    pair_counts = Counter(
        item["decision"]
        for item in pair_audit
    )

    target_summary = {}

    for act, (low, high) in ACT_TARGETS.items():

        count = act_after.get(
            act,
            0,
        )

        target_summary[act] = {
            "count": count,
            "target_min": low,
            "target_max": high,
            "status": target_status(
                act,
                count,
            ),
        }

    return {
        "run_timestamp_utc":
            utc_now_iso(),

        "input_candidate_count":
            len(raw_entries),

        "prepared_candidate_count":
            len(prepared),

        "consolidated_requirement_count":
            len(consolidated),

        "final_provisional_taxonomy_count":
            len(final_taxonomy),

        "automatic_merge_group_count":
            merge_count,

        "pair_comparison_count":
            len(pair_audit),

        "pair_decision_counts":
            dict(pair_counts),

        "scope_omit_review_count":
            len(omitted_review),

        "scope_review_count":
            len(scope_review),

        "target_reduction_review_count":
            len(target_review),

        "provenance_review_count":
            len(provenance_review),

        "act_counts_before_consolidation":
            dict(act_before),

        "act_counts_after_final_selection":
            dict(act_after),

        "target_summary":
            target_summary,

        "human_review_required":
            True,

        "legal_approval":
            False,

        "random_filtering":
            False,

        "target_ranges_used_as_legal_truth":
            False,

        "design_note":
            (
                "Target ranges are project-design constraints. "
                "They do not establish legal validity and "
                "displaced candidates remain available for review."
            ),
    }


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Consolidate the draft social-media governance "
            "compliance taxonomy using provenance, semantic "
            "similarity, legal anchors, and project-specific scope."
        )
    )

    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
    )

    parser.add_argument(
        "--clauses",
        type=Path,
        default=DEFAULT_CLAUSES,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help=(
            "Disable sentence-transformers and use TF-IDF fallback."
        ),
    )

    parser.add_argument(
        "--fail-on-provenance",
        action="store_true",
        help=(
            "Exit with status 2 if unresolved/partial provenance exists."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    global USE_SENTENCE_TRANSFORMER

    args = parse_args()

    if args.no_embeddings:
        USE_SENTENCE_TRANSFORMER = False

    print("=" * 80)
    print("FINAL TAXONOMY CONSOLIDATION ENGINE v2")
    print("=" * 80)

    print(
        f"\nTaxonomy input:\n  {args.taxonomy}"
    )

    print(
        f"\nGovernment clauses input:\n  {args.clauses}"
    )

    print(
        f"\nOutput directory:\n  {args.output_dir}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    raw_entries = load_taxonomy(
        args.taxonomy
    )

    clauses_df = load_clauses(
        args.clauses
    )

    clause_lookup = (
        clauses_df
        .set_index(
            "clause_id",
            drop=False,
        )
        .to_dict(
            orient="index"
        )
    )

    print(
        f"\nLoaded {len(raw_entries)} "
        "taxonomy candidates."
    )

    print(
        f"Loaded {len(clauses_df)} "
        "cleaned government clauses."
    )

    if not raw_entries:
        raise ValueError(
            "No taxonomy candidates were loaded."
        )

    # --------------------------------------------------------
    # Prepare candidates
    # --------------------------------------------------------

    print(
        "\n[1/6] Resolving provenance and project scope..."
    )

    prepared = prepare_candidates(
        raw_entries,
        clause_lookup,
    )

    # --------------------------------------------------------
    # Similarity engine
    # --------------------------------------------------------

    print(
        "\n[2/6] Building semantic similarity engine..."
    )

    similarity_engine = SimilarityEngine(
        prepared
    )

    # --------------------------------------------------------
    # Consolidation
    # --------------------------------------------------------

    print(
        "\n[3/6] Consolidating substantive duplicates..."
    )

    (
        consolidated,
        pair_audit,
        merge_audit,
    ) = consolidate_candidates(
        prepared,
        similarity_engine,
    )

    # --------------------------------------------------------
    # Scope reviews
    # --------------------------------------------------------

    print(
        "\n[4/6] Building scope/provenance review queues..."
    )

    (
        omitted_review,
        scope_review,
        provenance_review,
    ) = build_scope_reviews(
        prepared
    )

    # --------------------------------------------------------
    # Target handling
    # --------------------------------------------------------

    print(
        "\n[5/6] Applying project target ranges..."
    )

    (
        final_taxonomy,
        target_review,
    ) = apply_target_pressure(
        consolidated
    )

    # Clean internal fields.
    consolidated = [
        clean_internal_fields(record)
        for record in consolidated
    ]

    final_taxonomy = [
        clean_internal_fields(record)
        for record in final_taxonomy
    ]

    # --------------------------------------------------------
    # Add final review metadata
    # --------------------------------------------------------

    for record in final_taxonomy:

        record["review_status"] = (
            "PENDING_HUMAN_REVIEW"
        )

        record["legal_approval"] = False

        record["taxonomy_stage"] = (
            "PROVISIONAL_FINAL"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = make_summary(
        raw_entries=raw_entries,
        prepared=prepared,
        consolidated=consolidated,
        final_taxonomy=final_taxonomy,
        pair_audit=pair_audit,
        merge_audit=merge_audit,
        omitted_review=omitted_review,
        scope_review=scope_review,
        target_review=target_review,
        provenance_review=provenance_review,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Main outputs
    # --------------------------------------------------------

    write_json(
        output_dir
        / "FINAL_TAXONOMY_PROVISIONAL.json",
        final_taxonomy,
    )

    write_csv(
        output_dir
        / "FINAL_TAXONOMY_PROVISIONAL.csv",
        final_taxonomy,
    )

    write_json(
        output_dir
        / "ALL_CONSOLIDATED_REQUIREMENTS.json",
        consolidated,
    )

    write_json(
        output_dir
        / "MERGE_AUDIT.json",
        merge_audit,
    )

    write_json(
        output_dir
        / "DECISION_LOG.json",
        pair_audit,
    )

    write_json(
        output_dir
        / "OMITTED_CANDIDATES_REVIEW.json",
        omitted_review,
    )

    write_json(
        output_dir
        / "SCOPE_REVIEW_REQUIRED.json",
        scope_review,
    )

    write_json(
        output_dir
        / "TARGET_REDUCTION_REVIEW.json",
        target_review,
    )

    write_json(
        output_dir
        / "PROVENANCE_REVIEW.json",
        provenance_review,
    )

    write_json(
        output_dir
        / "CONSOLIDATION_SUMMARY.json",
        summary,
    )

    # --------------------------------------------------------
    # Per-Act outputs
    # --------------------------------------------------------

    per_act_dir = (
        output_dir
        / "per_act"
    )

    per_act_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_by_act = defaultdict(list)

    for record in final_taxonomy:

        final_by_act[
            record.get(
                "act",
                "UNRESOLVED",
            )
        ].append(record)

    for act, records in final_by_act.items():

        path = (
            per_act_dir
            / (
                safe_filename(act)
                + "_FINAL_PROVISIONAL.json"
            )
        )

        write_json(
            path,
            records,
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CONSOLIDATION RESULT")
    print("=" * 80)

    print(
        f"\nInput candidates:              "
        f"{len(raw_entries)}"
    )

    print(
        f"After consolidation:           "
        f"{len(consolidated)}"
    )

    print(
        f"Provisional final taxonomy:    "
        f"{len(final_taxonomy)}"
    )

    print(
        f"Automatic merge groups:        "
        f"{len(merge_audit)}"
    )

    print(
        f"Scope omit reviews:             "
        f"{len(omitted_review)}"
    )

    print(
        f"Scope reviews:                  "
        f"{len(scope_review)}"
    )

    print(
        f"Target reduction reviews:      "
        f"{len(target_review)}"
    )

    print(
        f"Provenance reviews:             "
        f"{len(provenance_review)}"
    )

    print("\n")
    print(
        f"{'ACT':30s}"
        f"{'FINAL':>8s}"
        f"{'TARGET':>12s}"
        f"{'STATUS':>22s}"
    )

    print("-" * 76)

    for act, (
        low,
        high,
    ) in ACT_TARGETS.items():

        count = sum(
            1
            for record in final_taxonomy
            if record.get(
                "act"
            ) == act
        )

        print(
            f"{act:30s}"
            f"{count:8d}"
            f"{f'{low}-{high}':>12s}"
            f"{target_status(act, count):>22s}"
        )

    print("\n")
    print(
        "Output written to:"
    )

    print(
        f"  {output_dir}"
    )

    print("\nKey files:")

    print(
        "  FINAL_TAXONOMY_PROVISIONAL.json"
    )

    print(
        "  FINAL_TAXONOMY_PROVISIONAL.csv"
    )

    print(
        "  ALL_CONSOLIDATED_REQUIREMENTS.json"
    )

    print(
        "  MERGE_AUDIT.json"
    )

    print(
        "  DECISION_LOG.json"
    )

    print(
        "  OMITTED_CANDIDATES_REVIEW.json"
    )

    print(
        "  SCOPE_REVIEW_REQUIRED.json"
    )

    print(
        "  TARGET_REDUCTION_REVIEW.json"
    )

    print(
        "  PROVENANCE_REVIEW.json"
    )

    print(
        "  CONSOLIDATION_SUMMARY.json"
    )

    print("\n")
    print("=" * 80)
    print(
        "IMPORTANT: The resulting taxonomy is "
        "PROVISIONAL and requires human review."
    )
    print(
        "No requirement was legally approved by this script."
    )
    print(
        "No candidate was silently deleted."
    )
    print(
        "Target reduction records remain available "
        "for review."
    )
    print("=" * 80)

    if (
        args.fail_on_provenance
        and provenance_review
    ):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )