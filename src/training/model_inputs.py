"""
src/training/model_inputs.py
Label-free model inputs: the single place that decides what text a model sees.

The classifier may only see what is available for an unseen clause at inference
time: the clause text and (optionally) the platform name. Every column produced
by the LLM labelling step (verdict, chosen taxonomy rule, act, category, score,
justification) is a label or derived from the label and must never reach a
feature-building function.

History: the previous builder (build_context_texts) inserted the LLM-chosen
rule's checkable_test, or "General Policy" when the LLM chose NONE. Because
best_match_taxonomy_id == "NONE" <=> verdict == "Not Addressed" for every row of
the corpus, that leaked the label into every model input. It has been removed.
"""

import numpy as np
import pandas as pd

# Columns written by the LLM labeller. Any of these in a feature-building input is leakage.
LLM_OUTPUT_COLUMNS = (
    "best_match_taxonomy_id",
    "matched_act",
    "dpdp_category",
    "verdict",
    "verdict_score",
    "justification",
)

# The only columns a feature-building function may receive.
ALLOWED_INPUT_COLUMNS = ("clause_text", "app_name")


class LabelLeakageError(ValueError):
    """Raised when a feature-building function receives anything but clause_text/app_name."""


def model_inputs(df):
    """Return a copy of df restricted to the allowed input columns (clause_text, app_name)."""
    inputs = pd.DataFrame({"clause_text": df["clause_text"].astype(str).values})
    if "app_name" in df.columns:
        inputs["app_name"] = df["app_name"].astype(str).values
    else:
        inputs["app_name"] = "Platform"
    return inputs


def assert_label_free(inputs):
    """Fail if inputs carries any column other than clause_text/app_name (LLM columns first)."""
    columns = list(getattr(inputs, "columns", []))
    leaked = [c for c in columns if c in LLM_OUTPUT_COLUMNS]
    if leaked:
        raise LabelLeakageError(
            f"LLM output columns passed to a feature-building function: {leaked}. "
            f"Pass model_inputs(df) instead."
        )
    unexpected = [c for c in columns if c not in ALLOWED_INPUT_COLUMNS]
    if unexpected:
        raise LabelLeakageError(
            f"Unexpected columns passed to a feature-building function: {unexpected}. "
            f"Only {list(ALLOWED_INPUT_COLUMNS)} are allowed."
        )
    if "clause_text" not in columns:
        raise LabelLeakageError("Feature-building input must contain clause_text")


def build_input_texts(inputs, include_app_name=False):
    """
    Build label-free texts for encoding: clause_text, or "[app_name] clause_text".

    The encoder prefix (e.g. "query: " for E5) is added by encode_texts, so the
    text a model sees is encoder_prefix(model) + build_input_texts(...)[i].
    """
    assert_label_free(inputs)
    clauses = [str(t).strip() for t in inputs["clause_text"].values]
    if not include_app_name:
        return clauses
    apps = [str(a).strip() for a in inputs["app_name"].values]
    return [f"[{app}] {clause}" for app, clause in zip(apps, clauses)]


def encoder_prefix(model_name):
    """E5 models expect a "query: " prefix; other encoders take raw text."""
    return "query: " if "e5-" in model_name else ""


def encode_texts(embedder, texts, model_name, show_progress_bar=False):
    """Encode texts with the encoder's prefix and L2-normalised embeddings."""
    prefix = encoder_prefix(model_name)
    return embedder.encode(
        [prefix + t for t in texts],
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
