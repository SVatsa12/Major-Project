"""
tests/test_label_leakage.py
Guards against label leakage into model inputs.

Every feature-building function must receive only clause_text and app_name.
Passing any LLM output column (verdict, chosen rule, act, category, score,
justification) must fail before any feature is computed.

Run: python -m unittest discover -s tests -v
"""

import re
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.model_inputs import (
    LLM_OUTPUT_COLUMNS,
    LabelLeakageError,
    assert_label_free,
    build_input_texts,
    model_inputs,
)
from src.evaluation.feature_variants import (
    FEATURE_EXTRACTORS,
    extract_baseline_features,
    extract_clause_rule_pairing_b2a,
    extract_encoder_swap_features,
)
from src.evaluation.incremental_features import extract_all_rules_similarity, extract_rubric_features
from src.training.train_final_model import build_feature_matrix

TAXONOMY = [{"taxonomy_id": "R1", "requirement": "Notice must be given.", "category": "Notice",
             "checkable_test": "Does the policy give notice?"}]


def labelled_frame():
    """A frame shaped like corpus/labeled_clauses_bootstrap.csv."""
    return pd.DataFrame({
        "clause_id": ["c1", "c2"],
        "app_name": ["AppA", "AppB"],
        "clause_text": ["We may share your data.", "You can delete your account in Settings within 30 days."],
        "best_match_taxonomy_id": ["NONE", "R1"],
        "dpdp_category": ["", "Notice"],
        "matched_act": ["", "DPDP_ACT_2023"],
        "verdict": ["Not Addressed", "Compliant"],
        "verdict_score": [0, 2],
        "justification": ["n/a", "explicit"],
    })


# Every feature-building entry point, called the way the experiment scripts call it.
FEATURE_BUILDERS = {
    "build_input_texts": lambda df: build_input_texts(df),
    "extract_baseline_features": lambda df: extract_baseline_features(df, TAXONOMY),
    "extract_encoder_swap_features": lambda df: extract_encoder_swap_features(df, TAXONOMY, "intfloat/e5-base-v2"),
    "extract_clause_rule_pairing_b2a": lambda df: extract_clause_rule_pairing_b2a(df, TAXONOMY),
    "extract_all_rules_similarity": lambda df: extract_all_rules_similarity(df, TAXONOMY),
    "extract_rubric_features": lambda df: extract_rubric_features(df),
    "train_final_model.build_feature_matrix": lambda df: build_feature_matrix(df, TAXONOMY),
    **{f"FEATURE_EXTRACTORS[{name}]": (lambda fn: lambda df: fn(df, TAXONOMY))(fn)
       for name, fn in FEATURE_EXTRACTORS.items()},
}


class TestLabelFreeInputs(unittest.TestCase):

    def test_each_llm_column_is_rejected(self):
        for col in LLM_OUTPUT_COLUMNS:
            with self.subTest(column=col):
                df = model_inputs(labelled_frame())
                df[col] = labelled_frame()[col].values
                with self.assertRaises(LabelLeakageError):
                    assert_label_free(df)

    def test_every_feature_builder_rejects_llm_columns(self):
        for name, builder in FEATURE_BUILDERS.items():
            for col in LLM_OUTPUT_COLUMNS:
                with self.subTest(builder=name, column=col):
                    df = model_inputs(labelled_frame())
                    df[col] = labelled_frame()[col].values
                    with self.assertRaises(LabelLeakageError):
                        builder(df)

    def test_every_feature_builder_rejects_raw_corpus_frame(self):
        for name, builder in FEATURE_BUILDERS.items():
            with self.subTest(builder=name):
                with self.assertRaises(LabelLeakageError):
                    builder(labelled_frame())

    def test_model_inputs_keeps_only_clause_text_and_app_name(self):
        self.assertEqual(list(model_inputs(labelled_frame()).columns), ["clause_text", "app_name"])

    def test_input_text_is_the_clause_and_ignores_labels(self):
        df = labelled_frame()
        flipped = df.copy()
        flipped["verdict"] = flipped["verdict"].iloc[::-1].values
        flipped["best_match_taxonomy_id"] = flipped["best_match_taxonomy_id"].iloc[::-1].values

        texts = build_input_texts(model_inputs(df))
        self.assertEqual(texts, build_input_texts(model_inputs(flipped)))
        self.assertEqual(texts, list(df["clause_text"]))
        self.assertEqual(build_input_texts(model_inputs(df), include_app_name=True)[0],
                         "[AppA] We may share your data.")
        for text in texts:
            self.assertNotIn("Mandate", text)
            self.assertNotIn("General Policy", text)


class TestNoLeakyCodePaths(unittest.TestCase):

    def python_sources(self):
        for folder in ("src", "scripts"):
            yield from (PROJECT_ROOT / folder).rglob("*.py")

    def test_context_text_builder_is_gone(self):
        pattern = re.compile(r"build_context_texts\s*\(|import[^\n]*build_context_texts")
        offenders = [str(p) for p in self.python_sources() if pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])

    def test_no_noop_assertions(self):
        pattern = re.compile(r"assert[^\n]*\bor True\b")
        offenders = [str(p) for p in self.python_sources() if pattern.search(p.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
