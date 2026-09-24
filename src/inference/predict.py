"""
src/inference/predict.py

Standalone production inference module for the InLegalBERT compliance classifier.
Uses calibrated probability thresholds loaded from thresholds.json.

Decision rule:
    Compliant            if P(Compliant) >= tau_C
    Partially Compliant  if P(PC) >= tau_PC  AND  P(Compliant) < tau_C
    Not Addressed        otherwise

Input format (must match training):
    clause_text + " [SEP] Category: " + dpdp_category + " | Rule: " + best_match_taxonomy_id
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("models/inlegalbert_classifier")
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"

DEFAULT_THRESHOLDS = {
    "compliant": 0.45,
    "partially_compliant": 0.20
}

# Label ordering MUST match training: index 0=Not Addressed, 1=Partially Compliant, 2=Compliant
LABEL_NAMES = ["Not Addressed", "Partially Compliant", "Compliant"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}


class CompliancePredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, float]:
        if THRESHOLDS_FILE.exists():
            try:
                with open(THRESHOLDS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_THRESHOLDS

    def _format_input(self, clause_text: str, category: str = "General", rule_id: str = "None") -> str:
        cat_str = str(category).strip() if category else "General"
        rule_str = str(rule_id).strip() if rule_id else "None"
        # Normalise NONE sentinel from labeler to "None"
        if rule_str.upper() == "NONE" or rule_str == "":
            rule_str = "None"
        return f"{clause_text.strip()} [SEP] Category: {cat_str} | Rule: {rule_str}"

    def _apply_thresholds(self, probs: np.ndarray) -> str:
        """Apply calibrated decision rule to a probability vector (length 3)."""
        p_c  = probs[2]   # index 2 = Compliant
        p_pc = probs[1]   # index 1 = Partially Compliant

        t_c  = self.thresholds.get("compliant",          DEFAULT_THRESHOLDS["compliant"])
        t_pc = self.thresholds.get("partially_compliant", DEFAULT_THRESHOLDS["partially_compliant"])

        if p_c >= t_c:
            return "Compliant"
        elif p_pc >= t_pc:
            return "Partially Compliant"
        else:
            return "Not Addressed"

    def predict_single(
        self,
        clause_text: str,
        category: str = "General",
        rule_id: str = "None",
    ) -> Dict:
        """
        Predict compliance verdict for a single clause.

        Args:
            clause_text: Raw policy clause text.
            category:    DPDP category from upstream rule retrieval (e.g. "Consent & Notice").
            rule_id:     Matched taxonomy rule ID (e.g. "DPDP_ACT-0026") or "None".

        Returns:
            dict with keys: verdict, probabilities, calibrated_thresholds
        """
        formatted_text = self._format_input(clause_text, category, rule_id)
        encoded = self.tokenizer(
            formatted_text,
            max_length=512,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        predicted_verdict = self._apply_thresholds(probs)

        return {
            "verdict": predicted_verdict,
            "probabilities": {
                "Not Addressed":      float(probs[0]),
                "Partially Compliant": float(probs[1]),
                "Compliant":           float(probs[2]),
            },
            "calibrated_thresholds": self.thresholds,
        }

    def predict_batch(
        self,
        clauses: List[Dict[str, str]],
        batch_size: int = 16,
    ) -> List[Dict]:
        """
        Predict compliance verdicts for a list of clause dicts.

        Each dict should contain:
            - clause_text           (required)
            - dpdp_category         (optional, default "General")
            - best_match_taxonomy_id (optional, default "None")

        Returns:
            List of dicts with: clause_text, verdict, probabilities
        """
        formatted_inputs = [
            self._format_input(
                c.get("clause_text", ""),
                c.get("dpdp_category", "General"),
                c.get("best_match_taxonomy_id", "None"),
            )
            for c in clauses
        ]

        results = []
        for i in range(0, len(formatted_inputs), batch_size):
            batch_texts = formatted_inputs[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for j, p in enumerate(probs):
                results.append({
                    "clause_text": clauses[i + j].get("clause_text", ""),
                    "verdict": self._apply_thresholds(p),
                    "probabilities": {
                        "Not Addressed":      float(p[0]),
                        "Partially Compliant": float(p[1]),
                        "Compliant":           float(p[2]),
                    },
                })

        return results


if __name__ == "__main__":
    predictor = CompliancePredictor()
    test_clause = "Users may withdraw consent at any time via privacy settings."
    res = predictor.predict_single(
        test_clause,
        category="Consent Management",
        rule_id="SEC_06_1",
    )
    print("Inference Test Output:")
    print(json.dumps(res, indent=2))
