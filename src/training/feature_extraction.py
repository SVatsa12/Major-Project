"""
src/training/feature_extraction.py
Actionability Feature Extraction Module.

Implements the 8 actionability features per the research paper specification:
0. hedge_density: (count of hedge words) / (clause word count)
1. action_verb_count: raw count of specific action verbs
2. has_ui_path: binary, regex for settings navigation
3. has_contact_mechanism: binary, regex for contact info
4. has_timeline_sla: binary, regex for time commitments
5. commitment_ratio: C / (C + H + epsilon), commitment vs hedge words
6. clause_length_norm: min(1.0, word_count / 100.0)
7. parental_verification_marker: binary, regex for child/parental consent
"""

import re
import numpy as np

# Hedge words per specification
HEDGE_WORDS = {
    "may",
    "might",
    "could",
    "endeavor",
    "attempt",
    "where feasible",
    "reasonable",
    "appropriate",
    "generally",
    "typically",
    "from time to time",
}

# Action verbs per specification
ACTION_VERBS = {
    "delete",
    "erase",
    "notify",
    "provide",
    "restrict",
    "rectify",
    "withdraw",
    "inform",
    "escalate",
    "terminate",
    "purge",
}

# Commitment verbs (for commitment_ratio)
COMMITMENT_VERBS = {"will", "shall", "must", "guarantee", "ensure"}


def extract_actionability_features(texts):
    """
    Extract 8 actionability features from clause texts.
    
    Args:
        texts: list or array of clause texts
    
    Returns:
        np.ndarray of shape (len(texts), 8) with features in specified order
    """
    features = []
    
    for text in texts:
        text_str = str(text).strip()
        text_lower = text_str.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # Feature 0: hedge_density
        hedge_count = 0
        for hedge in HEDGE_WORDS:
            hedge_count += text_lower.count(hedge)
        hedge_density = hedge_count / max(1, word_count)
        
        # Feature 1: action_verb_count
        action_verb_count = 0
        for verb in ACTION_VERBS:
            action_verb_count += text_lower.count(verb)
        
        # Feature 2: has_ui_path (binary)
        ui_path_pattern = (
            r"(\bsettings\s*(?:>|->|to)\s*\w+)|"
            r"(\baccount\s*settings\b)|"
            r"(\bprivacy\s*(?:dashboard|settings|center|tab)\b)|"
            r"(\bmanage\s*(?:your)?\s*(?:data|account|cookies)\b)"
        )
        has_ui_path = 1 if re.search(ui_path_pattern, text_str, re.IGNORECASE) else 0
        
        # Feature 3: has_contact_mechanism (binary)
        contact_pattern = (
            r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|"
            r"(https?://[^\s]+)|"
            r"(\bgrievance\s*officer\b)|"
            r"(\bdata\s*protection\s*officer\b)|"
            r"(\bdpo\b)|"
            r"(\bcontact\s*(?:us|our\s*team)\s*at\b)"
        )
        has_contact_mechanism = 1 if re.search(contact_pattern, text_str, re.IGNORECASE) else 0
        
        # Feature 4: has_timeline_sla (binary)
        timeline_pattern = (
            r"(\d+\s*(days?|hours?|hrs?|working\s*days?|months?))|"
            r"(promptly)|"
            r"(\b24\s*hours?\b)|"
            r"(\b72\s*hours?\b)"
        )
        has_timeline_sla = 1 if re.search(timeline_pattern, text_str, re.IGNORECASE) else 0
        
        # Feature 5: commitment_ratio = C / (C + H + epsilon)
        commitment_count = 0
        for verb in COMMITMENT_VERBS:
            commitment_count += text_lower.count(verb)
        commitment_ratio = commitment_count / (commitment_count + hedge_count + 1e-8)
        
        # Feature 6: clause_length_norm = min(1.0, word_count / 100.0)
        clause_length_norm = min(1.0, word_count / 100.0)
        
        # Feature 7: parental_verification_marker (binary)
        parental_pattern = (
            r"(\bparental\s*(?:consent|permission|authorization|supervision)\b)|"
            r"(\bverifiable\s*(?:parental\s*)?consent\b)|"
            r"(\bguardian\b)|"
            r"(\bchild(?:ren)?\s*(?:under|aged?)\s*(?:18|16|13)\b)|"
            r"(\bminor\s*(?:user)?\b)"
        )
        parental_verification_marker = 1 if re.search(parental_pattern, text_str, re.IGNORECASE) else 0
        
        features.append([
            hedge_density,
            action_verb_count,
            has_ui_path,
            has_contact_mechanism,
            has_timeline_sla,
            commitment_ratio,
            clause_length_norm,
            parental_verification_marker,
        ])
    
    return np.array(features, dtype=np.float32)
