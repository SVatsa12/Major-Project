"""
src/training/feature_extraction.py
Actionability Feature Extraction Module - Phase B: Feature Union.

Implements the union of:
- 8 handcrafted actionability features (baseline)
- 7 count-based features from earlier evaluator

Total: 15 features (or fewer if duplicates are detected and removed).
"""

import re
import numpy as np

# === Current 8 Actionability Features Lexicons ===

# Feature 0: hedge_density
HEDGE_WORDS = {
    "may", "might", "could", "endeavor", "attempt", "where feasible",
    "reasonable", "appropriate", "generally", "typically", "from time to time",
}

# Feature 1: action_verb_count
ACTION_VERBS = {
    "delete", "erase", "notify", "provide", "restrict", "rectify",
    "withdraw", "inform", "escalate", "terminate", "purge",
}

# Feature 5: commitment_ratio
COMMITMENT_VERBS = {"will", "shall", "must", "guarantee", "ensure"}

# === Phase B: Additional Count-Based Features Lexicons ===

# Feature 8: hedge_phrase_count (overlaps with hedge but uses full phrases)
HEDGE_PHRASES = {
    "may", "where feasible", "endeavor", "reasonable efforts", "strive",
    "when appropriate", "from time to time", "to the extent", "as applicable",
    "could", "might"
}

# Feature 9: action_phrase_count
ACTION_PHRASES = {
    "settings", "click", "delete your account", "will delete", "end-to-end encrypt",
    "transparency report", "do not share", "never sell", "must obtain", "you can request",
    "you can access", "download your data", "port your data", "opt-out", "unsubscribe",
    "disable", "turn off", "manage", "control"
}

# Feature 10: requirement_count
REQUIREMENT_PHRASES = {
    "requirement", "must", "shall", "required", "needs to", "has to",
    "is mandatory", "subject to"
}

# Feature 11: commitment_count
COMMITMENT_PHRASES = {
    "commit", "committed", "commitment", "committed to", "dedicated",
    "ensure", "guarantee", "will"
}

# Feature 12: mechanism_count
MECHANISM_PHRASES = {
    "mechanism", "process", "procedure", "steps", "how to", "instructions",
    "click", "link", "button", "navigate", "access", "go to"
}


def _count_phrase_occurrences(text_lower, phrases):
    """Count occurrences of phrases in text (case-insensitive substring)."""
    count = 0
    for phrase in phrases:
        count += text_lower.count(phrase)
    return count


def extract_actionability_features(texts):
    """
    Extract 15 actionability features from clause texts (Phase B: Feature Union).
    
    Returns 15-dim feature array:
    0. hedge_density
    1. action_verb_count
    2. has_ui_path
    3. has_contact_mechanism
    4. has_timeline_sla
    5. commitment_ratio
    6. clause_length_norm
    7. parental_verification_marker
    8. hedge_phrase_count
    9. action_phrase_count
    10. requirement_count
    11. commitment_count
    12. mechanism_count
    13. actionability_ratio
    14. has_nav
    """
    features = []
    
    for text in texts:
        text_str = str(text).strip()
        text_lower = text_str.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # ===== Original 8 Features =====
        
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
        
        # Feature 5: commitment_ratio
        commitment_count = 0
        for verb in COMMITMENT_VERBS:
            commitment_count += text_lower.count(verb)
        commitment_ratio = commitment_count / (commitment_count + hedge_count + 1e-8)
        
        # Feature 6: clause_length_norm
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
        
        # ===== Phase B: 7 Additional Count-Based Features =====
        
        # Feature 8: hedge_phrase_count
        hedge_phrase_count = _count_phrase_occurrences(text_lower, HEDGE_PHRASES)
        
        # Feature 9: action_phrase_count
        action_phrase_count = _count_phrase_occurrences(text_lower, ACTION_PHRASES)
        
        # Feature 10: requirement_count
        requirement_count = _count_phrase_occurrences(text_lower, REQUIREMENT_PHRASES)
        
        # Feature 11: commitment_count (from phrases)
        commitment_phrase_count = _count_phrase_occurrences(text_lower, COMMITMENT_PHRASES)
        
        # Feature 12: mechanism_count
        mechanism_count = _count_phrase_occurrences(text_lower, MECHANISM_PHRASES)
        
        # Feature 13: actionability_ratio
        total_action_signals = action_phrase_count + mechanism_count + commitment_phrase_count
        actionability_ratio = total_action_signals / max(1, hedge_phrase_count + total_action_signals)
        
        # Feature 14: has_nav (binary)
        has_nav = 1 if any(c in text_str for c in [">", "@", "http"]) or \
                      any(c in text_str.lower() for c in ["->", "click"]) else 0
        
        features.append([
            hedge_density,
            action_verb_count,
            has_ui_path,
            has_contact_mechanism,
            has_timeline_sla,
            commitment_ratio,
            clause_length_norm,
            parental_verification_marker,
            hedge_phrase_count,
            action_phrase_count,
            requirement_count,
            commitment_phrase_count,
            mechanism_count,
            actionability_ratio,
            has_nav,
        ])
    
    return np.array(features, dtype=np.float32)


# ===== Duplicate Detection Analysis =====
"""
DUPLICATE ANALYSIS (Feature Union):

Comparing original 8 features with added 7:

1. hedge_density (Feature 0) vs hedge_phrase_count (Feature 8)
   - hedge_density: (count of hedge words) / word_count
   - hedge_phrase_count: raw count of hedge phrases
   - NOT DUPLICATES: Different computation (density vs count). KEEP BOTH.

2. action_verb_count (Feature 1) vs action_phrase_count (Feature 9)
   - action_verb_count: count of specific action verbs
   - action_phrase_count: count of action phrases (superset)
   - NOT DUPLICATES: Different lexicons. KEEP BOTH.

3. commitment_ratio (Feature 5) vs commitment_phrase_count (Feature 11)
   - commitment_ratio: ratio C/(C+H+eps)
   - commitment_phrase_count: raw count
   - NOT DUPLICATES: Different computation. KEEP BOTH.

4. has_nav (Feature 14) - NEW feature from archive
   - Similar to has_ui_path (Feature 2) but different regex
   - has_ui_path: looks for UI navigation patterns (settings>..., account settings, etc.)
   - has_nav: looks for raw text markers (>, ->, http, @, click)
   - NOT DUPLICATES: Different definitions. KEEP BOTH.

CONCLUSION: NO DUPLICATES DETECTED. All 15 features are distinct and kept.
"""
