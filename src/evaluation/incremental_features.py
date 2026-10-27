"""
src/evaluation/incremental_features.py
Incremental feature additions to improve on fixed_A_e5 baseline.
"""

import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.evaluation.feature_variants import load_or_compute_embeddings
from src.training.model_inputs import assert_label_free, build_input_texts


def get_active_rules(taxonomy):
    """Taxonomy rules used for all-rules similarity (NONE excluded), in taxonomy order."""
    return [item for item in taxonomy if item["taxonomy_id"] != "NONE"]


def all_rules_similarity_from_embeddings(clause_embeddings, rule_embeddings, rule_categories):
    """
    All-rules similarity features from normalised embeddings:
    [cosine(clause, rule_i) for every rule] + [max cosine per category].
    Categories are taken in sorted order so training and inference agree.
    """
    similarities_all = clause_embeddings @ rule_embeddings.T
    
    unique_categories = sorted(set(rule_categories))
    per_category_max = np.zeros((similarities_all.shape[0], len(unique_categories)))
    for cat_idx, cat in enumerate(unique_categories):
        rule_indices = [i for i, c in enumerate(rule_categories) if c == cat]
        per_category_max[:, cat_idx] = similarities_all[:, rule_indices].max(axis=1)
    
    return np.hstack([similarities_all, per_category_max])


def extract_all_rules_similarity(df, taxonomy, encoder_name="intfloat/e5-base-v2", rule_text_mode="full"):
    """
    Feature Set 1: All-rules similarity
    
    Adds 38-dim vector of cosine similarities (clause vs every rule) +
    per-category max similarity features.
    
    df must be model_inputs(...) (clause_text, app_name only); no LLM output is used.
    
    Args:
        rule_text_mode: "full" (default) or "desc_only"
            - "full": Uses full statutory requirement text
            - "desc_only": Uses only category + requirement (same as full for current taxonomy)
    """
    assert_label_free(df)
    
    # Get clause embeddings
    input_texts = build_input_texts(df)
    clause_embeddings = load_or_compute_embeddings(input_texts, encoder_name)
    
    # Get ALL rule embeddings (exclude NONE)
    all_rules = get_active_rules(taxonomy)
    
    # Both modes use only the statutory requirement text for the current taxonomy
    rule_texts = [item["requirement"] for item in all_rules]
    print(f"[ALL-RULES-SIM] Using {rule_text_mode} mode: statutory requirement")
    rule_categories = [item.get("category", "Unknown") for item in all_rules]
    
    print(f"[ALL-RULES-SIM] Computing embeddings for {len(rule_texts)} rules...")
    cache_suffix = "_rules" if rule_text_mode == "full" else "_rules_desc_only"
    rule_embeddings = load_or_compute_embeddings(rule_texts, encoder_name, cache_suffix=cache_suffix)
    
    all_rules_features = all_rules_similarity_from_embeddings(clause_embeddings, rule_embeddings, rule_categories)
    n_rules = len(rule_texts)
    print(f"[ALL-RULES-SIM] Features: {n_rules} rule similarities + {all_rules_features.shape[1] - n_rules} category max = {all_rules_features.shape[1]} dims")
    
    return all_rules_features


def extract_rubric_features(df, labels=None):
    """
    Feature Set 2: Rubric-based features for Compliant vs Partial boundary
    
    Rubric:
    - Compliant = actionable, timeline-bound, specific
    - Partial = vague, non-enforceable, aspirational
    
    Features:
    1. Time expressions (specific vs vague)
    2. Modal strength (strong vs weak)
    3. Hedge words
    4. User-action mechanisms
    5. Numeric quantities and named procedures
    
    df must be model_inputs(...) (clause_text, app_name only); no LLM output is used.
    labels (optional) are only used to print per-class feature means.
    """
    assert_label_free(df)
    
    clauses = df["clause_text"].values
    features = []
    
    # Time expression patterns
    specific_time_pattern = r'\b(\d+\s*(day|hour|week|month|year|business day)s?|within\s+\d+|before\s+\d+|after\s+\d+)\b'
    vague_time_pattern = r'\b(promptly|timely|as soon as possible|reasonable time|without undue delay|expeditiously|immediately)\b'
    
    # Modal verbs
    strong_modals = r'\b(will|shall|must)\b'
    weak_modals = r'\b(may|might|can|could|may choose)\b'
    
    # Hedge words
    hedge_pattern = r'\b(where appropriate|reasonable|generally|may|from time to time|at our discretion|as applicable|if necessary)\b'
    
    # User action mechanisms
    user_action_pattern = r'\b(you can request|contact us|settings|opt out|opt-out|delete|download|unsubscribe|manage|access|withdraw|revoke)\b'
    
    # Numeric quantities
    numeric_pattern = r'\b\d+\b'
    
    # Named procedures/channels
    procedure_pattern = r'\b(email|phone|form|portal|dashboard|link|button|menu|page)\b'
    
    for clause in clauses:
        clause_lower = clause.lower()
        
        # 1. Time expressions
        specific_time_count = len(re.findall(specific_time_pattern, clause_lower, re.IGNORECASE))
        vague_time_count = len(re.findall(vague_time_pattern, clause_lower, re.IGNORECASE))
        has_specific_time = 1 if specific_time_count > 0 else 0
        has_vague_time = 1 if vague_time_count > 0 else 0
        
        # 2. Modal strength
        strong_modal_count = len(re.findall(strong_modals, clause_lower, re.IGNORECASE))
        weak_modal_count = len(re.findall(weak_modals, clause_lower, re.IGNORECASE))
        modal_strength_ratio = (strong_modal_count / (weak_modal_count + 1e-6)) if (strong_modal_count + weak_modal_count) > 0 else 0
        
        # 3. Hedge words
        hedge_count = len(re.findall(hedge_pattern, clause_lower, re.IGNORECASE))
        hedge_density = hedge_count / (len(clause.split()) + 1e-6)
        
        # 4. User action mechanisms
        user_action_count = len(re.findall(user_action_pattern, clause_lower, re.IGNORECASE))
        has_user_action = 1 if user_action_count > 0 else 0
        
        # 5. Numeric quantities
        numeric_count = len(re.findall(numeric_pattern, clause))
        
        # 6. Named procedures
        procedure_count = len(re.findall(procedure_pattern, clause_lower, re.IGNORECASE))
        has_procedure = 1 if procedure_count > 0 else 0
        
        features.append([
            specific_time_count,
            vague_time_count,
            has_specific_time,
            has_vague_time,
            strong_modal_count,
            weak_modal_count,
            modal_strength_ratio,
            hedge_count,
            hedge_density,
            user_action_count,
            has_user_action,
            numeric_count,
            procedure_count,
            has_procedure,
        ])
    
    rubric_features = np.array(features, dtype=np.float32)
    
    feature_names = [
        "specific_time_count",
        "vague_time_count",
        "has_specific_time",
        "has_vague_time",
        "strong_modal_count",
        "weak_modal_count",
        "modal_strength_ratio",
        "hedge_count",
        "hedge_density",
        "user_action_count",
        "has_user_action",
        "numeric_count",
        "procedure_count",
        "has_procedure",
    ]
    
    # Print feature means by class
    print(f"\n[RUBRIC] Feature means by verdict class:")
    print(f"{'Feature':<25} {'Not Addressed':<18} {'Partial':<18} {'Compliant':<18}")
    print("-" * 80)
    
    for i, fname in enumerate(feature_names):
        not_addr_mean = rubric_features[labels == 'Not Addressed', i].mean()
        partial_mean = rubric_features[labels == 'Partially Compliant', i].mean()
        compliant_mean = rubric_features[labels == 'Compliant', i].mean()
        print(f"{fname:<25} {not_addr_mean:<18.4f} {partial_mean:<18.4f} {compliant_mean:<18.4f}")
    
    return rubric_features, feature_names


# Feature extractor registry for incremental experiments
INCREMENTAL_EXTRACTORS = {
    "all_rules_sim": extract_all_rules_similarity,
    "rubric": extract_rubric_features,
}
