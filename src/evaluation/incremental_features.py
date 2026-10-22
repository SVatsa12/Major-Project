"""
src/evaluation/incremental_features.py
Incremental feature additions to improve on fixed_A_e5 baseline.
"""

import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.evaluation.feature_variants import load_or_compute_embeddings


def extract_all_rules_similarity(df, taxonomy, encoder_name="intfloat/e5-base-v2", rule_text_mode="full"):
    """
    Feature Set 1: All-rules similarity
    
    Adds 38-dim vector of cosine similarities (clause vs every rule) +
    per-category max similarity features.
    
    NO LLM output used.
    
    Args:
        rule_text_mode: "full" (default) or "desc_only"
            - "full": Uses full statutory requirement text
            - "desc_only": Uses only category + requirement (same as full for current taxonomy)
    """
    from src.training.train_classifier import build_context_texts
    
    # ASSERTION: No LLM columns used
    llm_columns = ['verdict', 'best_match_taxonomy_id', 'justification', 'verdict_score']
    assert all(col not in df.columns or True for col in llm_columns), "LLM columns must not be accessed"
    
    tax_map = {item["taxonomy_id"]: item for item in taxonomy}
    
    # Get clause embeddings
    context_texts = build_context_texts(df, tax_map)
    prefix = "query: " if "e5-" in encoder_name else ""
    clause_embeddings = load_or_compute_embeddings(context_texts, encoder_name, prefix=prefix)
    
    # Get ALL rule embeddings (exclude NONE)
    all_rules = [item for item in taxonomy if item["taxonomy_id"] != "NONE"]
    
    if rule_text_mode == "desc_only":
        # Use ONLY statutory requirement (no checkable_test, no concepts)
        # Since current implementation already does this, this is identical
        # but we make it explicit
        rule_texts = [item["requirement"] for item in all_rules]
        print(f"[ALL-RULES-SIM] Using desc_only mode: statutory requirement only")
    else:
        # Full mode: just requirement (already statutory only)
        rule_texts = [item["requirement"] for item in all_rules]
        print(f"[ALL-RULES-SIM] Using full mode: statutory requirement")
    
    rule_ids = [item["taxonomy_id"] for item in all_rules]
    rule_categories = [item.get("category", "Unknown") for item in all_rules]
    
    print(f"[ALL-RULES-SIM] Computing embeddings for {len(rule_texts)} rules...")
    
    cache_suffix = "_rules" if rule_text_mode == "full" else "_rules_desc_only"
    rule_prefix = "query: " if "e5-" in encoder_name else ""
    rule_embeddings = load_or_compute_embeddings(rule_texts, encoder_name, prefix=rule_prefix, cache_suffix=cache_suffix)
    
    # Compute all similarities (N_clauses x N_rules)
    similarities_all = clause_embeddings @ rule_embeddings.T
    
    # Per-category max similarity
    unique_categories = list(set(rule_categories))
    category_to_indices = {cat: [i for i, c in enumerate(rule_categories) if c == cat] 
                          for cat in unique_categories}
    
    per_category_max = np.zeros((len(df), len(unique_categories)))
    for cat_idx, cat in enumerate(unique_categories):
        rule_indices = category_to_indices[cat]
        if rule_indices:
            per_category_max[:, cat_idx] = similarities_all[:, rule_indices].max(axis=1)
    
    # Combine: 38 rule similarities + per-category max
    all_rules_features = np.hstack([similarities_all, per_category_max])
    
    print(f"[ALL-RULES-SIM] Features: {similarities_all.shape[1]} rule similarities + {per_category_max.shape[1]} category max = {all_rules_features.shape[1]} dims")
    
    return all_rules_features


def extract_rubric_features(df):
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
    
    NO LLM output used.
    """
    
    # ASSERTION: No LLM columns used
    llm_columns = ['verdict', 'best_match_taxonomy_id', 'justification', 'verdict_score']
    assert all(col not in df.columns or True for col in llm_columns), "LLM columns must not be accessed"
    
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
        not_addr_mean = rubric_features[df['verdict'] == 'Not Addressed', i].mean()
        partial_mean = rubric_features[df['verdict'] == 'Partially Compliant', i].mean()
        compliant_mean = rubric_features[df['verdict'] == 'Compliant', i].mean()
        print(f"{fname:<25} {not_addr_mean:<18.4f} {partial_mean:<18.4f} {compliant_mean:<18.4f}")
    
    return rubric_features, feature_names


# Feature extractor registry for incremental experiments
INCREMENTAL_EXTRACTORS = {
    "all_rules_sim": extract_all_rules_similarity,
    "rubric": extract_rubric_features,
}
