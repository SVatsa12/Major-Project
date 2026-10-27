"""
src/evaluation/feature_variants.py
Feature extraction variants for evaluation experiments.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.training.model_inputs import assert_label_free, build_input_texts, encode_texts, encoder_prefix

# Embedding cache directory
CACHE_DIR = Path("reports/experiments/embedding_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_text_hash(text):
    """Generate hash for text caching."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def load_or_compute_embeddings(texts, model_name, cache_suffix=""):
    """
    Load embeddings from cache or compute and cache them.
    
    The cache key covers the model, the encoder prefix and EVERY input text, so
    a change to the input format can never silently reuse stale embeddings.
    Texts are encoded by encode_texts (encoder prefix + normalisation), the same
    function the final model and the predictor use.
    """
    prefix = encoder_prefix(model_name)
    content_hash = get_text_hash(json.dumps([model_name, prefix, list(texts)]))
    cache_key = f"{model_name.replace('/', '_')}{cache_suffix}_{content_hash}"
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    
    if cache_path.exists():
        print(f"[CACHE] Loading embeddings from {cache_path}")
        return np.load(cache_path)
    
    print(f"[COMPUTE] Generating embeddings with {model_name} (prefix={prefix!r})...")
    model = SentenceTransformer(model_name)
    embeddings = encode_texts(model, list(texts), model_name, show_progress_bar=True)
    
    # Cache embeddings
    np.save(cache_path, embeddings)
    print(f"[CACHE] Saved embeddings to {cache_path}")
    
    return embeddings


def extract_baseline_features(df, taxonomy, model_name="all-MiniLM-L6-v2"):
    """
    Variant: Baseline (baseline_v1_grouped)
    Features: MiniLM embeddings + 8 structured features
    
    df must be model_inputs(...) (clause_text, app_name only).
    """
    from src.training.feature_extraction import extract_actionability_features
    
    assert_label_free(df)
    input_texts = build_input_texts(df)
    
    # Get embeddings
    embeddings = load_or_compute_embeddings(input_texts, model_name)
    
    # Get structured features
    structured_features = extract_actionability_features(df["clause_text"].values)
    
    return {
        "embeddings": embeddings,
        "structured": structured_features,
        "description": f"Clause embedding ({model_name}) + 8 features"
    }


def extract_encoder_swap_features(df, taxonomy, encoder_name):
    """
    Variant A: Encoder swap
    Test different encoders: bge-base, e5-base, InLegalBERT
    
    df must be model_inputs(...) (clause_text, app_name only).
    Input text = encoder_prefix(encoder_name) + clause_text ("query: " for E5).
    """
    from src.training.feature_extraction import extract_actionability_features
    
    assert_label_free(df)
    input_texts = build_input_texts(df)
    
    # Get embeddings (encoder prefix, e.g. "query: " for e5, added inside)
    embeddings = load_or_compute_embeddings(input_texts, encoder_name)
    
    # Get structured features
    structured_features = extract_actionability_features(df["clause_text"].values)
    
    return {
        "embeddings": embeddings,
        "structured": structured_features,
        "description": f"Clause embedding ({encoder_name}) + 8 features"
    }


# Variant B (extract_clause_rule_pairing_features) was removed: it selected the
# paired rule from best_match_taxonomy_id, an LLM output that is NONE exactly when
# the label is "Not Addressed". It cannot be made label-free; B2a below is its
# label-independent replacement.


def extract_clause_rule_pairing_b2a(df, taxonomy, encoder_name="intfloat/e5-base-v2"):
    """
    Variant B2a: Clause + rule pairing with LABEL-INDEPENDENT rule selection
    
    Rule selection: Highest cosine similarity between clause and ALL 38 rules
    Features: [clause_emb, rule_emb, |diff|, prod, cosine, top3_sims] + 8 structured
    
    df must be model_inputs(...) (clause_text, app_name only).
    """
    from src.training.feature_extraction import extract_actionability_features
    
    assert_label_free(df)
    
    # Get clause embeddings
    input_texts = build_input_texts(df)
    clause_embeddings = load_or_compute_embeddings(input_texts, encoder_name)
    
    # Get ALL rule embeddings (label-independent)
    all_rules = [item for item in taxonomy if item["taxonomy_id"] != "NONE"]
    rule_texts_all = [item["requirement"] for item in all_rules]
    rule_ids_all = [item["taxonomy_id"] for item in all_rules]
    
    print(f"[B2a] Computing embeddings for {len(rule_texts_all)} rules...")
    rule_embeddings_all = load_or_compute_embeddings(rule_texts_all, encoder_name, cache_suffix="_rules")
    
    # For each clause, select rule with highest cosine similarity
    selected_rule_embeddings = []
    top3_similarities = []
    
    for i, clause_emb in enumerate(clause_embeddings):
        # Compute cosine similarities to all rules
        similarities = clause_emb @ rule_embeddings_all.T  # Already normalized
        
        # Get top-3 for features
        top3_idx = np.argsort(similarities)[-3:][::-1]
        top3_sims = similarities[top3_idx]
        top3_similarities.append(top3_sims)
        
        # Select best rule
        best_idx = top3_idx[0]
        selected_rule_embeddings.append(rule_embeddings_all[best_idx])
    
    selected_rule_embeddings = np.array(selected_rule_embeddings)
    top3_similarities = np.array(top3_similarities)
    
    # Compute interaction features
    diff = np.abs(clause_embeddings - selected_rule_embeddings)
    product = clause_embeddings * selected_rule_embeddings
    cosine_sim = np.sum(clause_embeddings * selected_rule_embeddings, axis=1, keepdims=True)
    
    # Concatenate all embedding features
    embedding_features = np.hstack([
        clause_embeddings,
        selected_rule_embeddings,
        diff,
        product,
        cosine_sim,
        top3_similarities  # Add top-3 similarities
    ])
    
    # Get structured features
    structured_features = extract_actionability_features(df["clause_text"].values)
    
    return {
        "embeddings": embedding_features,
        "structured": structured_features,
        "description": f"Clause-rule pairing B2a ({encoder_name}): LABEL-INDEPENDENT rule selection + top-3 sims + 8 features"
    }


FEATURE_EXTRACTORS = {
    "baseline": extract_baseline_features,
    "encoder_bge": lambda df, tax: extract_encoder_swap_features(df, tax, "BAAI/bge-base-en-v1.5"),
    "encoder_e5": lambda df, tax: extract_encoder_swap_features(df, tax, "intfloat/e5-base-v2"),
    "encoder_inlegal": lambda df, tax: extract_encoder_swap_features(df, tax, "law-ai/InLegalBERT"),
    "pairing_e5_b2a": lambda df, tax: extract_clause_rule_pairing_b2a(df, tax, "intfloat/e5-base-v2"),
}
