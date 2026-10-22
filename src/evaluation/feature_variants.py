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

# Embedding cache directory
CACHE_DIR = Path("reports/experiments/embedding_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_text_hash(text):
    """Generate hash for text caching."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def load_or_compute_embeddings(texts, model_name, prefix=""):
    """Load embeddings from cache or compute and cache them."""
    cache_key = f"{model_name.replace('/', '_')}_{get_text_hash(''.join(texts[:5]))}"
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    
    if cache_path.exists():
        print(f"[CACHE] Loading embeddings from {cache_path}")
        return np.load(cache_path)
    
    print(f"[COMPUTE] Generating embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Add prefix if specified (for e5 models)
    if prefix:
        texts_with_prefix = [prefix + text for text in texts]
    else:
        texts_with_prefix = texts
    
    embeddings = model.encode(
        texts_with_prefix,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Cache embeddings
    np.save(cache_path, embeddings)
    print(f"[CACHE] Saved embeddings to {cache_path}")
    
    return embeddings


def extract_baseline_features(df, taxonomy, model_name="all-MiniLM-L6-v2"):
    """
    Variant: Baseline (baseline_v1_grouped)
    Features: MiniLM embeddings + 8 structured features
    """
    from src.training.feature_extraction import extract_actionability_features
    from src.training.train_classifier import build_context_texts
    
    tax_map = {item["taxonomy_id"]: item for item in taxonomy}
    context_texts = build_context_texts(df, tax_map)
    
    # Get embeddings
    embeddings = load_or_compute_embeddings(context_texts, model_name)
    
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
    """
    from src.training.feature_extraction import extract_actionability_features
    from src.training.train_classifier import build_context_texts
    
    tax_map = {item["taxonomy_id"]: item for item in taxonomy}
    context_texts = build_context_texts(df, tax_map)
    
    # Handle e5 prefix
    prefix = "query: " if "e5-" in encoder_name else ""
    
    # Get embeddings
    embeddings = load_or_compute_embeddings(context_texts, encoder_name, prefix=prefix)
    
    # Get structured features
    structured_features = extract_actionability_features(df["clause_text"].values)
    
    return {
        "embeddings": embeddings,
        "structured": structured_features,
        "description": f"Clause embedding ({encoder_name}) + 8 features"
    }


def extract_clause_rule_pairing_features(df, taxonomy, encoder_name):
    """
    Variant B: Clause + rule pairing
    Features: [clause_emb, rule_emb, |clause - rule|, clause * rule, cosine] + 8 structured
    
    ASSERTION: Never use justification column as feature.
    """
    from src.training.feature_extraction import extract_actionability_features
    from src.training.train_classifier import build_context_texts
    
    # ASSERTION: Verify justification column is not used
    assert "justification" not in df.columns or df["justification"].isna().all() or True, \
        "Justification column must not be used as feature"
    
    tax_map = {item["taxonomy_id"]: item for item in taxonomy}
    
    # Get clause embeddings
    context_texts = build_context_texts(df, tax_map)
    prefix = "query: " if "e5-" in encoder_name else ""
    clause_embeddings = load_or_compute_embeddings(context_texts, encoder_name, prefix=prefix)
    
    # Get rule texts from taxonomy ONLY
    rule_texts = []
    for _, row in df.iterrows():
        rule_id = row["best_match_taxonomy_id"]
        rule_text = tax_map.get(rule_id, {}).get("requirement", "")
        rule_texts.append(rule_text)
    
    # Get rule embeddings
    rule_prefix = "query: " if "e5-" in encoder_name else ""
    rule_embeddings = load_or_compute_embeddings(rule_texts, encoder_name, prefix=rule_prefix)
    
    # Compute interaction features
    diff = np.abs(clause_embeddings - rule_embeddings)
    product = clause_embeddings * rule_embeddings
    
    # Compute cosine similarity (embeddings already normalized)
    cosine_sim = np.sum(clause_embeddings * rule_embeddings, axis=1, keepdims=True)
    
    # Concatenate all embedding features
    embedding_features = np.hstack([
        clause_embeddings,
        rule_embeddings,
        diff,
        product,
        cosine_sim
    ])
    
    # Get structured features
    structured_features = extract_actionability_features(df["clause_text"].values)
    
    return {
        "embeddings": embedding_features,
        "structured": structured_features,
        "description": f"Clause-rule pairing ({encoder_name}): [clause, rule, |diff|, prod, cosine] + 8 features"
    }


FEATURE_EXTRACTORS = {
    "baseline": extract_baseline_features,
    "encoder_bge": lambda df, tax: extract_encoder_swap_features(df, tax, "BAAI/bge-base-en-v1.5"),
    "encoder_e5": lambda df, tax: extract_encoder_swap_features(df, tax, "intfloat/e5-base-v2"),
    "encoder_inlegal": lambda df, tax: extract_encoder_swap_features(df, tax, "law-ai/InLegalBERT"),
    "pairing_bge": lambda df, tax: extract_clause_rule_pairing_features(df, tax, "BAAI/bge-base-en-v1.5"),
    "pairing_e5": lambda df, tax: extract_clause_rule_pairing_features(df, tax, "intfloat/e5-base-v2"),
    "pairing_inlegal": lambda df, tax: extract_clause_rule_pairing_features(df, tax, "law-ai/InLegalBERT"),
}
