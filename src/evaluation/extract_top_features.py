"""
src/evaluation/extract_top_features.py
Extract top 10 most influential all-rules similarity features
by absolute LogisticRegression coefficient per class.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib


def main():
    print("=" * 120)
    print("EXTRACTING TOP 10 INFLUENTIAL ALL-RULES FEATURES PER CLASS")
    print("=" * 120)
    
    # Load model
    model_path = Path("models/final_classifier/model.joblib")
    print(f"\n[INFO] Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load feature config
    feature_config_path = Path("models/final_classifier/feature_config.json")
    with open(feature_config_path, 'r', encoding='utf-8') as f:
        feature_config = json.load(f)
    
    print(f"[INFO] Total features: {feature_config['total_features']}")
    print(f"[INFO] Feature breakdown:")
    for k, v in feature_config['feature_breakdown'].items():
        print(f"  - {k}: {v}")
    
    # Load taxonomy to get rule IDs and texts
    taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    # Filter out NONE and get rule info
    all_rules = [item for item in taxonomy if item["taxonomy_id"] != "NONE"]
    rule_ids = [item["taxonomy_id"] for item in all_rules]
    rule_requirements = [item["requirement"] for item in all_rules]
    rule_categories = [item.get("category", "Unknown") for item in all_rules]
    
    print(f"\n[INFO] Loaded {len(all_rules)} rules from taxonomy")
    
    # Get unique categories for per-category max features
    unique_categories = sorted(list(set(rule_categories)))
    print(f"[INFO] Categories: {len(unique_categories)}")
    
    # Feature indices for all-rules similarity
    # Structure: [E5 embeddings (768), actionability (8), all-rules (49)]
    # all-rules = [38 rule similarities, 11 category max]
    e5_dims = feature_config['feature_breakdown']['e5_embeddings']
    action_dims = feature_config['feature_breakdown']['actionability_features']
    all_rules_start = e5_dims + action_dims
    all_rules_dims = feature_config['feature_breakdown']['all_rules_similarity']
    
    print(f"\n[INFO] All-rules features start at index {all_rules_start}, span {all_rules_dims} dimensions")
    
    # Get model coefficients
    coef = model.coef_  # shape: (n_classes, n_features)
    classes = model.classes_
    
    print(f"[INFO] Model classes: {classes.tolist()}")
    print(f"[INFO] Coefficient matrix shape: {coef.shape}")
    
    # Extract all-rules coefficients
    all_rules_coef = coef[:, all_rules_start:all_rules_start+all_rules_dims]
    
    # First 38 are individual rule similarities
    n_rules = len(rule_ids)
    rule_sim_coef = all_rules_coef[:, :n_rules]
    
    print(f"\n[INFO] Rule similarity coefficients shape: {rule_sim_coef.shape}")
    
    # For each class, get top 10 by absolute coefficient
    output_dir = Path("reports/paper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_class_features = []
    
    for class_idx, class_name in enumerate(classes):
        print(f"\n" + "=" * 120)
        print(f"CLASS: {class_name}")
        print("=" * 120)
        
        class_coef = rule_sim_coef[class_idx, :]
        abs_coef = np.abs(class_coef)
        
        # Get top 10 indices
        top_10_indices = np.argsort(abs_coef)[-10:][::-1]
        
        print(f"\nTop 10 Most Influential All-Rules Features:")
        print(f"{'Rank':<6} {'Rule ID':<12} {'Coefficient':<15} {'Abs Coef':<15} {'Category':<25}")
        print("-" * 120)
        
        for rank, idx in enumerate(top_10_indices, 1):
            rule_id = rule_ids[idx]
            coefficient = class_coef[idx]
            abs_coefficient = abs_coef[idx]
            category = rule_categories[idx]
            requirement_text = rule_requirements[idx][:80] + "..." if len(rule_requirements[idx]) > 80 else rule_requirements[idx]
            
            print(f"{rank:<6} {rule_id:<12} {coefficient:<15.6f} {abs_coefficient:<15.6f} {category:<25}")
            print(f"       Requirement: {requirement_text}")
            print()
            
            all_class_features.append({
                "class": class_name,
                "rank": rank,
                "rule_id": rule_id,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs_coefficient),
                "category": category,
                "requirement": rule_requirements[idx]
            })
    
    # Save to CSV
    df = pd.DataFrame(all_class_features)
    csv_path = output_dir / "top_10_features_per_class.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Saved top features to {csv_path}")
    
    # Save to JSON with full details
    json_path = output_dir / "top_10_features_per_class.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_class_features, f, indent=2)
    print(f"[SUCCESS] Saved top features (with full text) to {json_path}")
    
    print("\n[SUCCESS] Feature extraction complete")


if __name__ == "__main__":
    main()
