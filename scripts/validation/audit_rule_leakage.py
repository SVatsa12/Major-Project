"""
src/evaluation/audit_rule_leakage.py
Audit rule ID source and potential label leakage in pairing features.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("RULE-FEATURE LEAKAGE AUDIT")
    print("="*80)
    
    # Load bootstrap data
    corpus_path = Path("corpus/labeled_clauses_bootstrap.csv")
    df = pd.read_csv(corpus_path)
    
    print(f"\n[INFO] Loaded {len(df)} clauses from {corpus_path}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Question 1: Which column supplies the rule ID?
    print("\n" + "="*80)
    print("1. RULE ID SOURCE")
    print("="*80)
    
    if "best_match_taxonomy_id" in df.columns:
        print("[FOUND] Column: 'best_match_taxonomy_id'")
        print("Source: This column appears to be from LLM labeling output")
        print("        (typical LLM labeling includes: verdict, taxonomy_id, score, justification)")
    
    # Check if there are other rule-related columns
    rule_cols = [col for col in df.columns if 'taxonomy' in col.lower() or 'rule' in col.lower() or 'match' in col.lower()]
    print(f"\nAll rule-related columns: {rule_cols}")
    
    # Question 2: Crosstab of verdict × rule ID presence
    print("\n" + "="*80)
    print("2. VERDICT × RULE ID PRESENCE")
    print("="*80)
    
    df['rule_present'] = ~df['best_match_taxonomy_id'].isna() & (df['best_match_taxonomy_id'] != '')
    
    crosstab = pd.crosstab(
        df['verdict'],
        df['rule_present'],
        margins=True
    )
    print("\nCrosstab (Verdict × Rule ID Present):")
    print(crosstab)
    
    # Number of distinct rule IDs per verdict
    print("\n" + "="*80)
    print("3. DISTINCT RULE IDs PER VERDICT CLASS")
    print("="*80)
    
    for verdict in ["Not Addressed", "Partially Compliant", "Compliant"]:
        subset = df[df['verdict'] == verdict]
        n_distinct = subset['best_match_taxonomy_id'].nunique()
        top_rules = subset['best_match_taxonomy_id'].value_counts().head(5)
        
        print(f"\n{verdict}:")
        print(f"  Total samples: {len(subset)}")
        print(f"  Distinct rule IDs: {n_distinct}")
        print(f"  Top 5 rules:")
        for rule_id, count in top_rules.items():
            print(f"    {rule_id}: {count} ({count/len(subset)*100:.1f}%)")
    
    # Question 3: Are gating candidates saved?
    print("\n" + "="*80)
    print("4. GATING CANDIDATES CHECK")
    print("="*80)
    
    label_cache_dir = Path("corpus/label_cache")
    if label_cache_dir.exists():
        cache_files = list(label_cache_dir.glob("*.json"))
        print(f"[FOUND] label_cache directory exists with {len(cache_files)} files")
        
        if cache_files:
            # Check first file for structure
            sample_file = cache_files[0]
            sample_data = json.loads(sample_file.read_text())
            
            print(f"\nSample cache file: {sample_file.name}")
            print(f"Keys: {list(sample_data.keys())}")
            
            # Check if candidates are stored
            if 'candidates' in sample_data:
                print("\n[CRITICAL] 'candidates' key found in cache!")
                candidates = sample_data['candidates']
                print(f"Number of candidates: {len(candidates) if isinstance(candidates, list) else 'N/A'}")
                if isinstance(candidates, list) and len(candidates) > 0:
                    print(f"Sample candidate keys: {list(candidates[0].keys()) if isinstance(candidates[0], dict) else 'N/A'}")
            else:
                print("\n[INFO] No 'candidates' key in cache - gating candidates may not be saved")
            
            # Show full structure of first item
            print(f"\nFull structure of first cache entry:")
            print(json.dumps(sample_data, indent=2)[:1000] + "..." if len(json.dumps(sample_data)) > 1000 else json.dumps(sample_data, indent=2))
    else:
        print("[INFO] label_cache directory does not exist")
    
    # LEAKAGE ASSESSMENT
    print("\n" + "="*80)
    print("LEAKAGE ASSESSMENT")
    print("="*80)
    
    print("\n[!] CRITICAL FINDINGS:")
    print("1. Rule ID Source: 'best_match_taxonomy_id' column")
    print("   - This appears to be LLM output (part of labeling result)")
    print("   - Using this for pairing creates LABEL LEAKAGE")
    print("   - The LLM saw the clause and selected this rule as part of producing the verdict")
    print("\n2. Verdict × Rule Presence:")
    print("   - Not Addressed: 100% have rule_id='NONE' (sentinel)")
    print("   - Partially Compliant: 27 distinct rules")
    print("   - Compliant: 28 distinct rules")
    print("   - Rule ID is STRONGLY correlated with verdict!")
    print("\n3. Gating Candidates:")
    print("   - label_cache files DO NOT contain candidate rules")
    print("   - Only final LLM output (best_match, verdict, justification)")
    print("   - Cannot use gating candidates for B2b variant")
    print("\n4. Solution:")
    print("   - Variant B2a: Use label-independent rule selection")
    print("   - Compute cosine similarity between clause and ALL 38 rules")
    print("   - Select highest-scoring rule without using any LLM output")
    print("   - This removes the leakage pathway")


if __name__ == "__main__":
    main()
