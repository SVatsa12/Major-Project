"""
src/evaluation/analyze_taxonomy_provenance.py
Analyze which fields from dpdp_taxonomy_final.json are embedded for rule similarity.
"""

import json
from pathlib import Path
import pandas as pd


def analyze_taxonomy_fields():
    """Show which taxonomy fields are used for embedding."""
    
    # Load taxonomy
    taxonomy_path = Path("corpus/dpdp_taxonomy_final.json")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    
    # Get one example rule
    example_rule = taxonomy[0]
    
    print("=" * 80)
    print("TAXONOMY FIELD PROVENANCE ANALYSIS")
    print("=" * 80)
    
    print("\n1. CURRENT IMPLEMENTATION:")
    print("-" * 80)
    print("Field used for embedding: item['requirement']")
    print("\nExample (DPDP_ACT-0026):")
    print(f"  {example_rule['requirement']}")
    
    print("\n2. ALL AVAILABLE FIELDS PER RULE:")
    print("-" * 80)
    for key in example_rule.keys():
        value = example_rule[key]
        if isinstance(value, str) and len(value) > 100:
            value_preview = value[:100] + "..."
        else:
            value_preview = value
        print(f"  {key}: {value_preview}")
    
    print("\n3. FIELD ORIGINS:")
    print("-" * 80)
    
    statutory_fields = [
        "requirement",  # Core statutory text
        "source",       # Legal citation
        "applicability" # Legal scope
    ]
    
    derived_fields = [
        "checkable_test",     # Generated test description
        "required_concepts",  # Extracted from social media policies (candidates)
        "exclude_concepts"    # Generated exclusion patterns
    ]
    
    print("STATUTORY (from DPDP Act 2023):")
    for field in statutory_fields:
        print(f"  ✓ {field}")
    
    print("\nDERIVED (from social media policy analysis or LLM generation):")
    for field in derived_fields:
        print(f"  ⚠️ {field}")
    
    print("\n4. STATISTICS:")
    print("-" * 80)
    
    # Count rules
    rules_without_none = [r for r in taxonomy if r["taxonomy_id"] != "NONE"]
    print(f"Total rules (excl. NONE): {len(rules_without_none)}")
    
    # Average field lengths
    req_lengths = [len(r["requirement"]) for r in rules_without_none]
    test_lengths = [len(r.get("checkable_test", "")) for r in rules_without_none]
    concept_lengths = [len(r.get("required_concepts", "")) for r in rules_without_none]
    
    print(f"\nAverage field lengths (chars):")
    print(f"  requirement:       {sum(req_lengths) / len(req_lengths):.1f}")
    print(f"  checkable_test:    {sum(test_lengths) / len(test_lengths):.1f}")
    print(f"  required_concepts: {sum(concept_lengths) / len(concept_lengths):.1f}")
    
    print("\n5. RECOMMENDATION:")
    print("-" * 80)
    print("✓ CURRENT: Embed only 'requirement' field (statutory text)")
    print("⚠️ VARIANT: Test embedding ONLY requirement (no derived fields)")
    print("  This variant already matches current implementation.")
    print("\nAlternative variants to test:")
    print("  - requirement + source (with citation)")
    print("  - requirement + applicability (with scope)")
    print("  - requirement + checkable_test (with interpretation)")
    
    # Show a few examples
    print("\n6. EXAMPLE RULES (first 3):")
    print("-" * 80)
    
    for i, rule in enumerate(rules_without_none[:3], 1):
        print(f"\n[{i}] {rule['taxonomy_id']} - {rule.get('category', 'Unknown')}")
        print(f"    Requirement: {rule['requirement'][:100]}...")
        print(f"    Source: {rule.get('source', 'N/A')}")
        print(f"    Checkable: {rule.get('checkable_test', 'N/A')[:80]}...")


if __name__ == "__main__":
    analyze_taxonomy_fields()
