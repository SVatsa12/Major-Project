# Near-Duplicate Grouping Discrepancy Analysis

## Problem

- **fixed_folds_v1.json** (seeds 42-44): 639 groups, 60 multi-member
- **confirm_folds_v1.json** (seeds 45-49): 815 groups, 11 multi-member

Grouping should be IDENTICAL because it depends only on data, not seed.

## Root Cause: Different Input Text

### generate_fixed_folds.py (CORRECT)
```python
# Uses build_context_texts() which creates enriched text:
# "Policy category: {category}. {clause_text}"
tax_map = {item["taxonomy_id"]: item for item in taxonomy}
context_texts = build_context_texts(df, tax_map)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    context_texts,  # ← CONTEXT TEXT (policy category + clause)
    normalize_embeddings=True
)
```

### generate_confirm_folds.py (INCORRECT)
```python
# Uses raw clause_text only, NO category prefix:
clause_texts = df["clause_text"].tolist()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(
    clause_texts,  # ← RAW CLAUSE TEXT ONLY
    convert_to_numpy=True
)
```

## Differences

| Aspect | fixed_folds_v1.py | confirm_folds.py |
|--------|-------------------|------------------|
| **Input text** | `build_context_texts()` with category prefix | Raw `clause_text` only |
| **Normalization** | `normalize_embeddings=True` | No normalization |
| **Algorithm** | scipy.sparse.csgraph.connected_components | Manual DFS traversal |
| **Result** | 639 groups (60 multi-member) | 815 groups (11 multi-member) |

## Impact

Different input text → different embeddings → different similarity scores → different groups

Example:
```
# fixed_folds (with context):
"Policy category: Data Retention. We retain your data for 90 days."
  ↓ embed ↓
More context, richer semantics, more near-duplicates found

# confirm_folds (raw):
"We retain your data for 90 days."
  ↓ embed ↓
Less context, misses near-duplicates that differ only in category
```

## Why This Matters

With different groupings:
- **Fold splits are fundamentally different structures**
- Training fold groups may split differently in confirmation
- Near-duplicates in training might land in train+test in confirmation
- Comparisons between training and confirmation are INVALID

## Fix

1. Save the ORIGINAL group assignment from fixed_folds_v1 (639 groups)
2. Regenerate confirm_folds using IDENTICAL groups (load, don't recompute)
3. Re-run all confirmation experiments
4. Compare fairly (same grouping, different random seeds)

## Algorithm Differences (Secondary Issue)

Both use connected components but with different implementations:

**fixed_folds:** scipy sparse CSR matrix → connected_components  
**confirm_folds:** Manual DFS on adjacency dict

These should be equivalent IF input is identical, but the primary issue is the input text difference.
