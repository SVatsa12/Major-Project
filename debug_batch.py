"""
Single-batch smoke test for generate_taxonomy.py
-------------------------------------------------
Runs exactly ONE batch (up to 8 clauses) from the real filtered dataset through
the model using the same logic as generate_taxonomy.py, then prints a detailed
pass/fail report.

Run:
    python debug_batch.py

The script reads the updated SYSTEM_PROMPT and parsing helpers from
generate_taxonomy.py so they stay in sync automatically.
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the src directory so we can import helpers from generate_taxonomy.py
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent / "src" / "taxonomy"
sys.path.insert(0, str(SRC_DIR))

try:
    from generate_taxonomy import (
        SYSTEM_PROMPT,
        build_user_message,
        extract_json_array,
        validate_entry,
        ALLOWED_CATEGORIES,
        SAME_CATEGORY_DUPLICATE_THRESHOLD,
        _get_dedup_model,
    )
except ImportError as exc:
    sys.exit(f"ERROR: Could not import from generate_taxonomy.py: {exc}")

try:
    from sentence_transformers import util
except ImportError:
    util = None  # type: ignore

try:
    import ollama
except ImportError:
    sys.exit("ERROR: ollama package not installed. Run: pip install ollama")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH = Path("datasets/GovernmentActs/government_clauses_filtered.json")
MODEL = "qwen3:8b"
BATCH_SIZE = 8          # number of clauses to test
MIN_CLAUSE_LEN = 80     # skip very short / header clauses


# ---------------------------------------------------------------------------
# Load a batch of real, substantive clauses
# ---------------------------------------------------------------------------
def load_test_batch(path: Path, size: int, min_len: int) -> list[dict]:
    if not path.exists():
        sys.exit(f"ERROR: Dataset not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("records", [])

    # Prefer high-relevance DPDP clauses; fall back to any long clause
    high = [
        r for r in records
        if r.get("relevance_signal_level") == "high"
        and len(r.get("clause_text", "")) >= min_len
        and "DPDP" in r.get("source_act", "")
    ]
    pool = high if len(high) >= size else [
        r for r in records if len(r.get("clause_text", "")) >= min_len
    ]

    batch = []
    seen_texts: set[str] = set()
    for r in pool:
        text = r.get("clause_text", "").strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            batch.append({
                "representative_clause_id": str(r["clause_id"]),
                "clause_text": text,
                "source_act": str(r.get("source_act", "")),
                "source_file": str(r.get("source_file", "")),
                "page_start": str(r.get("page_start", "")),
                "section_heading": str(r.get("section_heading", "") or ""),
            })
        if len(batch) >= size:
            break

    if not batch:
        sys.exit("ERROR: No suitable clauses found in dataset.")
    return batch


# ---------------------------------------------------------------------------
# Call the model (no format="json" -- same fix as generate_taxonomy.py)
# ---------------------------------------------------------------------------
def run_batch(batch: list[dict]) -> tuple[str, list]:
    user_message = build_user_message(batch)

    print(f"\n[DEBUG] Sending {len(batch)} clauses to {MODEL}...")
    print("[DEBUG] Clause IDs:", [b["representative_clause_id"] for b in batch])

    response = ollama.chat(
        model=MODEL,
        think=False,
        # format="json" intentionally omitted -- forces a JSON object wrapper
        # that breaks array extraction (same fix as generate_taxonomy.py)
        options={"num_predict": 8192, "temperature": 0.1},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = str(response["message"]["content"]).strip()
    print(f"\n[DEBUG] Raw response ({len(raw)} chars):\n{raw[:1000]}{'...' if len(raw) > 1000 else ''}")
    return raw, extract_json_array(raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("  generate_taxonomy.py  --  single-batch smoke test")
    print("=" * 60)

    batch = load_test_batch(INPUT_PATH, BATCH_SIZE, MIN_CLAUSE_LEN)
    print(f"\nLoaded {len(batch)} test clauses from {INPUT_PATH}")

    allowed_ids = {item["representative_clause_id"] for item in batch}

    try:
        raw_response, parsed = run_batch(batch)
    except Exception as exc:
        print(f"\n[FAIL] Model call or JSON extraction failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"\n[OK] Parsed {len(parsed)} entries from model response")

    valid, invalid = [], []
    for i, entry in enumerate(parsed):
        norm, error = validate_entry(entry, allowed_ids)
        if norm:
            valid.append(norm)
        else:
            invalid.append({"position": i, "entry": entry, "error": error})

    # ----- Embedding-aware conflict resolution (mirrors call_model in generate_taxonomy.py) -----
    # Different categories sharing a clause_id = distinct obligations → KEEP ALL.
    # Same category sharing a clause_id = run cosine similarity → discard if near-identical.
    from collections import defaultdict
    source_to_positions: dict[str, list[int]] = defaultdict(list)
    for pos, entry in enumerate(valid):
        for cid in entry.get("source_clause_ids", []):
            source_to_positions[cid].append(pos)

    discard_positions: set[int] = set()
    for source_id, positions in source_to_positions.items():
        if len(positions) <= 1:
            continue
        groups_by_category: dict[str, list[int]] = defaultdict(list)
        for pos in positions:
            groups_by_category[valid[pos]["category"]].append(pos)

        kept_local: list[int] = []
        discarded_local: list[int] = []
        cats_str = ", ".join(groups_by_category.keys())
        print(f"\n[MULTI-MAPPED] {source_id} -> {len(positions)} entries across: {cats_str}")

        for cat, cat_positions in groups_by_category.items():
            if len(cat_positions) == 1:
                kept_local.append(cat_positions[0])
                print(f"  [KEEP] pos={cat_positions[0]} category='{cat}' (only entry in category)")
                continue
            embed_model = _get_dedup_model()
            if embed_model is None or util is None:
                kept_local.append(cat_positions[0])
                discarded_local.extend(cat_positions[1:])
                print(f"  [KEEP-FIRST] pos={cat_positions[0]} (no embedding model; discarded {cat_positions[1:]})")
                continue
            texts = [valid[p]["requirement"] for p in cat_positions]
            embeddings = embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            local_kept = [cat_positions[0]]
            for i in range(1, len(cat_positions)):
                sim = float(util.cos_sim(embeddings[i], embeddings[cat_positions.index(local_kept[0])]).item())
                if sim >= SAME_CATEGORY_DUPLICATE_THRESHOLD:
                    discarded_local.append(cat_positions[i])
                    print(f"  [DISCARD] pos={cat_positions[i]} sim={sim:.3f} >= threshold -> duplicate")
                else:
                    local_kept.append(cat_positions[i])
                    print(f"  [KEEP] pos={cat_positions[i]} sim={sim:.3f} < threshold -> distinct")
            kept_local.extend(local_kept)

        discard_positions.update(discarded_local)

    if discard_positions:
        for pos in sorted(discard_positions):
            invalid.append({
                "position": pos,
                "entry": valid[pos],
                "error": "discarded as same-category, semantically similar duplicate of another entry sharing this source clause",
            })
        valid = [e for pos, e in enumerate(valid) if pos not in discard_positions]
        print(f"\n[DEDUP] Discarded {len(discard_positions)} duplicate entries; {len(valid)} remain")

    # ----- Report -----
    print("\n" + "=" * 60)
    print(f"  RESULTS: {len(valid)} valid  |  {len(invalid)} invalid  |  {len(parsed)} total")
    print("=" * 60)

    if valid:
        print("\n--- Valid entries ---")
        for v in valid:
            print(json.dumps(v, indent=2, ensure_ascii=False))

    if invalid:
        print("\n--- Invalid entries ---")
        for inv in invalid:
            print(f"  [position {inv['position']}] ERROR: {inv['error']}")
            print(f"  entry: {json.dumps(inv['entry'], ensure_ascii=False)[:300]}")

    unmapped = sorted(allowed_ids - {
        cid
        for e in valid
        for cid in e.get("source_clause_ids", [])
    })
    if unmapped:
        print(f"\n--- Unmapped clause IDs ({len(unmapped)}) ---")
        for cid in unmapped:
            print(f"  {cid}")

    if valid and not invalid:
        print("\nPASS -- all parsed entries are valid")
        return 0
    elif valid:
        print("\nPARTIAL -- some entries valid, some invalid")
        return 0
    else:
        print("\nFAIL -- no valid entries produced")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())