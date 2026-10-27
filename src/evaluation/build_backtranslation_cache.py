"""
src/evaluation/build_backtranslation_cache.py
Back-translation paraphrases (en -> de -> en, Helsinki-NLP MarianMT) for every
Partially Compliant clause, cached by source-text hash.

Generation is deterministic (beam search): en->de returns the top-2 beams, each
is translated back to English with beam search, giving up to 2 paraphrases per
clause. Paraphrases depend only on the source clause text. The cache holds one
entry per Partially Compliant clause in the corpus; run_round2_variants.py adds
a clause's paraphrases to a model only when that clause is in the model's own
training rows (asserted per fold).

Run: python src/evaluation/build_backtranslation_cache.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CACHE_PATH = Path("reports/experiments/augmentation/backtranslation_en_de_en.json")
EN_DE = "Helsinki-NLP/opus-mt-en-de"
DE_EN = "Helsinki-NLP/opus-mt-de-en"
N_PARAPHRASES = 2
NUM_BEAMS = 4
MAX_LENGTH = 512
BATCH_SIZE = 8


def text_key(text):
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()[:16]


def translate(texts, tokenizer, model, num_return_sequences=1):
    outputs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer(texts[start:start + BATCH_SIZE], return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            generated = model.generate(**batch, num_beams=NUM_BEAMS, num_return_sequences=num_return_sequences,
                                       max_length=MAX_LENGTH)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def main():
    torch.manual_seed(0)
    df = pd.read_csv("corpus/labeled_clauses_bootstrap.csv")
    sources = df.loc[df["verdict"] == "Partially Compliant", "clause_text"].astype(str).tolist()
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {"entries": {}}
    todo = [t for t in dict.fromkeys(sources) if text_key(t) not in cache["entries"]]
    print(f"[INFO] {len(sources)} Partially Compliant clauses, {len(todo)} not yet cached")

    if todo:
        tok_ende, mt_ende = MarianTokenizer.from_pretrained(EN_DE), MarianMTModel.from_pretrained(EN_DE).eval()
        tok_deen, mt_deen = MarianTokenizer.from_pretrained(DE_EN), MarianMTModel.from_pretrained(DE_EN).eval()
        german = translate(todo, tok_ende, mt_ende, num_return_sequences=N_PARAPHRASES)
        english = translate(german, tok_deen, mt_deen)
        for i, text in enumerate(todo):
            paraphrases = english[i * N_PARAPHRASES:(i + 1) * N_PARAPHRASES]
            cache["entries"][text_key(text)] = {"source_text": text, "paraphrases": paraphrases}
            print(f"  [{i + 1}/{len(todo)}] {text[:60]!r} -> {paraphrases[0][:60]!r}")

    cache["metadata"] = {
        "models": [EN_DE, DE_EN], "n_paraphrases": N_PARAPHRASES, "num_beams": NUM_BEAMS,
        "max_length": MAX_LENGTH, "key": "sha256(whitespace-normalised source text)[:16]",
        "sources": "every Partially Compliant clause in corpus/labeled_clauses_bootstrap.csv",
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    entries = cache["entries"].values()
    identical = sum(p.strip().lower() == e["source_text"].strip().lower() for e in entries for p in e["paraphrases"])
    print(f"[SUCCESS] {len(cache['entries'])} entries in {CACHE_PATH}; {identical} paraphrases identical to source")


if __name__ == "__main__":
    main()
