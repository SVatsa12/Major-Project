"""
src/training/inspect_candidates.py
Prints each discovered candidate alongside its text and statutory justification.
"""

from pathlib import Path
import pandas as pd

CANDIDATES_PATH = Path("corpus/candidates_for_compliance.csv")
df = pd.read_csv(CANDIDATES_PATH)

print(f"Total candidates to review: {len(df)}\n")
for i, row in df.iterrows():
    print(f"[{i+1}/{len(df)}] ID: {row['clause_id']} | App: {row['app_name']}")
    print(f"Target: {row['statutory_target']}")
    print(f"Text: {row['clause_text']}")
    print("-" * 80)