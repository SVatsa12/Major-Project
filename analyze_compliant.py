import pandas as pd
df = pd.read_csv('corpus/labeled_clauses_bootstrap.csv')
compliant = df[df['verdict'] == 'Compliant']
print(f"Total Compliant clauses: {len(compliant)}")
print("\nFirst 5 Compliant clauses:")
for i, (idx, row) in enumerate(compliant.head(5).iterrows(), 1):
    print(f"\n{i}. [{row['app_name']}] Rule: {row.get('best_match_taxonomy_id', 'NONE')}")
    print(f"   {row['clause_text'][:180]}...")
