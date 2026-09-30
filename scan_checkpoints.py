import pandas as pd
from transformers import pipeline
from sklearn.metrics import f1_score

df = pd.read_csv('datasets/splits/test.csv')
df = df[df['verdict'].isin(['Not Addressed', 'Partially Compliant', 'Compliant'])]
texts = df['clause_text'].tolist()
true_labels = df['verdict'].tolist()

checkpoints = [
    'checkpoint-138',
    'checkpoint-207',
    'checkpoint-276',
    'checkpoint-345',
    'checkpoint-414',
    'checkpoint-483'
]

print("Scanning checkpoints on Test Set...")
for ckpt in checkpoints:
    model_path = f"models/inlegalbert_checkpoints/{ckpt}"
    pipe = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=-1,
        truncation=True,
        max_length=384
    )
    preds = [res['label'] for res in pipe(texts, batch_size=16)]
    score = f1_score(true_labels, preds, average="macro", zero_division=0)
    print(f"{ckpt}: Macro-F1 = {score:.4f}")
