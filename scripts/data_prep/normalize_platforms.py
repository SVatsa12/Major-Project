"""
src/evaluation/normalize_platforms.py

Normalizes platform names in corpus/labeled_clauses_bootstrap.csv
to the 6 canonical platforms and updates related matrices.
"""

from pathlib import Path
import pandas as pd

CORPUS_PATH = Path("corpus/labeled_clauses_bootstrap.csv")

df = pd.read_csv(CORPUS_PATH)

print("[INFO] Pre-normalization platform distribution:")
print(df["app_name"].value_counts())
print(f"\nTotal unique platforms before: {len(df['app_name'].unique())}")

PLATFORM_MAP = {
    # YouTube casing
    "Youtube": "YouTube",
    "youtube": "YouTube",
    "YOUTUBE": "YouTube",
    
    # Meta consolidations
    "Meta/Instagram": "Meta",
    "Instagram": "Meta",
    "Meta/Facebook": "Meta",
    "meta": "Meta",
    "facebook": "Meta",
    "Facebook": "Meta",
    
    # Other platforms casing safety
    "whatsapp": "WhatsApp",
    "WHATSAPP": "WhatsApp",
    "google": "Google",
    "Google": "Google",
    "GOOGLE": "Google",
    "snapchat": "Snapchat",
    "SNAPCHAT": "Snapchat",
    "telegram": "Telegram",
    "TELEGRAM": "Telegram",
}

# Apply canonical mapping
df["app_name"] = df["app_name"].replace(PLATFORM_MAP)

# Save back to CSV
df.to_csv(CORPUS_PATH, index=False)

print("\n[SUCCESS] Post-normalization platform distribution (must be exactly 6):")
print(df["app_name"].value_counts())
print(f"\nTotal unique platforms after: {len(df['app_name'].unique())}")
print(f"Unique platforms: {sorted(df['app_name'].unique())}")

assert len(df["app_name"].unique()) == 6, f"Expected 6 platforms, found {len(df['app_name'].unique())}"
print("\n✅ Platform normalization COMPLETE. Corpus ready for re-evaluation.")
