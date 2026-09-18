from pathlib import Path

from src.labeling.label_policy_clauses import (
    extract_json_array,
    load_corpus,
    load_taxonomy,
    validate_results,
)

taxonomy, _ = load_taxonomy(Path("corpus/dpdp_taxonomy_final.json"))
taxonomy_ids = {item["taxonomy_id"] for item in taxonomy}
corpus, _ = load_corpus(Path("datasets/SocialMediaPolicies/social_media_clauses_flagged.csv"))
batch = corpus[corpus["clause_id"] == "S000055"].to_dict(orient="records")
recovered = validate_results(
    extract_json_array('[{"clause_id":"S000055","matches":[]}]'),
    batch,
    taxonomy_ids,
)
print(recovered)
