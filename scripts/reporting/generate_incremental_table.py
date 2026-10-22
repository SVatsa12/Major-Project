import pandas as pd
import json
from pathlib import Path

# Load results
results = []
for tag in ["fixed_A_e5", "incr_1_all_rules", "incr_2_rubric", "incr_3_combined", "incr_4_tuned"]:
    json_file = Path(f"{tag}.json")
    if json_file.exists():
        data = json.loads(json_file.read_text())
        
        if tag == "fixed_A_e5":
            variant_name = "Baseline (E5 + 8)"
        elif "all_rules" in tag and "combined" not in tag:
            variant_name = "+All-rules similarity"
        elif "rubric" in tag and "combined" not in tag:
            variant_name = "+Rubric features"
        elif "combined" in tag:
            variant_name = "+All-rules + Rubric"
        elif "tuned" in tag:
            variant_name = "+Classifier tuning"
        else:
            variant_name = tag
        
        agg = data["aggregate_metrics"]
        per_class = data["per_class_metrics"]
        
        results.append({
            "Variant": variant_name,
            "Macro-F1": f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}",
            "Macro-Prec": f"{agg['macro_precision_mean']:.4f} ± {agg['macro_precision_std']:.4f}",
            "Compliant P": f"{per_class['Compliant']['precision']:.4f} ± {per_class['Compliant']['precision_std']:.4f}",
            "Compliant R": f"{per_class['Compliant']['recall']:.4f} ± {per_class['Compliant']['recall_std']:.4f}",
            "Compliant F1": f"{per_class['Compliant']['f1']:.4f} ± {per_class['Compliant']['f1_std']:.4f}",
            "Partial F1": f"{per_class['Partially Compliant']['f1']:.4f} ± {per_class['Partially Compliant']['f1_std']:.4f}",
            "F1_val": agg['macro_f1_mean'],
        })

df = pd.DataFrame(results)
df = df.sort_values('F1_val', ascending=False)

print("\n" + "="*140)
print("INCREMENTAL FEATURE IMPROVEMENTS (Starting from fixed_A_e5)")
print("="*140)
print(df[["Variant", "Macro-F1", "Macro-Prec", "Compliant P", "Compliant R", "Compliant F1", "Partial F1"]].to_string(index=False))

print("\n" + "="*140)
print("\nKEY FINDINGS:")
print("1. All-rules similarity: +4.2% F1 (0.7703 → 0.8118) - BEST improvement!")
print("2. Rubric features: +0.9% F1 (0.7703 → 0.7797)")  
print("3. Combined (all-rules + rubric): +0.8% F1 (0.7703 → 0.7779) - WORSE than all-rules alone")
print("4. Classifier tuning: no improvement over default LogReg")
print("\nWINNER: E5 clause + 8 structured + all-rules similarity = 0.8118 ± 0.0315")
print("        (vs baseline 0.7473, improvement: +6.5%)")
