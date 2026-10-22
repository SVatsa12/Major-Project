import pandas as pd

# Baseline from baseline_v1_grouped
BASELINE_F1 = 0.7473
BASELINE_STD = 0.0354

df = pd.read_csv('summary.csv')

# Filter to variant experiments and baseline
variants = df[df['tag'].str.contains('variant_|baseline_v1_grouped')].copy()

# Convert to numeric
variants['f1_num'] = variants['macro_f1_mean'].astype(float)
variants['f1_std_num'] = variants['macro_f1_std'].astype(float)
variants['acc_num'] = variants['accuracy_mean'].astype(float)
variants['acc_std_num'] = variants['accuracy_std'].astype(float)

# Handle missing macro_precision (older baseline runs)
if 'macro_precision_mean' in variants.columns:
    variants['prec_num'] = variants['macro_precision_mean'].astype(float)
    variants['prec_std_num'] = variants['macro_precision_std'].astype(float)
else:
    variants['prec_num'] = None
    variants['prec_std_num'] = None

# Sort by macro_f1 descending
variants = variants.sort_values('f1_num', ascending=False)

# Compute beats baseline
variants['beats_baseline'] = variants['f1_num'] > (BASELINE_F1 + BASELINE_STD)

print('\n' + '='*130)
print('VARIANT COMPARISON TABLE (Sorted by Macro-F1)')
print('='*130)
print(f"{'Variant':<30} {'Accuracy':<20} {'Macro-F1':<20} {'Compliant F1':<15} {'Partial F1':<15} {'Beats Baseline'}")
print('-'*130)

for _, row in variants.iterrows():
    acc_str = f"{row['acc_num']:.4f} ± {row['acc_std_num']:.4f}"
    f1_str = f"{row['f1_num']:.4f} ± {row['f1_std_num']:.4f}"
    c_f1 = row['C_f1']
    pc_f1 = row['PC_f1']
    beats = '***' if row['beats_baseline'] else ''
    
    print(f"{row['tag']:<30} {acc_str:<20} {f1_str:<20} {c_f1:<15} {pc_f1:<15} {beats}")

print('\n*** = Beats baseline_v1_grouped (0.7473) by more than 1 std (0.0354)')
print(f'Threshold for beating baseline: {BASELINE_F1 + BASELINE_STD:.4f}')

