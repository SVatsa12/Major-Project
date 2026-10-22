import pandas as pd

df = pd.read_csv('summary.csv')
fixed = df[df['tag'].str.startswith('fixed_')].copy()
fixed['f1'] = fixed['macro_f1_mean'].astype(float)
fixed = fixed.sort_values('f1', ascending=False)

print('\n' + '='*140)
print('FINAL RESULTS: Fixed Folds + Label-Independent Features')
print('='*140)
print(f"{'Variant':<20} {'Rule Source':<20} {'Grouped':<10} {'Acc':<18} {'Macro-F1':<18} {'Macro-Prec':<18} {'C_F1':<12} {'PC_F1':<12}")
print('-'*140)

for _, row in fixed.iterrows():
    tag = row['tag'].replace('fixed_', '')
    
    if 'B_e5_ref' in tag:
        rule_src = 'LLM output (leak)'
    elif 'B2a' in tag:
        rule_src = 'Label-independent'
    elif 'baseline' in tag or 'A_' in tag:
        rule_src = 'N/A'
    elif 'C_' in tag:
        rule_src = 'B2a (independent)'
    else:
        rule_src = 'Unknown'
    
    grouped = 'YES'
    acc_str = f"{row['accuracy_mean']} ± {row['accuracy_std']}"
    f1_str = f"{row['macro_f1_mean']} ± {row['macro_f1_std']}"
    
    if 'macro_precision_mean' in row and pd.notna(row['macro_precision_mean']):
        prec_str = f"{row['macro_precision_mean']} ± {row['macro_precision_std']}"
    else:
        prec_str = 'N/A'
    
    print(f"{tag:<20} {rule_src:<20} {grouped:<10} {acc_str:<18} {f1_str:<18} {prec_str:<18} {row['C_f1']:<12} {row['PC_f1']:<12}")

print('\n' + '='*140)
print('\nKEY FINDINGS:')
print('1. With fixed folds, variant B (LLM rule, leaky) dropped from 0.8050 to 0.7923 (-1.3%)')
print('2. Variant B2a (label-independent rule): 0.7663 - TRUE performance without leakage')
print('3. B2a improvement over baseline: +1.9% (0.7473 → 0.7663)')
print('4. B leaky improvement over baseline: +4.5% (0.7473 → 0.7923) - INFLATED')
print('5. Two-stage (C) slightly better than B2a: 0.7675 vs 0.7663')
print('6. E5 encoder swap (A): 0.7703 - best single change')
print('\nCONCLUSION: Real pairing improvement is ~2%, not the 5-8% from leaky variants')
