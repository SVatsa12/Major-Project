# Phase Execution Commands - Copy & Paste Ready

## Phase 0: Branch Creation

```bash
git checkout -b experiment/f1-opt-80
```

---

## Phase 1: Decision Threshold Sweep

### Execute:
```bash
python tune_threshold.py
```

### After execution, report back:
1. Best validation tau_compliant and tau_partial
2. Test Macro-F1 score
3. Did we reach >= 0.8000?

---

## Phase 2: Regularization & Class Weight Calibration

### Execute:
```bash
python train_calibrated.py
```

### After execution, report back:
1. Best C value
2. Best tau_compliant and tau_partial
3. Test Macro-F1 score
4. Did we reach >= 0.8000?

---

## Phase 3: Stratified 5-Fold Ensemble

### Execute:
```bash
python train_5fold_ensemble.py
```

### After execution, report back:
1. Per-fold validation F1 scores
2. OOF calibration thresholds (tau_ens_c, tau_ens_p)
3. Individual fold test F1s
4. Ensemble test F1 score
5. Did we reach >= 0.8000?

---

## Phase 4: Hybrid Domain Embeddings (Fallback)

### Execute:
```bash
python train_hybrid_embeddings.py
```

### After execution, report back:
1. Validation F1 with hybrid embeddings
2. Test Macro-F1 score
3. Per-class metrics
4. Did we reach >= 0.8000?

---

## Commit After Each Phase

```bash
git add *.py *.md reports/*.json models/*.joblib
git commit -m "Phase [N]: [Method Name] - Test F1 = [score]"
```

---

## Final Commit (When Target Achieved)

```bash
git add -A
git commit -m "F1 Optimization Complete: Test F1 = [final_score] via [best_method]"
git push -u origin experiment/f1-opt-80
```

---

## Quick Metrics Check

### View latest results:
```bash
cat reports/threshold_sweep_results.json | grep "test_macro_f1"
cat reports/calibrated_results.json | grep "test_macro_f1"
cat reports/ensemble_5fold_results.json | grep "test_macro_f1"
cat reports/hybrid_embeddings_results.json | grep "test_macro_f1"
```

### Compare all phases:
```bash
echo "=== Phase 1 ===" ; python -c "import json; print('Test F1:', json.load(open('reports/threshold_sweep_results.json'))['optimized']['test_macro_f1'])" 2>/dev/null || echo "Not run yet"
echo "=== Phase 2 ===" ; python -c "import json; print('Test F1:', json.load(open('reports/calibrated_results.json'))['test_results']['macro_f1'])" 2>/dev/null || echo "Not run yet"
echo "=== Phase 3 ===" ; python -c "import json; print('Test F1:', json.load(open('reports/ensemble_5fold_results.json'))['ensemble_test_results']['macro_f1'])" 2>/dev/null || echo "Not run yet"
echo "=== Phase 4 ===" ; python -c "import json; print('Test F1:', json.load(open('reports/hybrid_embeddings_results.json'))['optimized']['test_macro_f1'])" 2>/dev/null || echo "Not run yet"
```
