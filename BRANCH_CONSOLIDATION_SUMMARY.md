# Experimental Branch Consolidation Summary

**Date Consolidated**: 2026-10-14  
**Status**: ✅ Complete — All rejected experimental branches consolidated to main

---

## Branch Consolidation Record

### Phase A: Threshold Objective Variants

| Field | Value |
|-------|-------|
| **Branch Name** | `experiment/phase-a-thresholds` |
| **Final Commit Hash** | `fa9a4001d4c07bf06c6d74e78a17120f8ce90413` |
| **Status** | ❌ REJECTED (F1 gain +0.0018 < 0.01 threshold) |
| **Experiment Outcome** | Mean Macro-F1 0.7730 vs baseline 0.7712 |
| **Logs Consolidated** | `PROJECT_REPORT.md` (End-to-End Technical Report) |
| **Consolidated to Main** | ✅ YES (commit fa9a400, now reachable from main) |

**Key Experiment Records**:
- Threshold objective variants tested for precision-recall trade-offs
- Calibration analysis for decision boundaries
- Full pipeline documentation

---

### Phase B: Feature Union Experiment

| Field | Value |
|-------|-------|
| **Branch Name** | `phase-b-feature-union` |
| **Final Commit Hash** | `968c641e525fcf3613c0e0fa93f672e7558eaa4f` |
| **Status** | ❌ REJECTED (F1 gain +0.0020 < 0.01 threshold) |
| **Experiment Outcome** | Mean Macro-F1 0.7732 vs baseline 0.7712 (insufficient gain) |
| **Logs Consolidated** | `EXPERIMENT_LOG.md` (Feature Union Experiment Log) |
| **Consolidated to Main** | ✅ YES (commit 31d99ff) |

**Key Experiment Records**:
- 7 new count-based features added to baseline 8-dim → 15-dim union
- Marginal F1 improvement with improved stability (std reduced 16.83%)
- Per-class metrics all stable or improved
- Threshold calibration results documented

---

### Phase E: MPNet Encoder Experiment

| Field | Value |
|-------|-------|
| **Branch Name** | `phase-e-mpnet-encoder` |
| **Final Commit Hash** | `d93f3195c2a13eaf86406ae03b5eb684560d8a2b` |
| **Status** | ❌ REJECTED (F1 regression -0.0359; all criteria failed) |
| **Experiment Outcome** | Mean Macro-F1 0.7353 vs baseline 0.7712 (F1 drop -0.0359, -4.66%) |
| **Logs Consolidated** | `PHASE_E_EXPERIMENT_LOG.md` (MPNet Encoder Experiment Log) |
| **Consolidated to Main** | ✅ YES (commit 86f9089) |

**Key Experiment Records**:
- Encoder swap: all-MiniLM-L6-v2 (384-d) → all-mpnet-base-v2 (768-d)
- Complete performance degradation across all metrics
- Per-class instability (Partially Compliant F1 drop -0.0411 > 0.02 tolerance)
- Documented failure reasons and technical analysis

---

## Main Branch Commit History (Post-Consolidation)

```
86f9089 (HEAD -> main)
Merge phase-e-mpnet-encoder: MPNet Encoder experiment (REJECTED) + PHASE_E_EXPERIMENT_LOG.md

31d99ff
Merge phase-b-feature-union: Feature Union experiment (REJECTED) + EXPERIMENT_LOG.md

fa9a400 (experiment/phase-a-thresholds, threshold-evaluation)
Phase A: threshold objective variants experiment - mean macro-F1 0.7730 vs baseline 0.7712

1789082 (origin/main, origin/HEAD)
fixing the training pipeline
```

---

## Verification: All Logs Accessible on Main

✅ **PROJECT_REPORT.md** (Phase A) — End-to-End Technical Report  
✅ **EXPERIMENT_LOG.md** (Phase B) — Feature Union Experiment Log  
✅ **PHASE_E_EXPERIMENT_LOG.md** (Phase E) — MPNet Encoder Experiment Log  

All experiment records are now reachable from `main` and permanently archived.

---

## Next Steps: Branch Deletion

**Do NOT execute until this summary is verified and committed.**

The following branches can now be safely deleted:
```bash
git branch -d experiment/phase-a-thresholds
git branch -d phase-b-feature-union
git branch -d phase-e-mpnet-encoder

git push origin --delete experiment/phase-a-thresholds
git push origin --delete phase-b-feature-union
git push origin --delete phase-e-mpnet-encoder
```

All experiment data is preserved in main's history. No records will be lost.

---

**Consolidation Status**: ✅ COMPLETE  
**Records Lost**: 0  
**Records Preserved**: 3 (PROJECT_REPORT.md, EXPERIMENT_LOG.md, PHASE_E_EXPERIMENT_LOG.md)  
