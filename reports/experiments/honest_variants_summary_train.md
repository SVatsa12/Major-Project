| variant | description | macro_f1 | Not Addressed F1 | Partially Compliant F1 | Compliant F1 | Δ vs v1 (wins/15) | Wilcoxon p vs v1 | step (a) macro-F1 | step (b) macro-F1 | C chosen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_final | E5+8+all-rules, LogReg C=1, inner-CV thresholds (current final model) | 0.6134 ± 0.0286 | 0.8390 ± 0.0276 | 0.3941 ± 0.0548 | 0.6070 ± 0.0625 | +0.0000 (0/15) |  |  |  |  |
| v5_two_step | Two-step on v4_C_select: (a) NA vs Addressed, (b) Partial vs Compliant (truly-Addressed rows), binary inner-CV thresholds | 0.6263 ± 0.0339 | 0.8612 ± 0.0195 | 0.4137 ± 0.0891 | 0.6040 ± 0.0816 | +0.0129 (9/15) | 0.2524 | 0.7651 ± 0.0282 | 0.7200 ± 0.0580 | 0.1×17, 0.3×8, 3.0×5 |
