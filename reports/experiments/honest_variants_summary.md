| variant | description | macro_f1 | Not Addressed F1 | Partially Compliant F1 | Compliant F1 | Δ vs v1 (wins/25) | Wilcoxon p vs v1 | C chosen | step (a) macro-F1 | step (b) macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ref_argmax | E5+8+all-rules, LogReg C=1, argmax (current final model as reported, 0.6106) | 0.6106 ± 0.0323 | 0.8498 ± 0.0229 | 0.3718 ± 0.0845 | 0.6102 ± 0.0722 | -0.0026 (10/25) | 0.4418 |  |  |  |
| v1_final | E5+8+all-rules, LogReg C=1, inner-CV thresholds (current final model) | 0.6132 ± 0.0344 | 0.8414 ± 0.0263 | 0.3865 ± 0.0817 | 0.6118 ± 0.0746 | +0.0000 (0/25) |  |  |  |  |
| v2_tfidf | TF-IDF (1-2 grams, sublinear, min_df=2) + LogReg C=1, inner-CV thresholds | 0.6148 ± 0.0364 | 0.8304 ± 0.0436 | 0.3839 ± 0.0706 | 0.6302 ± 0.0694 | +0.0016 (12/25) | 0.8119 |  |  |  |
| v3_tfidf_stacked | TF-IDF + E5+8+all-rules, LogReg C=1, inner-CV thresholds | 0.6170 ± 0.0339 | 0.8423 ± 0.0252 | 0.3917 ± 0.0817 | 0.6170 ± 0.0755 | +0.0037 (10/25) | 0.1118 |  |  |  |
| v4_C_select | E5+8+all-rules, LogReg C in {0.1,0.3,1,3} by inner CV, inner-CV thresholds | 0.6250 ± 0.0346 | 0.8385 ± 0.0291 | 0.4087 ± 0.0846 | 0.6279 ± 0.0702 | +0.0118 (17/25) | 0.0119 | 0.1×19, 0.3×4, 1.0×1, 3.0×1 |  |  |
| v5_two_step | Two-step on v4_C_select: (a) NA vs Addressed, (b) Partial vs Compliant (truly-Addressed rows), binary inner-CV thresholds | 0.6370 ± 0.0387 | 0.8598 ± 0.0257 | 0.4075 ± 0.0850 | 0.6439 ± 0.0742 | +0.0238 (19/25) | 0.0020 | 0.1×35, 0.3×10, 1.0×5 | 0.7658 ± 0.0359 | 0.7401 ± 0.0569 |
