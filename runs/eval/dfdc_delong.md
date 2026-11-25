Paired DeLong test of test ROC-AUC (same seed, same test videos). Model A vs model B.

| A | B | n | AUC A | AUC B | z | p |
|---|---|---:|---:|---:|---:|---:|
| dfdc_hybrid_best_test.json | dfdc_baseline_5ep_best_test.json | 52 | 0.612 | 0.665 | -0.34 | 0.736 |
| dfdc_hybrid_notemporal_best_test.json | dfdc_baseline_5ep_best_test.json | 52 | 0.827 | 0.665 | +1.83 | 0.067 |
| dfdc_hybrid_best_test.json | dfdc_hybrid_notemporal_best_test.json | 52 | 0.612 | 0.827 | -1.83 | 0.067 |
| dfdc_hybrid_s7_best_test.json | dfdc_baseline_s7_best_test.json | 69 | 0.708 | 0.669 | +0.51 | 0.611 |
| dfdc_hybrid_notemporal_s7_best_test.json | dfdc_baseline_s7_best_test.json | 69 | 0.688 | 0.669 | +0.15 | 0.879 |
| dfdc_hybrid_s7_best_test.json | dfdc_hybrid_notemporal_s7_best_test.json | 69 | 0.708 | 0.688 | +0.27 | 0.789 |
| dfdc_hybrid_s123_best_test.json | dfdc_baseline_s123_best_test.json | 48 | 0.826 | 0.767 | +1.15 | 0.250 |
| dfdc_hybrid_notemporal_s123_best_test.json | dfdc_baseline_s123_best_test.json | 48 | 0.720 | 0.767 | -0.42 | 0.673 |
| dfdc_hybrid_s123_best_test.json | dfdc_hybrid_notemporal_s123_best_test.json | 48 | 0.826 | 0.720 | +1.38 | 0.169 |
