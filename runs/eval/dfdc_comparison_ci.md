| Model | Checkpoint | Split | Threshold | n | Accuracy | Precision | Recall | F1 | ROC-AUC | TN/FP/FN/TP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_5ep_best.pt | test | 0.500 | 52 | 0.327 | 0.875 | 0.171 | 0.286 [0.133, 0.444] | 0.665 [0.455, 0.851] | 10/1/34/7 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_5ep_best.pt | test | 0.099 (val) | 52 | 0.673 | 0.833 | 0.732 | 0.779 [0.676, 0.867] | 0.665 [0.455, 0.851] | 5/6/11/30 |
| spatial + frequency + mean-pool | dfdc_hybrid_notemporal_best.pt | test | 0.500 | 52 | 0.500 | 1.000 | 0.366 | 0.536 [0.360, 0.677] | 0.827 [0.698, 0.933] | 11/0/26/15 |
| spatial + frequency + mean-pool | dfdc_hybrid_notemporal_best.pt | test | 0.154 (val) | 52 | 0.750 | 0.889 | 0.780 | 0.831 [0.740, 0.909] | 0.827 [0.698, 0.933] | 7/4/9/32 |
| spatial + frequency + Transformer | dfdc_hybrid_best.pt | test | 0.500 | 52 | 0.692 | 0.857 | 0.732 | 0.789 [0.693, 0.878] | 0.612 [0.386, 0.831] | 6/5/11/30 |
| spatial + frequency + Transformer | dfdc_hybrid_best.pt | test | 0.442 (val) | 52 | 0.712 | 0.825 | 0.805 | 0.815 [0.730, 0.884] | 0.612 [0.386, 0.831] | 4/7/8/33 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_s7_best.pt | test | 0.500 | 69 | 0.464 | 0.839 | 0.448 | 0.584 [0.463, 0.694] | 0.669 [0.473, 0.862] | 6/5/32/26 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_s7_best.pt | test | 0.598 (val) | 69 | 0.464 | 0.889 | 0.414 | 0.565 [0.441, 0.681] | 0.669 [0.473, 0.862] | 8/3/34/24 |
| spatial + frequency + mean-pool | dfdc_hybrid_notemporal_s7_best.pt | test | 0.500 | 69 | 0.870 | 0.902 | 0.948 | 0.924 [0.881, 0.959] | 0.688 [0.481, 0.868] | 5/6/3/55 |
| spatial + frequency + mean-pool | dfdc_hybrid_notemporal_s7_best.pt | test | 0.618 (val) | 69 | 0.609 | 0.897 | 0.603 | 0.722 [0.622, 0.808] | 0.688 [0.481, 0.868] | 7/4/23/35 |
| spatial + frequency + Transformer | dfdc_hybrid_s7_best.pt | test | 0.500 | 69 | 0.710 | 0.896 | 0.741 | 0.811 [0.731, 0.883] | 0.708 [0.505, 0.887] | 6/5/15/43 |
| spatial + frequency + Transformer | dfdc_hybrid_s7_best.pt | test | 0.396 (val) | 69 | 0.870 | 0.915 | 0.931 | 0.923 [0.877, 0.959] | 0.708 [0.505, 0.887] | 6/5/4/54 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_s123_best.pt | test | 0.500 | 48 | 0.542 | 0.895 | 0.459 | 0.607 [0.444, 0.733] | 0.767 [0.590, 0.916] | 9/2/20/17 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_s123_best.pt | test | 0.535 (val) | 48 | 0.521 | 0.889 | 0.432 | 0.582 [0.417, 0.712] | 0.767 [0.590, 0.916] | 9/2/21/16 |
| spatial + frequency + mean-pool | dfdc_hybrid_notemporal_s123_best.pt | test | 0.500 | 48 | 0.667 | 0.839 | 0.703 | 0.765 [0.656, 0.861] | 0.720 [0.555, 0.877] | 6/5/11/26 |
| spatial + frequency + mean-pool | dfdc_hybrid_notemporal_s123_best.pt | test | 0.446 (val) | 48 | 0.729 | 0.833 | 0.811 | 0.822 [0.735, 0.897] | 0.720 [0.555, 0.877] | 5/6/7/30 |
| spatial + frequency + Transformer | dfdc_hybrid_s123_best.pt | test | 0.500 | 48 | 0.708 | 0.871 | 0.730 | 0.794 [0.687, 0.882] | 0.826 [0.693, 0.941] | 7/4/10/27 |
| spatial + frequency + Transformer | dfdc_hybrid_s123_best.pt | test | 0.562 (val) | 48 | 0.729 | 0.900 | 0.730 | 0.806 [0.698, 0.896] | 0.826 [0.693, 0.941] | 8/3/10/27 |
