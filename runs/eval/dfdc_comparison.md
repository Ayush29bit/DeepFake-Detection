| Model | Checkpoint | Split | Threshold | n | Accuracy | Precision | Recall | F1 | ROC-AUC | TN/FP/FN/TP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_5ep_best.pt | test | 0.500 | 52 | 0.327 | 0.875 | 0.171 | 0.286 | 0.665 | 10/1/34/7 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_5ep_best.pt | test | 0.099 (val) | 52 | 0.673 | 0.833 | 0.732 | 0.779 | 0.665 | 5/6/11/30 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_best.pt | test | 0.500 | 52 | 0.442 | 1.000 | 0.293 | 0.453 | 0.752 | 11/0/29/12 |
| ResNet18 baseline (frame-logit mean) | dfdc_baseline_best.pt | test | 0.048 (val) | 52 | 0.827 | 0.864 | 0.927 | 0.894 | 0.752 | 5/6/3/38 |
| spatial + frequency + Transformer | dfdc_hybrid_best.pt | test | 0.500 | 52 | 0.692 | 0.857 | 0.732 | 0.789 | 0.612 | 6/5/11/30 |
| spatial + frequency + Transformer | dfdc_hybrid_best.pt | test | 0.442 (val) | 52 | 0.712 | 0.825 | 0.805 | 0.815 | 0.612 | 4/7/8/33 |
