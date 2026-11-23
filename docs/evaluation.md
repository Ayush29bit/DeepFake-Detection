# Evaluation

This document describes how models are evaluated and compared. See
[`architecture.md`](architecture.md) for model definitions and
[`datasets.md`](datasets.md) for split strategy.

**No fabricated results.** This document defines the evaluation protocol
and metrics only. Actual numbers are filled in only after real experiments
are run, and must be reported alongside the exact model/checkpoint/split
used to produce them.

## Models Compared

1. **Baseline** — ResNet18-only, frame-level, averaged over `NUM_FRAMES`
   frames (see [`architecture.md`](architecture.md#baseline-resnet18-only)).
2. **Proposed (Hybrid)** — spatial + frequency fusion + Transformer temporal
   modeling (see [`architecture.md`](architecture.md)).

Both are trained and evaluated on identical data splits so results are
directly comparable.

## Metrics

Computed on the held-out validation/test split (video-level predictions,
i.e. one prediction per video, not per frame):

- **Accuracy**
- **Precision**
- **Recall**
- **F1 score**
- **ROC-AUC** (if feasible given the label distribution and time)

All metrics computed with scikit-learn (`sklearn.metrics`) from model
outputs (predicted probability + threshold at 0.5 for hard labels), by
`evaluation/metrics.py`. FAKE (label 1) is the positive class. A confusion
matrix is reported alongside. ROC-AUC comes back as `null` when a split holds
a single class, where it is undefined — it is never silently reported as 0.

`evaluation/evaluate.py` scores a checkpoint on a chosen split and writes the
metrics to JSON under `runs/eval/`, recording the checkpoint, split, model and
seed alongside the numbers so a result can always be traced back to the run
that produced it.

## Ablation Experiments

To isolate the contribution of each component of the hybrid model, three
configurations are evaluated on the same data split:

1. **Spatial only** — ResNet18 spatial embeddings, mean-pooled across
   frames (no frequency branch, no Transformer), fed to the classification
   head. This isolates the value of the spatial branch alone (close to, but
   not identical to, the baseline — see note below).
2. **Spatial + Frequency** — spatial and frequency embeddings fused per
   frame (as in the full model), then mean-pooled across frames (no
   Transformer) before the classification head. Isolates the contribution
   of the frequency branch.
3. **Spatial + Frequency + Temporal** — the full proposed hybrid model,
   including the Transformer encoder for temporal modeling. Isolates the
   contribution of temporal modeling on top of the fused per-frame features.

Note: Ablation (1) differs from the Baseline model in that it reuses the
hybrid model's mean-pooling classification path rather than per-frame
logit averaging; both are valid "spatial only" references and should be
reported separately if both are run.

## Cross-Dataset Evaluation (If Feasible)

- Train on FaceForensics++ only.
- Evaluate (no fine-tuning, no additional training) on Celeb-DF.
- Report the same metrics (accuracy/precision/recall/F1/ROC-AUC) on the
  cross-dataset test to measure generalization beyond the training
  distribution.
- This step depends on Celeb-DF being available and preprocessed (see
  [`datasets.md`](datasets.md)); it is not required for the 40% milestone's
  definition of done.

## Reporting Format

When results are produced, report them as a table with, at minimum:
model name, dataset(s), split, accuracy, precision, recall, F1, ROC-AUC
(if computed), and the checkpoint/commit used to generate them. Do not
present placeholder or illustrative numbers as if they were real results.

## Threshold Selection

The default decision threshold is 0.5. Because the training loss is
class-weighted (see [`datasets.md`](datasets.md)), 0.5 is not necessarily a good
operating point, so `evaluate.py --val-threshold` picks the threshold that
maximises Youden's J (TPR - FPR) **on the validation split** and applies it
unchanged to test. Test data is never used to choose it. Both operating points
are reported. ROC-AUC is threshold-free and is the primary comparison metric.
The inference CLI and UI still use the checkpoint threshold (0.5).

## Current Status — first real results (DFDC training sample)

Dataset: DFDC training sample, 400 videos (77 REAL / 323 FAKE), grouped split by
source video, seed 42. Test split: 52 videos (11 REAL / 41 FAKE). CPU only
(4 cores), 8 frames x 224 px, AdamW lr 1e-4, wd 1e-4, `--train-jitter false`,
augmentation on, best checkpoint by validation ROC-AUC.

| Model | Epochs run (best) | Batch | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | TN/FP/FN/TP |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ResNet18 baseline | 2 of 5, stopped by OOM (2) | 4 | 0.500 | 0.442 | 1.000 | 0.293 | 0.453 | 0.752 | 11/0/29/12 |
| ResNet18 baseline | 2 of 5, stopped by OOM (2) | 4 | 0.048 (val) | 0.827 | 0.864 | 0.927 | 0.894 | 0.752 | 5/6/3/38 |
| ResNet18 baseline (rerun) | 5 of 5 (1) | 2 | 0.500 | 0.327 | 0.875 | 0.171 | 0.286 | 0.665 | 10/1/34/7 |
| ResNet18 baseline (rerun) | 5 of 5 (1) | 2 | 0.099 (val) | 0.673 | 0.833 | 0.732 | 0.779 | 0.665 | 5/6/11/30 |
| Spatial + frequency + Transformer | 5 (5) | 2 | 0.500 | 0.692 | 0.857 | 0.732 | 0.789 | 0.612 | 6/5/11/30 |
| Spatial + frequency + Transformer | 5 (5) | 2 | 0.442 (val) | 0.712 | 0.825 | 0.805 | 0.815 | 0.612 | 4/7/8/33 |

Sources: `runs/eval/dfdc_*_best_test*.json` (per-video probabilities included),
rendered by `python -m evaluation.compare`; training logs and curves in
`runs/dfdc_baseline/`, `runs/dfdc_baseline_5ep/`, `runs/dfdc_hybrid/`
(`history.csv`). Validation ROC-AUC of the selected checkpoints: baseline 0.893,
baseline rerun 0.907, hybrid 0.861.

Interpretation, kept deliberately cautious:

- On this data the hybrid model does **not** outperform the spatial baseline in
  test ROC-AUC (hybrid 0.612; baseline 0.752 and 0.665 across two runs). The
  hypothesis in the roadmap is not supported by this experiment; it is not
  refuted either.
- The two baseline runs share data, split and seed yet differ by 0.09 test
  ROC-AUC, so run-to-run variance is as large as the baseline-vs-hybrid gap.
  No difference between the models can be claimed until several seeds are run.
- The test split holds only 11 REAL videos, so every metric moves by about 9
  percentage points per misclassified real video. The val/test ROC-AUC gap
  (0.86 -> 0.61 for the hybrid) is consistent with that variance and with
  overfitting on ~300 training videos.
- The first baseline run stopped after 2 epochs (out of memory on the host, not
  a model error) at batch size 4. The rerun used the same budget as the hybrid
  (5 epochs, batch size 2); its best validation epoch was epoch 1, after which
  validation loss became unstable, a sign of overfitting on ~300 videos.
- At 0.5 the baseline predicts REAL far too often; the class-weighted loss
  shifts its probabilities down. The validation-chosen threshold fixes most of
  that, which is why both operating points are shown.
- No ablation (spatial+frequency without Transformer), repeated seeds, or
  cross-dataset (Celeb-DF / FaceForensics++) results exist yet.

Earlier smoke-test numbers on synthetic clips are executability artefacts and
are not results.
