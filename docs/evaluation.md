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

## Current Status

**No experimental results exist yet.** The training, validation, evaluation and
inference code has been executed end to end, but only on the tiny synthetic
smoke-test clips described in [`datasets.md`](datasets.md). Those clips are not
deepfakes, the splits they produce hold two videos each, and the numbers they
generate are executability artefacts with no research meaning.

The tables in this document stay empty until the models are trained on
FaceForensics++ and evaluated on a real held-out split.
