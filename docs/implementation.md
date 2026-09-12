# Implementation Guide

This document describes how the project is structured and built. See
[`CLAUDE.md`](../CLAUDE.md) for scope/rules and
[`architecture.md`](architecture.md) for model design.

## Repository Structure

```
DeepFake-Detection/
  CLAUDE.md
  docs/
    architecture.md
    implementation.md
    datasets.md
    ui.md
    evaluation.md
  data/
    raw/                  # untouched downloaded datasets (gitignored)
    processed/             # cached face crops, one .npy per video (gitignored)
  preprocessing/
    video.py               # VideoReader + uniform frame sampling
    face.py                # face detection, cropping, fallback chain
    dataset.py             # PyTorch Dataset + DataLoader + augmentation
    splits.py              # video-level train/val/test split logic
  models/
    baseline.py            # SpatialBranch (backbone) + ResNetBaseline
    frequency.py           # FFT log-magnitude + lightweight FrequencyCNN
    hybrid.py              # positional encoding, fusion, Transformer, head
  training/
    config.py              # Config dataclass + CLI flags (IMAGE_SIZE, D_MODEL, ...)
    train.py               # training/validation loop, checkpointing
  evaluation/
    metrics.py             # accuracy/precision/recall/F1/ROC-AUC/confusion matrix
    evaluate.py            # evaluate a checkpoint on a held-out split
  inference/
    predict.py             # end-to-end video -> REAL/FAKE + confidence
  app/
    app.py                 # Streamlit demo
  tests/
    smoke_test.py          # end-to-end executability checks
    make_synthetic_videos.py # tiny synthetic clips for offline smoke tests
  checkpoints/             # saved model weights (gitignored)
  runs/                    # per-run split manifests, history.csv, metrics (gitignored)
  requirements.txt
```

Notes on how this differs from earlier drafts of this document:

- The package layout is flat (no `src/` prefix), so every module is importable
  as `python -m training.train`, `python -m tests.smoke_test`, and so on from
  the repository root.
- The spatial branch, fusion and Transformer live together in `models/` rather
  than in one file each. They are small, and splitting a ~40-line `nn.Module`
  across four files adds indirection without adding clarity.
- Configuration is a `Config` dataclass in `training/config.py`, not a YAML
  file. Every field is exposed as a command-line flag automatically, which
  removes the need for a separate `configs/` directory and a YAML parser.
- Metrics live under `evaluation/`, next to the evaluation entry point, rather
  than under `training/`.

## Environment

```
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt
```

Two environment details worth recording, both discovered while building this:

- `opencv-python` is pinned below 5.0. The 5.x wheels no longer ship
  `cv2.CascadeClassifier` or the bundled Haar cascade XML files, which the
  default face detector uses.
- Downloading the ImageNet backbone weights fails with
  `CERTIFICATE_VERIFY_FAILED` on machines whose system trust store is
  incomplete. `models/baseline.py` points `SSL_CERT_FILE` at certifi's CA
  bundle before the download, and falls back to random initialisation with a
  loud warning if the download still fails.

## Implementation Phases

Phases follow the development order in [`CLAUDE.md`](../CLAUDE.md).
Phases 1-9 are implemented; see "Status" at the end of this document for what
that does and does not mean.

### Phase 1 — Preprocessing

- `preprocessing/video.py`: `VideoReader` opens a video with
  `cv2.VideoCapture` and exposes the frame count plus indexed frame access. A
  failed seek falls back to a sequential scan, and a frame that still cannot be
  decoded reuses the nearest decoded frame, so a partially corrupt file does
  not abort the run.
- `sample_indices(total_frames, num_frames)` in the same module returns
  evenly-spaced indices computed from the video's own length — deterministic by
  default, with optional jitter used only during training (see
  [`datasets.md`](datasets.md)). Videos shorter than `NUM_FRAMES` repeat
  indices so the sequence length is always `NUM_FRAMES`.
- `preprocessing/face.py`: `FaceExtractor` detects the largest face and returns
  a square, margin-expanded crop resized to `IMAGE_SIZE x IMAGE_SIZE`. The
  detector is configurable: `haar` (default, OpenCV's bundled cascade, no extra
  dependency), `mtcnn` (via `facenet-pytorch`, only if that optional package is
  installed) or `none`. The no-face fallback chain is documented in
  [`datasets.md`](datasets.md).

### Phase 2 — Dataset / DataLoader

- `preprocessing/dataset.py`: `VideoFaceDataset` returns
  `(NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)` ImageNet-normalised face crops plus
  a binary label (0 = REAL, 1 = FAKE). Videos are decoded one at a time inside
  `__getitem__`, so the corpus is never held in RAM.
- Face crops are cached to `data/processed/` as one uint8 `.npy` per
  (video, `NUM_FRAMES`, `IMAGE_SIZE`, detector) combination. Jittered training
  sequences are not cached, so the cache always holds deterministic sampling.
- `preprocessing/splits.py`: `discover_videos` / `load_manifest` collect videos,
  and `split_videos` splits them at the video level, stratified by class and
  seeded. `assert_no_leakage` fails loudly if a video id lands in two splits.
- `build_dataloader` wraps the dataset in a standard `DataLoader`.

### Phase 3 — Baseline

- `models/baseline.py`: `SpatialBranch` is a torchvision backbone with its
  ImageNet classifier replaced by `nn.Identity`; `ResNetBaseline` applies it per
  frame through a shared linear head and averages the per-frame logits into one
  video-level logit.

### Phase 4 — Frequency Branch

- `models/frequency.py` computes the FFT inside the model with
  `torch.fft.fft2` (the preferred option of the two this document offered): the
  representation stays on whatever device the model is on, and no second
  preprocessing cache is needed. The crop is de-normalised back to `[0, 1]`
  first so the spectrum is computed from actual pixel values rather than from
  ImageNet-standardised ones.
- `FrequencyCNN` is four conv+BN+ReLU+pool blocks followed by adaptive pooling
  and a linear layer, about 114k parameters, mapping the spectrum to `F_DIM`
  (default 128).

### Phase 5 — Hybrid Model (Fusion + Transformer)

- `models/hybrid.py` holds the whole hybrid model: per-frame concat of the
  spatial and frequency embeddings, a `Linear` projection to `D_MODEL` with a
  `LayerNorm`, sinusoidal positional encoding, `nn.TransformerEncoder`
  (`TRANSFORMER_LAYERS` layers, `N_HEADS` heads, `batch_first=True`), mean
  pooling over frames and a linear classification head.
- `forward(frames) -> logit` mirrors the baseline's interface exactly, so both
  models share the training loop, the loss and the metric code.
- `use_frequency` and `use_temporal` constructor flags select the three
  ablation configurations defined in [`evaluation.md`](evaluation.md).

### Phase 6 — Training Loop

- `training/train.py`: forward pass, `BCEWithLogitsLoss`, backward pass, AdamW
  step, per-epoch train and validation logging, for either model.
- Each run writes to `runs/<run_name>/`: the three split manifests, a
  `history.csv` loss/metric curve, the resolved config, and the final-epoch
  validation metrics. The best checkpoint by validation F1 (configurable) goes
  to `checkpoints/<run_name>_best.pt` together with the config needed to
  rebuild the model.
- Experiment tracking is plain CSV and JSON. No MLflow, no W&B.

### Phase 7 — Evaluation

- `evaluation/metrics.py`: accuracy, precision, recall, F1, ROC-AUC and a
  confusion matrix from scikit-learn, with FAKE as the positive class. ROC-AUC
  is reported as `null` when a split holds a single class, where it is
  undefined.
- `evaluation/evaluate.py` rebuilds the model from a checkpoint, regenerates
  the same seeded split, and scores a chosen split, writing JSON to
  `runs/eval/`.
- See [`evaluation.md`](evaluation.md) for the comparison protocol.

### Phase 8 — Inference

- `inference/predict.py`: given a video path and a checkpoint, runs sampling,
  face extraction and a forward pass, and returns a `Prediction` carrying the
  REAL/FAKE label, `P(FAKE)` from the model's sigmoid, the confidence in the
  predicted class, and how many frames had a detected face.

### Phase 9 — UI (optional, time permitting)

- `app/app.py`: minimal Streamlit app wrapping `inference/predict.py`, with the
  checkpoint cached via `st.cache_resource`. See [`ui.md`](ui.md) for scope.

## Testing

`tests/smoke_test.py` runs 19 executability checks covering every stage: video
opening, sampling across several video lengths, face detection and its
fallback, dataset item shape, DataLoader batching, the crop cache, split
reproducibility and leakage, the baseline, the FFT, the frequency CNN, fusion,
the Transformer, the three ablations, loss, backpropagation, a training epoch, a
validation pass, metric correctness against hand-computed values, end-to-end
inference, a checkpoint round-trip, and the Streamlit script.

`tests/make_synthetic_videos.py` generates the tiny clips the smoke tests use.
They are built from real face photographs (scikit-learn's Olivetti faces) so
the face detector is genuinely exercised, but the REAL/FAKE distinction in them
is a synthetic marker, not real manipulation. **No metric produced from these
clips means anything about detection performance.**

## Status

Implemented and executed end to end on synthetic smoke data. Not yet run on
FaceForensics++ — see [`datasets.md`](datasets.md). No real experimental
results exist yet, and [`evaluation.md`](evaluation.md) must stay empty of
numbers until they do.

## Notes

- Keep the baseline and hybrid model interchangeable at the training-loop
  level (same input shape, same loss, same metrics) so comparisons in
  [`evaluation.md`](evaluation.md) are apples-to-apples.
- Configuration is one `Config` dataclass with per-field CLI flags. Do not grow
  it into a general-purpose config framework — this is a research script, not a
  library.
