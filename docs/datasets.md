# Datasets

This document describes which datasets are used, and how data is split and
preprocessed. See [`CLAUDE.md`](../CLAUDE.md) for overall scope.

## FaceForensics++ (Primary)

- Primary dataset for training and evaluation.
- Contains real videos plus multiple manipulation methods (e.g.
  Deepfakes, Face2Face, FaceSwap, NeuralTextures). For this milestone, all
  manipulation methods are pooled into a single FAKE class against REAL —
  per-method breakdown is not required unless requested later.
- Use the standard FaceForensics++ compression setting available locally
  (e.g. c23) — exact compression level is configurable, not finalized here.

## Celeb-DF (External/Generalization Evaluation)

- Used, if feasible, purely as a cross-dataset generalization test: train
  on FaceForensics++, evaluate (no fine-tuning) on Celeb-DF.
- Not required for the 40% milestone's core training loop — only relevant
  once baseline/hybrid models are already trained and evaluation
  infrastructure (`docs/evaluation.md`) is in place.

## DFDC (Optional)

- Optional and explicitly **not required** for the current milestone.
- May be considered later for additional generalization testing, only after
  FaceForensics++ and (optionally) Celeb-DF work is complete.

## Train/Validation/Test Strategy

- Splits are created at the **video level**, not the frame level.
- All frames sampled from a given video must fall entirely within one split
  (train, val, or test) — never split across sets. This prevents the model
  from memorizing identity/scene-specific cues from a video it has partially
  seen during training and being evaluated on frames from the same video.
- Where FaceForensics++ provides official train/val/test video ID lists,
  use those as the default split. Otherwise, split video IDs (not frames)
  with a fixed random seed for reproducibility, roughly 70/15/15 or
  matching whatever official split is available.
- Class balance (REAL vs FAKE) should be checked per split; FaceForensics++
  has more FAKE videos than REAL (multiple manipulation methods per real
  video), so consider balancing at the sampling stage (e.g. weighted
  sampling or capping fakes per real video) if the imbalance is severe.

## Frame Sampling

- `NUM_FRAMES` (default 8) frames are sampled per video, evenly spaced
  across the video's duration, to give the temporal model a spread of
  frames rather than a short burst.
- Sampling is deterministic at evaluation/inference time (same indices
  every run). During training, small random jitter around the evenly-spaced
  indices is an acceptable augmentation to reduce overfitting to exact
  frame positions.

## Face Preprocessing

- Each sampled frame is passed through a face detector to locate and crop
  the primary face (the largest detected box). The detector is configurable:
  `haar` (default), `mtcnn` or `none` — see
  [`implementation.md`](implementation.md). It is not finalized to one library.
- The detected box is expanded by a margin (default 25%), squared off and
  clamped to the frame, then resized to `IMAGE_SIZE x IMAGE_SIZE`
  (default 224x224).

### No-face fallback

Frames are never dropped, so a sequence is always exactly `NUM_FRAMES` long and
one failed frame cannot break the video. When a frame has no detected face, the
crop box is chosen in this order:

1. The box detected in that frame.
2. The box from the nearest **preceding** frame of the same video that had a
   detection.
3. The box from the nearest **following** frame of the same video that had a
   detection (this covers videos whose opening frames fail).
4. A centre crop of the frame, squared to its shorter side.

This chain is implemented once, in `FaceExtractor.extract_sequence`, and is
used identically by training, evaluation and inference, so the two can never
drift apart.

- Face crops are cached to `data/processed/` after first extraction to
  avoid redundant detection cost on repeated training runs. The cache key
  includes the video id, `NUM_FRAMES`, `IMAGE_SIZE` and the detector name.
  Jittered training sequences are not cached.

## Synthetic Smoke-Test Clips

`tests/make_synthetic_videos.py` writes a handful of tiny clips used only to
prove the pipeline executes without a real dataset present. They animate real
face photographs (scikit-learn's Olivetti faces, so the face detector is
actually exercised), and mark the FAKE class with a fine checkerboard overlay.

That overlay is an artificial high-frequency marker, **not** real manipulation,
and these clips are not a deepfake dataset. Any metric computed on them is a
smoke-test artefact and must never be reported as a result.

## Augmentation (Modest)

Kept intentionally light — this is a detection task where aggressive
augmentation can wash out the frequency-domain artifacts the model relies
on:

- Horizontal flip
- Small random rotation (a few degrees)
- Mild color jitter (brightness/contrast)
- Avoid heavy compression/blur/noise augmentation that could destroy the
  high-frequency artifacts the frequency branch depends on, unless
  specifically testing robustness later.
- Augmentation is applied to the spatial input **before** the frequency
  branch's FFT is computed, so both branches see a consistent augmented
  frame; do not augment the FFT output directly.
