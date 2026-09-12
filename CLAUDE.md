# CLAUDE.md

## Project Objective

Build a research-oriented video deepfake detector that classifies a face
video as **REAL** or **FAKE**, using a hybrid architecture that fuses
spatial, frequency, and temporal features via a Transformer encoder.

## Source-of-Truth Rule

The files in this repository root and `docs/` are the source of truth for
scope, architecture, and implementation order:

- `CLAUDE.md` (this file)
- `docs/architecture.md`
- `docs/implementation.md`
- `docs/datasets.md`
- `docs/ui.md`
- `docs/evaluation.md`

Any code written for this project must be consistent with these documents.
If a change in approach is needed, update the relevant document first, then
write code. Do not silently diverge from what is documented here.

## Architecture Summary

```
Video
  -> Frame Sampling
  -> Face Detection/Cropping
  -> Spatial Feature Extraction   (pretrained ResNet18)
  -> Frequency Feature Extraction (2D FFT -> log magnitude -> lightweight CNN)
  -> Per-frame Feature Fusion     (concat + linear projection)
  -> Transformer Encoder          (temporal modeling across ~8 frames)
  -> Classification Head          (linear -> BCEWithLogitsLoss)
  -> REAL / FAKE
```

A separate **ResNet18-only baseline** (frame-level, no frequency branch, no
Transformer) is implemented alongside the hybrid model for comparison.

See `docs/architecture.md` for full detail.

## Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- scikit-learn
- Pillow
- Streamlit (minimal demo UI only)

No other frameworks are required for the current milestone.

## Implementation Target

Only **approximately 40%** of the full project is being implemented at this
stage (the "current milestone"). This milestone is scoped to a working,
demonstrable pipeline — not a fully tuned or production-ready system.

## Development Order

1. Video loading
2. Frame sampling
3. Face extraction/cropping
4. PyTorch Dataset/DataLoader
5. Real/Fake labels
6. Working ResNet18 baseline
7. FFT frequency extraction
8. Frequency feature extractor (lightweight CNN)
9. Spatial + frequency feature fusion
10. Transformer Encoder temporal module
11. Basic training loop
12. Validation/evaluation metrics
13. Video inference
14. Minimal Streamlit UI (if time permits)

Implement in this order. Do not start a later step before the earlier steps
it depends on are working.

## Coding Rules

- Do not overengineer. Prefer the simplest implementation that satisfies the
  documented architecture.
- Do not add abstractions, config systems, or plugin layers beyond what the
  current milestone needs.
- Do not introduce alternative architectures as confirmed decisions. If a
  backbone or face detector choice is not finalized, treat it as
  configurable (a constructor/config argument), not as a design decision to
  debate in code comments.
- Use PyTorch's `nn.TransformerEncoder` / `nn.MultiheadAttention`. Do not use
  Hugging Face or other LLM infrastructure.
- Keep default hyperparameters as specified in this document unless the user
  changes them.
- No commented-out code, no speculative "future work" stubs, no unused
  config flags.

## Default Configuration

```
IMAGE_SIZE          = 224
NUM_FRAMES           = 8
D_MODEL              = 256
TRANSFORMER_LAYERS   = 2
N_HEADS              = 8
```

## Explicitly Out of Scope

- LLMs / GPT / Hugging Face LLM infrastructure
- Databases
- Authentication
- Microservices
- Cloud infrastructure / deployment
- Distributed training
- Complex frontend (Streamlit only, minimal)
- Audio processing
- GAN-based generation of fake content
- Unnecessary third-party APIs
- Complicated experiment tracking (e.g. MLflow, W&B) — plain logs/metrics
  files are sufficient

## Definition of Done (40% Milestone)

The milestone is complete when:

- Videos can be loaded and frames sampled deterministically.
- Faces are detected and cropped from sampled frames.
- A PyTorch `Dataset`/`DataLoader` yields labeled (REAL/FAKE) frame
  sequences ready for training.
- The ResNet18 baseline trains end-to-end on the prepared data and produces
  predictions.
- The frequency branch (FFT -> log-magnitude -> lightweight CNN) runs and
  produces per-frame frequency embeddings.
- Spatial and frequency embeddings are fused per frame and passed through
  the Transformer encoder to produce a video-level prediction.
- A basic training loop runs for both the baseline and hybrid model and
  produces loss curves.
- Validation metrics (accuracy, precision, recall, F1) can be computed on a
  held-out split.
- Given a new video file, the pipeline produces a REAL/FAKE prediction with
  a confidence score end-to-end (script or function, UI optional).

The Streamlit UI is optional for this milestone and does not block
"done" status if time-constrained.
