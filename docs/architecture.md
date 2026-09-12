# Architecture

This document describes the model architecture. See [`CLAUDE.md`](../CLAUDE.md)
for project scope and rules.

## End-to-End Architecture

```
Video file
   |
   v
Frame Sampling            -> NUM_FRAMES (default 8) frames, evenly spaced
   |
   v
Face Detection/Cropping   -> face crop per frame, resized to IMAGE_SIZE x IMAGE_SIZE
   |
   +------------------------+
   |                        |
   v                        v
Spatial Branch          Frequency Branch
(ResNet18)               (FFT + lightweight CNN)
   |                        |
   v                        v
spatial_embed          freq_embed
(per frame)             (per frame)
   |                        |
   +-----------+------------+
               v
        Per-frame Fusion
        (concat -> linear projection to D_MODEL)
               |
               v
     Transformer Encoder
     (temporal modeling across the frame sequence)
               |
               v
        Sequence Pooling
        (mean pool or [CLS]-style token)
               |
               v
        Classification Head
        (linear -> 1 logit)
               |
               v
        BCEWithLogitsLoss / sigmoid
               |
               v
           REAL / FAKE + confidence
```

Input/output shapes below assume batch size `B`, `NUM_FRAMES = 8`,
`IMAGE_SIZE = 224`, `D_MODEL = 256`.

## Spatial Branch

- Backbone: pretrained ResNet18 (ImageNet weights), configurable to swap for
  another torchvision backbone if needed later — not finalized as anything
  other than ResNet18 for this milestone.
- The final fully-connected classification layer is removed; the backbone's
  pooled feature vector is used as the spatial embedding.
- Input per frame: `(B, 3, 224, 224)`
- Output per frame: `spatial_embed`, shape `(B, 512)` (ResNet18 penultimate
  feature size)

## Frequency Branch

- Input: the same face crop used by the spatial branch, converted to a
  frequency representation. The transform runs inside the model
  (`torch.fft.fft2`), on whatever device the model is on. Because the crop
  arrives ImageNet-normalised for the spatial branch, it is de-normalised back
  to `[0, 1]` first, so the spectrum is computed from actual pixel values:
  1. Convert crop to grayscale (or process per-channel — configurable).
  2. Apply 2D FFT (`torch.fft.fft2` or `numpy.fft.fft2`).
  3. Shift zero-frequency component to center (`fftshift`).
  4. Take magnitude spectrum.
  5. Log-transform: `log(1 + magnitude)`.
  6. Normalize (e.g. min-max or z-score) to a stable input range.
- This produces a single-channel (or few-channel) frequency image of shape
  `(B, 1, 224, 224)`.
- A lightweight CNN (a handful of conv/pool/batchnorm blocks — not a
  full ResNet) processes this frequency image into an embedding.
- Output per frame: `freq_embed`, shape `(B, F_DIM)`. `F_DIM` is a small
  embedding size (e.g. 128), configurable, kept smaller than the spatial
  embedding since the CNN is lightweight.

## Per-Frame Fusion

- For each frame: `fused = concat(spatial_embed, freq_embed)`, shape
  `(B, 512 + F_DIM)`.
- A linear projection maps the concatenated vector to the common Transformer
  input dimension: `fused_proj = Linear(512 + F_DIM -> D_MODEL)`, shape
  `(B, D_MODEL)` = `(B, 256)`. A `LayerNorm` follows the projection to keep the
  two branches' differing scales from destabilising the Transformer.
- This fusion is done independently per frame, before temporal modeling.

## Transformer Temporal Modeling

- After per-frame fusion, the video is represented as a sequence of
  `NUM_FRAMES` fused embeddings: shape `(B, NUM_FRAMES, D_MODEL)` =
  `(B, 8, 256)`.
- A positional encoding (fixed sinusoidal or learned embedding) is added to
  encode frame order.
- Implementation: `torch.nn.TransformerEncoder` built from
  `torch.nn.TransformerEncoderLayer`, using `TRANSFORMER_LAYERS` (default 2)
  layers and `N_HEADS` (default 8) attention heads, operating on `D_MODEL`
  (default 256).
- **This Transformer is NOT an LLM.** It contains no tokenizer, no
  vocabulary, and no language modeling objective. It is a standard
  `nn.TransformerEncoder` used purely as a sequence-modeling module over a
  short sequence (8) of numeric frame embeddings, analogous to its use in
  vision/video transformers (e.g. ViT-style temporal modeling), not a text
  model.
- Output: contextualized per-frame embeddings, shape `(B, 8, 256)`.

## Classification

- The per-frame Transformer outputs are pooled into a single video-level
  embedding, shape `(B, 256)`. Default pooling: mean pooling over the
  temporal dimension. (A learned `[CLS]` token is an acceptable configurable
  alternative but is not the default.)
- A linear classification head maps this to a single logit:
  `Linear(D_MODEL -> 1)`, shape `(B, 1)`.
- Loss: `BCEWithLogitsLoss` against the binary REAL(0)/FAKE(1) label.
- Inference: `sigmoid(logit)` gives the FAKE probability, used as the
  confidence score. Threshold at 0.5 for the REAL/FAKE decision by default.

## Baseline (ResNet18-only)

A separate, simpler model used for comparison against the hybrid model:

- Per sampled frame: pretrained ResNet18 -> pooled feature `(B, 512)` ->
  linear classification head -> `(B, 1)` logit.
- No frequency branch, no fusion step, no Transformer.
- Video-level prediction: average the per-frame logits (or probabilities)
  across the `NUM_FRAMES` sampled frames.
- Trained and evaluated with the same `BCEWithLogitsLoss` and the same
  data splits as the hybrid model, so results are directly comparable (see
  [`evaluation.md`](evaluation.md)).

## Summary of Tensor Shapes

| Stage                         | Shape                  |
|--------------------------------|-------------------------|
| Sampled face crops (per video) | `(B, 8, 3, 224, 224)`  |
| Spatial embedding (per frame)  | `(B, 8, 512)`          |
| Frequency image (per frame)    | `(B, 8, 1, 224, 224)`  |
| Frequency embedding (per frame)| `(B, 8, F_DIM)`        |
| Fused + projected (per frame)  | `(B, 8, 256)`          |
| Transformer output             | `(B, 8, 256)`          |
| Pooled video embedding         | `(B, 256)`             |
| Logit                          | `(B, 1)`               |
