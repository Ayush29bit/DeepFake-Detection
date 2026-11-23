# IMPLEMENTATION_ROADMAP.md
## Deepfake Detection — Second Major Project Evaluation

**Evaluation date:** 31 October 2026  
**Target:** At least 70% genuine implementation, with a working end-to-end demonstration and evidence from real experiments.

This document is the execution plan for Claude Code. It is deliberately focused: finish and validate the implementation first. Research-paper writing is Phase 2 and must be based on the actual code and measured results.

---

# 1. Project definition

**Working title:** Deepfake Detection via Hybrid Spatial-Frequency Feature Fusion and Transformer-Based Temporal Modeling

**Task:** Given a face video, predict whether it is REAL or FAKE.

**Research hypothesis:** Combining visual appearance, frequency-domain information, and relationships across video frames may improve detection compared with a spatial-only baseline. This is a hypothesis to test, not a result to assume.

## Target architecture

```text
Input video
   |
Uniform frame sampling
   |
Face detection / crop / resize
   |
For each sampled face frame
   |-----------------------------|
   |                             |
RGB face                    Grayscale face
   |                             |
Spatial CNN                  2D FFT
   |                             |
Spatial embedding            Magnitude spectrum
   |                             |
   |                         Frequency CNN
   |                             |
   |                         Frequency embedding
   |                             |
   |--------- concatenate -------|
                 |
          Linear projection
          (frame embedding)
                 |
      Sequence of frame embeddings
                 |
       Positional encoding
                 |
       Transformer Encoder
                 |
       Temporal aggregation
                 |
        Binary classifier
                 |
          REAL / FAKE
```

**Important architecture clarification:** The spatial and frequency branches operate on each sampled face frame. Their embeddings are fused per frame. The Transformer Encoder then models relationships across the ordered sequence of fused frame embeddings. In this design, temporal modeling is performed by the Transformer; it is not a separate third CNN branch.

## Initial configurable settings

- Image size: 224 × 224
- Frames per video: 8 initially
- Spatial backbone: ResNet18 pretrained on ImageNet
- Spatial embedding: 512 dimensions (verify from implementation)
- Frequency embedding: 128 dimensions (configurable)
- Fused frame embedding: 256 dimensions
- Transformer: 2 encoder layers, 8 attention heads, `d_model=256`
- Classifier: one logit for binary classification
- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW

These are starting values, not immutable requirements. Adjust batch size and model settings to the available hardware. Record any changes.

---

# 2. Two-phase execution strategy

## Phase 1 — Implementation and validation (do this first)

Objective: reach a defensible 70% implementation milestone.

The priority is not the number of files or lines of code. The priority is a functioning, tested system with real data, a trained checkpoint, reproducible evaluation, and a demonstrable inference path.

## Phase 2 — Research paper (start only after Phase 1 evidence exists)

Write the paper from the implemented system and actual experiment outputs. Do not invent results, citations, implementation details, or claims. The paper workflow is intentionally deferred until Phase 1 is complete.

---

# 3. First action: repository and progress audit

Before changing anything, inspect the existing repository and report:

1. Current directory and repository structure.
2. Existing source files and their purpose.
3. Existing documentation and project instructions.
4. Installed Python version, environment, dependencies and available compute device.
5. Which pipeline stages are implemented, partially implemented, missing, or broken.
6. Whether datasets are present, their directory layout, labels, and approximate sizes.
7. Whether trained checkpoints exist and how they were produced.
8. Whether training, validation, testing, inference, and UI commands currently work.
9. Existing test results and any known errors.

Do not assume that earlier planned code was completed. Inspect and run it.

Create a concise audit table:

| Component | Status | Evidence / command run | Blocker |
|---|---|---|---|
| Video preprocessing | Not started / Partial / Working | | |
| Face extraction | | | |
| Dataset and labels | | | |
| CNN baseline | | | |
| FFT frequency branch | | | |
| Per-frame fusion | | | |
| Transformer | | | |
| Training and validation | | | |
| Evaluation | | | |
| Video inference | | | |
| UI/demo | | | |

After the audit, continue with the highest-priority missing or broken component. Do not stop after writing the audit.

---

# 4. Implementation workstreams and acceptance criteria

## Workstream A — Data pipeline

Implement or validate:

- Video loading with OpenCV.
- Uniform temporal frame sampling.
- Face detection and cropping on sampled frames.
- Consistent resize and normalization.
- Graceful handling of unreadable videos and missing face detections.
- A PyTorch `Dataset` and `DataLoader`.
- Video-level labels: REAL = 0, FAKE = 1.

Expected sample shape:

`[num_frames, 3, 224, 224]`

Requirements:

- Split at video/source-video level, not individual frame level.
- Avoid placing frames or manipulated versions from the same source video across train and test splits where source relationships are known.
- Make splits reproducible with a fixed seed.
- Keep preprocessing deterministic for validation and test.
- Avoid loading the entire dataset into memory.

**Acceptance evidence:** Run a real sample through preprocessing and a DataLoader batch. Print and verify shapes, labels, frame count, and error handling.

## Workstream B — Baseline model

Implement a frame-level ResNet18 baseline:

`face frame → ResNet18 backbone → binary classification head`

Choose and document the video-level aggregation strategy for baseline predictions, such as averaging frame logits/probabilities. Do not silently compare a frame-level baseline against a video-level hybrid model without explaining aggregation.

**Acceptance evidence:**
- Forward pass succeeds.
- Loss calculation succeeds.
- One optimizer step succeeds.
- A short training run can save a checkpoint.
- Validation metrics can be generated.

## Workstream C — Frequency branch

For each face frame:

1. Convert RGB to grayscale.
2. Apply a 2D FFT.
3. Compute the magnitude spectrum.
4. Apply a numerically stable `log(1 + magnitude)` transform.
5. Normalize consistently.
6. Pass the frequency representation through a lightweight CNN.
7. Return a frequency embedding.

Use PyTorch FFT where practical. Make sure tensor dimensions and device placement are correct.

Do not claim that FFT inherently identifies fake content. It creates an alternate representation from which a model may learn useful signals.

**Acceptance evidence:** Visualize a sample input and its frequency representation; verify finite values, expected tensor shape, and a successful frequency-branch forward pass.

## Workstream D — Hybrid spatial-frequency frame encoder

For each sampled face frame:

- Spatial branch: ResNet18 feature vector.
- Frequency branch: FFT plus lightweight CNN feature vector.
- Concatenate the two embeddings.
- Project the concatenated vector to the common `d_model` dimension.

Conceptually:

`spatial_embedding || frequency_embedding → Linear projection → frame_embedding`

Process all sampled frames with shared branch weights.

**Acceptance evidence:** Verify output shape `[batch, num_frames, d_model]` and run a backward pass through both branches.

## Workstream E — Temporal Transformer and classifier

- Add positional encoding to preserve frame order.
- Pass the sequence of fused frame embeddings through a PyTorch Transformer Encoder.
- Aggregate sequence outputs (initially mean pooling).
- Pass the pooled representation to a binary classification head.
- Return logits during training; apply sigmoid only for probability/inference reporting.

Expected shapes:

- Input: `[batch, num_frames, 256]`
- Transformer output: `[batch, num_frames, 256]`
- Pooled representation: `[batch, 256]`
- Logits: `[batch]` or `[batch, 1]` (be consistent with labels and loss)

**Acceptance evidence:** Forward pass, loss calculation, backward pass, and optimizer update all succeed on a real DataLoader batch.

## Workstream F — Training and experiment management

Implement:

- Configurable training settings.
- Train and validation loops.
- `BCEWithLogitsLoss`.
- AdamW optimizer.
- Best-checkpoint saving based on a declared validation metric.
- Epoch-level train/validation loss and metrics.
- Fixed random seeds where practical.
- Clear logging of configuration and dataset split.
- Resume support only if straightforward and useful.

Prevent leakage:
- Do not use test data for model selection.
- Do not tune thresholds on the test set.
- Do not report training metrics as test performance.

Run a short smoke test first, then a meaningful training run with the available compute and time. Record epochs, batch size, learning rate, device, data subset, and checkpoint path.

**Acceptance evidence:** A real training run completes, a checkpoint is saved, and validation results are logged.

## Workstream G — Evaluation and analysis

Evaluate the baseline and hybrid model on the same held-out test split.

Report:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where both classes and score values permit it
- Confusion matrix

Use video-level predictions for video-level evaluation. State the aggregation method.

Create a compact comparison table:

| Model | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| ResNet18 baseline | measured | measured | measured | measured | measured |
| Spatial + frequency + Transformer | measured | measured | measured | measured | measured |

If time and data permit, run ablations:
1. Spatial only.
2. Spatial + frequency.
3. Spatial + frequency + temporal Transformer.

If an experiment cannot be completed, mark it as pending. Never fill missing cells with invented or illustrative values without explicitly labeling them as illustrative.

Cross-dataset evaluation on Celeb-DF is desirable only if the dataset is available and labels/preprocessing can be handled correctly. Do not mix it into training if claiming a held-out cross-dataset test.

**Acceptance evidence:** Saved metrics, confusion matrix, and a short interpretation of errors and limitations.

## Workstream H — Video inference

Implement a single inference entry point:

`video path → preprocessing → checkpoint → prediction`

Return:
- REAL or FAKE
- model probability
- checkpoint/model identifier
- number of sampled frames
- clear error if the video cannot be processed

The displayed confidence is the model's predicted probability, not a guarantee of correctness. Do not call it calibrated confidence unless calibration has been tested.

**Acceptance evidence:** Run inference on at least one real video and show the actual output. Use known labels for evaluation, not for altering predictions.

## Workstream I — Minimal demo UI

Only after inference works, complete a simple Streamlit UI:

- Project title and short description.
- Video upload and preview.
- Analyze button.
- REAL/FAKE output and probability.
- Clear loading/error states.
- Optional display of a sampled face frame.

The UI must call the real inference code and load a real checkpoint. No hardcoded or random predictions.

**Acceptance evidence:** Launch the UI and complete one real upload-to-prediction flow.

---

# 5. What counts as 70% implementation

Use the following milestone checklist. The percentage is a planning estimate, not a claim to the faculty. Claude must report completion based on tested evidence.

| Milestone | Weight |
|---|---:|
| Data ingestion, frame sampling, face preprocessing, dataset and split | 15% |
| Working ResNet18 baseline | 10% |
| FFT and frequency feature extractor | 10% |
| Spatial-frequency fusion and Transformer model | 20% |
| Training, validation, checkpoints and reproducibility | 15% |
| Test evaluation, metrics and result analysis | 10% |
| Video inference pipeline | 10% |
| Minimal functional UI/demo | 5% |
| Technical documentation and run instructions | 5% |
| **Total** | **100%** |

A component earns credit only when implemented and supported by evidence. Boilerplate files, TODOs, placeholder outputs, synthetic-only tests, or unexecuted code do not count as completed implementation.

**70% milestone target:** Complete the data pipeline, baseline, frequency branch, hybrid Transformer model, training/validation, test evaluation, and video inference. The UI can be minimal, but a real demo is strongly preferred.

---

# 6. Scope control: do not overengineer

Do not add unless a real requirement emerges:

- LLMs, GPT, LangChain, or language-model APIs.
- Audio analysis or audio-video fusion.
- Databases, authentication, user accounts, or dashboards.
- Microservices, Kubernetes, distributed training, or cloud deployment.
- Complex frontend/backend separation.
- GAN generation.
- Multiple competing backbones without an experiment-driven reason.
- Elaborate experiment tracking platforms.
- New documentation files beyond the existing project documentation set.

Prefer the existing repository conventions and dependencies. Do not overwrite working code without inspecting it first.

---

# 7. Technical quality and safety checks

Before calling the implementation milestone complete, verify:

- Model input/output dimensions are consistent.
- Labels and logits have compatible shapes.
- No NaN/Inf values are introduced by FFT preprocessing.
- All model branches are on the same device.
- Gradients flow through the intended trainable branches.
- Pretrained model weights and normalization are used correctly.
- Train/validation/test separation is respected.
- Test data is not used for tuning.
- Checkpoints include enough configuration to reproduce inference.
- Inference uses the same preprocessing as validation/test.
- UI predictions come from the actual model.
- Dataset and model licenses/usage terms are checked before redistribution.
- Face data is handled responsibly and only used for the project purpose.

---

# 8. Phase 1 completion report

At the end, produce a concise report in the terminal (and update existing project docs only where needed):

1. Repository audit summary.
2. Components implemented.
3. Components tested and exact evidence.
4. Components still incomplete.
5. Estimated milestone completion using the weighted table.
6. Dataset name, subset, split method, and sample counts.
7. Hardware/device used.
8. Baseline training configuration and checkpoint.
9. Hybrid training configuration and checkpoint.
10. Actual validation and test metrics.
11. Confusion matrix location.
12. Exact commands for training, evaluation, inference, and UI.
13. Known limitations and blockers.
14. Three highest-priority next tasks.

Do not report 70% merely because files exist. Report the evidence-backed percentage.

---

# 9. Phase 2 — Research paper (deferred)

Do not start writing the full research paper during the implementation work unless explicitly instructed.

Once Phase 1 is validated, prepare a paper using:
- The final implemented architecture.
- Verified literature and references.
- Actual experimental settings.
- Actual measured results.
- Honest limitations and error analysis.
- Original explanation and analysis by the team.

Never fabricate citations, DOIs, results, datasets, or claims. Cite all borrowed ideas and text. Follow the institution's rules on AI assistance and disclosure. Run the required similarity/plagiarism checks and review matches manually; do not attempt to evade detection systems.

---

# 10. Claude Code execution instructions

1. Read this roadmap and the existing `CLAUDE.md` and `docs/` files.
2. Audit the current repository before modifying it.
3. Reuse and repair existing implementation wherever sensible.
4. Present a short prioritized plan, then proceed without waiting for another approval.
5. Work in small stages and run tests after each stage.
6. Do not stop after creating boilerplate.
7. Do not spend time polishing the UI before the model pipeline works.
8. If the dataset is unavailable, implement and test the pipeline with synthetic tensors only for shape/gradient smoke tests, then clearly identify the real-data work still blocked. Synthetic tests do not count as real model training.
9. If compute is limited, reduce batch size, frame count, or model size in configuration; do not silently change the architecture.
10. Keep the project runnable and provide exact commands.

**Immediate next action:** Perform the repository audit, identify the largest gap against the 70% checklist, and begin implementing that gap.
