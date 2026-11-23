# DeepFake-Detection

A research-oriented video deepfake detector. Given a face video, it predicts
**REAL** or **FAKE** with a confidence score.

The main model is a **hybrid** that fuses three kinds of evidence:

- **Spatial** features: what each face frame looks like (ResNet18 pretrained on ImageNet).
- **Frequency** features: artefacts in the 2D FFT spectrum of each frame (a small CNN).
- **Temporal** relationships across the sampled frames (a Transformer encoder).

A frame-level **ResNet18 baseline** is trained alongside it for comparison.

## Architecture

```
Video
  -> Uniform frame sampling (8 frames)
  -> Face detection and crop, resize to 224x224
  -> For each frame:
       RGB face       -> ResNet18                      -> spatial embedding (512)
       grayscale face -> 2D FFT -> log magnitude -> CNN -> frequency embedding (128)
       concatenate    -> linear projection              -> frame embedding (256)
  -> Positional encoding + Transformer encoder (2 layers, 8 heads, d_model 256)
  -> Mean over frames -> linear head (1 logit, BCEWithLogitsLoss)
  -> REAL / FAKE + confidence
```

Baseline: ResNet18 scores each frame on its own, and the frame logits are
averaged to give one score per video.

Full details are in [docs/architecture.md](docs/architecture.md).

## Repository layout

| Path | Contents |
|---|---|
| `preprocessing/` | Video reading, frame sampling, face cropping, Dataset, grouped splits, manifest builder |
| `models/` | Baseline, frequency branch, hybrid model |
| `training/` | Config (every field is a command-line flag) and the training loop |
| `evaluation/` | Metrics, checkpoint evaluation, comparison table |
| `inference/` | Single-video prediction |
| `app/` | Streamlit demo |
| `tests/` | End-to-end smoke tests with synthetic clips |
| `runs/eval/` | Test-set metrics for every trained model |
| `paper/` | LaTeX paper draft, figures and plotted data |
| `presentation/` | Script that builds the project presentation deck |
| `docs/` | Architecture, datasets, implementation, evaluation and UI notes |

Not in git: `data/` (dataset licence), `checkpoints/` (~500 MB of weights)
and the rest of `runs/` (per-run logs and splits).

## Setup

Python 3.12 was used. From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Without a GPU, install the CPU build of PyTorch first:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

`opencv-python` is pinned below 5.0 because 5.x no longer ships the Haar
cascade used by the default face detector.

Check that everything runs (synthetic clips and random weights, so this tests
the pipeline, not accuracy):

```bash
python -m tests.smoke_test
```

## Dataset

Experiments use the **DFDC training sample**: 400 labelled videos (77 REAL,
323 FAKE) from the Kaggle Deepfake Detection Challenge. The official download
was blocked for our account, so the files come from the Kaggle re-upload
`francisbawa/dfdc-train-sample`, which has the same file names and the original
`metadata.json`. The videos must not be redistributed.

Put the videos and `metadata.json` in `data/raw/dfdc/`, then build the manifest:

```bash
python -m preprocessing.make_manifest --metadata data/raw/dfdc/metadata.json --output data/raw/dfdc/manifest.csv
```

Splits are made per source video, not per frame or per clip: a fake and the
real video it was made from always land in the same split. With seed 42 the
split is train 294, val 54, test 52 videos. Details are in
[docs/datasets.md](docs/datasets.md).

## Usage

All commands run from the repository root.

Train:

```bash
python -m training.train --model baseline --manifest data/raw/dfdc/manifest.csv --train-jitter false --num-epochs 5 --batch-size 2 --best-metric roc_auc --run-name dfdc_baseline
python -m training.train --model hybrid   --manifest data/raw/dfdc/manifest.csv --train-jitter false --num-epochs 5 --batch-size 2 --best-metric roc_auc --run-name dfdc_hybrid
```

Add `--seed 7` (or another seed) for repeat runs. Add `--use-temporal false`
for the ablation without the Transformer. `python -m training.train --help`
lists every option.

Evaluate on the test split. `--val-threshold` picks the decision threshold on
the validation split instead of using 0.5:

```bash
python -m evaluation.evaluate --checkpoint checkpoints/dfdc_hybrid_best.pt --manifest data/raw/dfdc/manifest.csv --split test --val-threshold
python -m evaluation.compare runs/eval/dfdc_*_test*.json
```

Predict a single video:

```bash
python -m inference.predict --video path/to/video.mp4 --checkpoint checkpoints/dfdc_hybrid_best.pt
```

Demo UI:

```bash
streamlit run app/app.py
```

## Results

Test split of the DFDC sample (48 to 69 videos per seed), threshold 0.5,
5 epochs. Raw numbers for every run are in [runs/eval/](runs/eval/).

| Model | Seed | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 baseline | 42 | 0.442 | 1.000 | 0.293 | 0.453 | 0.752 |
| ResNet18 baseline | 7 | 0.464 | 0.839 | 0.448 | 0.584 | 0.669 |
| ResNet18 baseline | 123 | 0.542 | 0.895 | 0.459 | 0.607 | 0.767 |
| Hybrid | 42 | 0.692 | 0.857 | 0.732 | 0.789 | 0.612 |
| Hybrid | 7 | 0.710 | 0.896 | 0.741 | 0.811 | 0.708 |
| Hybrid | 123 | 0.708 | 0.871 | 0.730 | 0.794 | 0.826 |
| Hybrid without Transformer | 42 | 0.500 | 1.000 | 0.366 | 0.536 | 0.827 |

How to read these numbers:

- At threshold 0.5 the hybrid has higher accuracy and F1 than the baseline on
  every seed. ROC-AUC, which does not depend on the threshold, is mixed.
- When the threshold is chosen on the validation split, the seed-42 baseline
  reaches F1 0.894, higher than the hybrid's 0.815. Much of the gap at 0.5
  comes from where the threshold sits, not from how well the models rank
  videos.
- The test sets are small, so one video moves accuracy by about 2 points, and
  the classes are imbalanced (about 80% FAKE). Treat these numbers as early
  evidence, not a final result.

## Limitations

- Small dataset (400 videos) and short training (5 epochs).
- The default face detector is OpenCV's Haar cascade. MTCNN is supported
  (`--face-detector mtcnn`, needs `pip install facenet-pytorch`) but was not
  used in the reported runs.
- No cross-dataset evaluation yet. FaceForensics++ and Celeb-DF are planned
  (see [docs/datasets.md](docs/datasets.md)).
- No audio analysis; the model only looks at video frames.
