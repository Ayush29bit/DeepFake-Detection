# UI

This document describes the minimal demo UI. See [`CLAUDE.md`](../CLAUDE.md)
for scope — the UI is optional for the current milestone and should stay
minimal.

## Purpose

A single-page Streamlit application to demo the trained model on an uploaded
video. It is a demo tool, not a product, but it is the face of the project in
presentations, so it is styled: a dark theme (`.streamlit/config.toml`) plus
CSS injected from `app/app.py`. There is still no separate frontend stack.

## Scope

The app consists of exactly:

1. **Video upload** — a file uploader accepting a video file (e.g. mp4).
2. **Video preview** — render the uploaded video in-browser so the user can
   see what was submitted.
3. **Analyze button** — a single button that triggers inference
   (`inference/predict.py`) on the uploaded video.
4. **REAL/FAKE prediction** — the model's decision, displayed clearly.
5. **Confidence** — the model's confidence score (e.g. sigmoid output as a
   percentage) displayed alongside the prediction.
6. **Face crop strip** — the `NUM_FRAMES` cropped faces given to the model,
   so the user can sanity-check that face detection worked.
7. **Spectrum strip** — for hybrid checkpoints, the log-magnitude FFT image
   of each crop, i.e. what the frequency branch saw.
8. **Model card** — checkpoint type (hybrid/baseline), best epoch, validation
   ROC-AUC and F1, threshold, read from the checkpoint file.

## Explicitly Out of Scope

- No user accounts, login, or session persistence beyond the current
  Streamlit session.
- No history/database of past predictions.
- No multi-page navigation or sidebar.
- No React/JS frontend or build step; styling is theme config + CSS only.
- No batch upload / multi-video comparison.
- Model selection is a single dropdown over `checkpoints/*.pt`; nothing more.
- No audio handling.

## Implementation Notes

- Single file: `app/app.py`.
- Calls directly into `inference/predict.py`; no separate backend
  server/API layer.
- Loads the trained model checkpoint once (cached via
  `st.cache_resource` or equivalent) rather than reloading per request.
- Runs locally via `streamlit run app/app.py`.
