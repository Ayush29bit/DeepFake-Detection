# UI

This document describes the minimal demo UI. See [`CLAUDE.md`](../CLAUDE.md)
for scope — the UI is optional for the current milestone and should stay
minimal.

## Purpose

A minimal Streamlit application to demo the trained model on an uploaded
video. This is a demo tool, not a product — it exists to show the pipeline
working end-to-end, not to be a polished frontend.

## Scope

The app consists of exactly:

1. **Video upload** — a file uploader accepting a video file (e.g. mp4).
2. **Video preview** — render the uploaded video in-browser so the user can
   see what was submitted.
3. **Analyze button** — a single button that triggers inference
   (`src/inference/predict_video.py`) on the uploaded video.
4. **REAL/FAKE prediction** — the model's decision, displayed clearly.
5. **Confidence** — the model's confidence score (e.g. sigmoid output as a
   percentage) displayed alongside the prediction.
6. **Optional sample face frame** — optionally display one extracted/cropped
   face frame used by the model, so the user can sanity-check face
   detection worked correctly.

## Explicitly Out of Scope

- No user accounts, login, or session persistence beyond the current
  Streamlit session.
- No history/database of past predictions.
- No multi-page navigation, theming, or custom styling beyond Streamlit
  defaults.
- No batch upload / multi-video comparison.
- No model selection UI (baseline vs hybrid) unless trivially exposed as a
  single dropdown — not required for the milestone.
- No audio handling.

## Implementation Notes

- Single file: `src/ui/app.py`.
- Calls directly into `src/inference/predict_video.py`; no separate backend
  server/API layer.
- Loads the trained model checkpoint once (cached via
  `st.cache_resource` or equivalent) rather than reloading per request.
- Runs locally via `streamlit run src/ui/app.py`.
