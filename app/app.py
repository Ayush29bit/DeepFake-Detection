"""Minimal Streamlit demo: upload a video, get REAL/FAKE plus confidence.

Scope is exactly what docs/ui.md specifies - upload, preview, analyse, result,
confidence, one sample face crop. Nothing else.

    streamlit run app/app.py
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import predict_video          # noqa: E402
from evaluation.evaluate import load_checkpoint      # noqa: E402

st.set_page_config(page_title="Deepfake Detection", layout="centered")


@st.cache_resource(show_spinner=False)
def get_model(checkpoint_path: str):
    """Load a checkpoint once and reuse it across reruns."""
    model, config, ckpt, device = load_checkpoint(checkpoint_path)
    return model, config, ckpt, device


def find_checkpoints() -> list:
    return sorted(glob.glob(os.path.join("checkpoints", "*.pt")))


st.title("Deepfake Detection")
st.caption("Hybrid spatial-frequency-temporal detector with Transformer-based "
           "fusion. Research prototype: a prediction here is a model output, "
           "not a verdict.")

checkpoints = find_checkpoints()
if not checkpoints:
    st.error("No checkpoint found in `checkpoints/`. Train a model first, e.g.\n\n"
             "`python -m training.train --model hybrid "
             "--real-dir data/raw/real --fake-dir data/raw/fake`")
    st.stop()

checkpoint = st.selectbox("Model checkpoint", checkpoints)
uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    st.video(uploaded)

    if st.button("Analyze", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Sampling frames, extracting faces, running the model..."):
                model, config, ckpt, device = get_model(checkpoint)
                result = predict_video(tmp_path, model, config, device=device,
                                       return_crops=True)

            left, right = st.columns(2)
            left.metric("Prediction", result.label)
            right.metric("Confidence", f"{result.confidence * 100:.1f}%")
            st.progress(min(max(result.fake_probability, 0.0), 1.0),
                        text=f"P(FAKE) = {result.fake_probability:.3f} "
                             f"(threshold {config.threshold})")
            st.caption(f"{result.frames_used} frames sampled, "
                       f"{result.faces_detected} with a detected face "
                       f"(the rest used the documented crop fallback).")

            if result.face_crops is not None and len(result.face_crops):
                st.image(result.face_crops[0],
                         caption="First face crop given to the model",
                         width=224)
        finally:
            os.unlink(tmp_path)
