"""Streamlit demo: upload a video, get REAL/FAKE plus confidence.

Scope is what docs/ui.md specifies - upload, preview, analyse, verdict,
confidence, the face crops the model saw and (for the hybrid model) the
spectra the frequency branch saw. Single file, no backend layer.

    streamlit run app/app.py
"""

from __future__ import annotations

import glob
import html
import math
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import predict_video          # noqa: E402
from evaluation.evaluate import load_checkpoint      # noqa: E402
from models.frequency import fft_log_magnitude       # noqa: E402
from preprocessing.dataset import normalise          # noqa: E402

st.set_page_config(page_title="Deepfake Detection", page_icon="🎭",
                   layout="centered", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--bg:#0b0f1a;--panel:rgba(19,26,43,.72);--line:rgba(120,140,190,.18);
      --text:#e8ecf5;--muted:#8e9ab5;--accent:#7c5cff;--accent2:#22d3ee;
      --fake:#fb7185;--real:#34d399;--mono:'JetBrains Mono',ui-monospace,monospace;}
html,body,.stApp,p,label,button,input,li{font-family:'Inter',system-ui,sans-serif;}
[data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded'!important;}
.stApp{background:
  radial-gradient(900px 520px at 12% -10%,rgba(124,92,255,.30),transparent 60%),
  radial-gradient(760px 420px at 96% 8%,rgba(34,211,238,.16),transparent 60%),
  var(--bg);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer,.stDeployButton,[data-testid="stToolbar"]{display:none!important;}
.block-container{max-width:1080px;padding-top:3.4rem;padding-bottom:4rem;}

/* hero */
h1{font-weight:800!important;letter-spacing:-.035em;font-size:3.3rem!important;
   line-height:1.04!important;padding:0!important;margin:0 0 .5rem!important;
   background:linear-gradient(92deg,#fff 0%,#c9d2ff 55%,#8fe7ff 100%);
   -webkit-background-clip:text;background-clip:text;color:transparent!important;}
h1 a{display:none;}
.eyebrow{display:inline-flex;gap:.6rem;align-items:center;font-size:.72rem;
   letter-spacing:.2em;text-transform:uppercase;color:var(--accent2);font-weight:600;}
.eyebrow::before{content:"";width:28px;height:2px;border-radius:2px;
   background:linear-gradient(90deg,var(--accent),var(--accent2));}
.lede{color:var(--muted);font-size:1.05rem;line-height:1.6;max-width:720px;margin:0 0 1.3rem;}
.pipe{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin-bottom:1.8rem;}
.pipe span{font-size:.78rem;padding:.38rem .72rem;border:1px solid var(--line);
   border-radius:999px;background:rgba(255,255,255,.03);color:#c7cfe3;font-weight:500;}
.pipe span.hot{border-color:rgba(124,92,255,.5);background:rgba(124,92,255,.12);color:#dcd4ff;}
.pipe i{color:var(--muted);font-style:normal;font-size:.8rem;}

/* cards (Streamlit bordered containers) */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);
   border:1px solid var(--line)!important;border-radius:18px!important;
   backdrop-filter:blur(14px);padding:.55rem .7rem;margin-bottom:.4rem;
   box-shadow:0 10px 40px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.04);}
.sec{display:flex;align-items:baseline;justify-content:space-between;margin:.1rem 0 .55rem;}
.sec h3{margin:0;padding:0;font-family:'Inter',system-ui,sans-serif!important;font-size:1.05rem!important;
   font-weight:700;letter-spacing:-.01em;color:var(--text);}
.sec > span{color:var(--muted);font-size:.8rem;font-family:var(--mono);}

/* widgets */
[data-testid="stWidgetLabel"] p{color:var(--muted)!important;font-size:.74rem!important;
   letter-spacing:.1em;text-transform:uppercase;font-weight:600!important;}
div[data-baseweb="select"]>div{background:rgba(255,255,255,.04)!important;
   border-color:var(--line)!important;border-radius:12px!important;min-height:48px;
   font-family:var(--mono);font-size:.9rem;}
[data-testid="stFileUploaderDropzone"]{background:rgba(124,92,255,.06)!important;
   border:1.5px dashed rgba(124,92,255,.45)!important;border-radius:16px!important;
   padding:1.5rem!important;transition:background .2s,border-color .2s;}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(124,92,255,.13)!important;
   border-color:var(--accent)!important;}
[data-testid="stFileUploaderDropzone"] button{border-radius:10px!important;}
[data-testid="stBaseButton-primary"]{background:linear-gradient(92deg,var(--accent),#5b8cff 60%,var(--accent2))!important;
   border:0!important;color:#fff!important;font-weight:700!important;letter-spacing:.02em;
   border-radius:12px!important;padding:.8rem 1.2rem!important;
   box-shadow:0 8px 28px rgba(124,92,255,.35);transition:transform .15s,box-shadow .15s;}
[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);
   box-shadow:0 12px 34px rgba(124,92,255,.5);}
[data-testid="stVideo"] video{border-radius:14px;}
[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--line);}
[data-testid="stImageCaption"]{color:var(--muted)!important;font-family:var(--mono);font-size:.7rem!important;}
[data-testid="stStatusWidget"]{display:none;}

/* model card */
.mc{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.6rem;margin-top:.6rem;}
.tile{background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:14px;padding:.75rem .9rem;}
.tile small{display:block;font-size:.66rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:.22rem;}
.tile b{font-size:1.15rem;font-weight:700;font-family:var(--mono);letter-spacing:-.02em;color:var(--text);}
.tag{display:inline-block;font-size:.68rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
   padding:.28rem .6rem;border-radius:8px;background:rgba(124,92,255,.16);color:#cfc5ff;border:1px solid rgba(124,92,255,.4);}
.tag.base{background:rgba(34,211,238,.12);color:#a5f0ff;border-color:rgba(34,211,238,.4);}

/* verdict */
.verdict{display:flex;align-items:center;gap:1.6rem;padding:.4rem .2rem .2rem;}
.verdict>div:first-child{flex:1;}
.badge{display:inline-flex;align-items:center;gap:.55rem;padding:.55rem 1.1rem;border-radius:999px;
   font-weight:800;letter-spacing:.16em;font-size:1rem;}
.badge .dot{width:10px;height:10px;border-radius:50%;background:currentColor;animation:pulse 1.6s infinite;}
.badge.fake{background:rgba(251,113,133,.14);color:var(--fake);border:1px solid rgba(251,113,133,.45);box-shadow:0 0 30px rgba(251,113,133,.25);}
.badge.real{background:rgba(52,211,153,.14);color:var(--real);border:1px solid rgba(52,211,153,.45);box-shadow:0 0 30px rgba(52,211,153,.25);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
.verdict h2{margin:.7rem 0 .25rem;font-size:1.65rem;font-weight:700;letter-spacing:-.02em;color:var(--text);}
.verdict p{margin:0;color:var(--muted);line-height:1.55;max-width:520px;}
.ring{position:relative;width:156px;height:156px;flex:0 0 156px;}
.ring svg{width:100%;height:100%;transform:rotate(-90deg);}
.ring circle{fill:none;stroke-width:9;stroke-linecap:round;}
.ring .track{stroke:rgba(255,255,255,.07);}
.ring .arc{animation:draw 1.1s cubic-bezier(.2,.8,.2,1);}
@keyframes draw{from{stroke-dashoffset:326.7}}
.ring-val{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.ring-val b{font-size:1.8rem;font-weight:800;letter-spacing:-.03em;font-family:var(--mono);color:var(--text);}
.ring-val span{font-size:.64rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}

/* probability bar */
.pbar{margin:1.3rem .2rem .2rem;}
.pbar-track{position:relative;height:12px;border-radius:999px;
   background:linear-gradient(90deg,#34d399,#facc15 50%,#fb7185);}
.pbar-cover{position:absolute;top:0;bottom:0;right:0;border-radius:0 999px 999px 0;background:rgba(11,15,26,.86);}
.pbar-thumb{position:absolute;top:50%;width:20px;height:20px;border-radius:50%;background:#fff;
   transform:translate(-50%,-50%);box-shadow:0 0 0 4px rgba(255,255,255,.18),0 4px 14px rgba(0,0,0,.5);}
.pbar-thr{position:absolute;top:-7px;bottom:-7px;width:2px;background:rgba(255,255,255,.55);transform:translateX(-50%);}
.pbar-thr i{position:absolute;top:-1.35rem;left:50%;transform:translateX(-50%);white-space:nowrap;
   font-style:normal;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.pbar-labels{display:flex;justify-content:space-between;margin-top:.55rem;font-size:.74rem;color:var(--muted);font-family:var(--mono);}
.pbar-labels b{color:var(--text);}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;margin-top:1.2rem;}
.note{color:var(--muted);font-size:.82rem;line-height:1.55;margin:.6rem .2rem 0;}
.foot{color:var(--muted);font-size:.78rem;text-align:center;margin-top:2.2rem;line-height:1.6;}
@media (max-width:720px){.tiles{grid-template-columns:repeat(2,1fr);}
   .verdict{flex-direction:column;align-items:flex-start;} h1{font-size:2.4rem!important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(checkpoint_path: str):
    """Load a checkpoint once and reuse it across reruns."""
    model, config, ckpt, device = load_checkpoint(checkpoint_path)
    return model, config, ckpt, device


def find_checkpoints() -> list:
    return sorted(glob.glob(os.path.join("checkpoints", "*.pt")))


def tile(label: str, value: str) -> str:
    return f'<div class="tile"><small>{label}</small><b>{value}</b></div>'


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def model_card(config, ckpt, device) -> str:
    is_hybrid = config.model == "hybrid"
    tag = ('<span class="tag">Hybrid · ResNet18 + FFT + Transformer</span>' if is_hybrid
           else '<span class="tag base">Baseline · frame-level ResNet18</span>')
    val = ckpt.get("val_metrics", {})
    return (f'{tag}<div class="mc">'
            + tile("Best epoch", str(ckpt.get("epoch", "n/a")))
            + tile("Val ROC-AUC", fmt(val.get("roc_auc")))
            + tile("Val F1", fmt(val.get("f1")))
            + tile("Threshold", fmt(config.threshold, 2))
            + tile("Frames / crop", f"{config.num_frames} / {config.image_size}px")
            + tile("Device", str(device).upper())
            + "</div>")


def verdict_card(result, threshold: float) -> str:
    fake = result.label == "FAKE"
    cls, colour = ("fake", "#fb7185") if fake else ("real", "#34d399")
    pct = result.confidence * 100
    circ = 2 * math.pi * 52
    offset = circ * (1 - result.confidence)
    margin = abs(result.fake_probability - threshold)
    if fake:
        headline = "This video shows signs of manipulation"
        body = (f"The model assigned a fake probability of {result.fake_probability:.3f}, "
                f"above the decision threshold of {threshold:.2f}.")
    else:
        headline = "No manipulation detected"
        body = (f"The model assigned a fake probability of {result.fake_probability:.3f}, "
                f"below the decision threshold of {threshold:.2f}.")
    if margin < 0.1:
        body += " The score sits close to the threshold, so treat this call as low-margin."
    return f"""
<div class="verdict">
  <div>
    <span class="badge {cls}"><span class="dot"></span>{result.label}</span>
    <h2>{headline}</h2>
    <p>{body}</p>
  </div>
  <div class="ring">
    <svg viewBox="0 0 120 120">
      <circle class="track" cx="60" cy="60" r="52"/>
      <circle class="arc" cx="60" cy="60" r="52"
              style="stroke:{colour};stroke-dasharray:{circ:.1f};stroke-dashoffset:{offset:.1f}"/>
    </svg>
    <div class="ring-val"><b>{pct:.1f}%</b><span>confidence</span></div>
  </div>
</div>"""


def probability_bar(p: float, threshold: float) -> str:
    p = min(max(p, 0.0), 1.0)
    return f"""
<div class="pbar">
  <div class="pbar-track">
    <div class="pbar-cover" style="width:{(1 - p) * 100:.2f}%"></div>
    <div class="pbar-thr" style="left:{threshold * 100:.2f}%"><i>threshold {threshold:.2f}</i></div>
    <div class="pbar-thumb" style="left:{p * 100:.2f}%"></div>
  </div>
  <div class="pbar-labels"><span>0.0 · REAL</span><span>P(FAKE) = <b>{p:.3f}</b></span><span>FAKE · 1.0</span></div>
</div>"""


def spectra(crops: np.ndarray) -> list:
    """Render what the frequency branch sees, as inferno-coloured images."""
    spec = fft_log_magnitude(normalise(crops)).squeeze(1).cpu().numpy()
    out = []
    for s in spec:
        img = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


# ------------------------------------------------------------------ hero
st.markdown('<div class="eyebrow">Hybrid spatial · frequency · temporal</div>',
            unsafe_allow_html=True)
st.title("Deepfake Detection")
st.markdown('<p class="lede">Upload a face video and the detector samples eight frames, '
            'crops the face in each, reads both the pixels and their Fourier spectrum, '
            'and lets a Transformer weigh the sequence before giving one verdict. '
            'Research prototype: a prediction here is a model output, not proof.</p>',
            unsafe_allow_html=True)
st.markdown('<div class="pipe"><span>Video</span><i>→</i><span>8 frames</span><i>→</i>'
            '<span>Face crops 224²</span><i>→</i><span class="hot">ResNet18</span>'
            '<span class="hot">2D FFT · CNN</span><i>→</i><span>Fusion</span><i>→</i>'
            '<span class="hot">Transformer</span><i>→</i><span>REAL / FAKE</span></div>',
            unsafe_allow_html=True)

checkpoints = find_checkpoints()
if not checkpoints:
    st.error("No checkpoint found in `checkpoints/`. Train a model first, e.g.\n\n"
             "`python -m training.train --model hybrid "
             "--real-dir data/raw/real --fake-dir data/raw/fake`")
    st.stop()

# ------------------------------------------------------------------ model
with st.container(border=True):
    st.markdown('<div class="sec"><h3>Model</h3><span>checkpoints/</span></div>',
                unsafe_allow_html=True)
    checkpoint = st.selectbox("Model checkpoint", checkpoints, format_func=os.path.basename)
    model, config, ckpt, device = get_model(checkpoint)
    st.markdown(model_card(config, ckpt, device), unsafe_allow_html=True)

# ------------------------------------------------------------------ video
with st.container(border=True):
    st.markdown('<div class="sec"><h3>Video</h3><span>mp4 · avi · mov · mkv</span></div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    analyze = False
    if uploaded is not None:
        st.video(uploaded)
        analyze = st.button("Analyze video", type="primary", width="stretch")

# ------------------------------------------------------------------ result
if uploaded is not None and analyze:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    try:
        with st.spinner("Sampling frames, cropping faces, running the model..."):
            started = time.perf_counter()
            result = predict_video(tmp_path, model, config, device=device, return_crops=True)
            elapsed = time.perf_counter() - started
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not process `{html.escape(uploaded.name)}`: {html.escape(str(exc))}")
        st.stop()
    finally:
        os.unlink(tmp_path)

    with st.container(border=True):
        st.markdown(f'<div class="sec"><h3>Verdict</h3><span>{html.escape(uploaded.name)}</span></div>',
                    unsafe_allow_html=True)
        st.markdown(verdict_card(result, config.threshold), unsafe_allow_html=True)
        st.markdown(probability_bar(result.fake_probability, config.threshold),
                    unsafe_allow_html=True)
        st.markdown('<div class="tiles">'
                    + tile("Logit", f"{result.logit:+.3f}")
                    + tile("Frames used", str(result.frames_used))
                    + tile("Faces detected", f"{result.faces_detected} / {result.frames_used}")
                    + tile("Inference", f"{elapsed:.1f}s")
                    + "</div>", unsafe_allow_html=True)
        if result.faces_detected < result.frames_used:
            st.markdown('<p class="note">Frames without a detected face reused a neighbouring '
                        'box or a centre crop, so some crops below may not contain a face.</p>',
                        unsafe_allow_html=True)

    crops = result.face_crops
    if crops is not None and len(crops):
        with st.container(border=True):
            st.markdown('<div class="sec"><h3>What the spatial branch saw</h3>'
                        f'<span>{len(crops)} face crops · {config.image_size}×{config.image_size}</span></div>',
                        unsafe_allow_html=True)
            for col, (i, crop) in zip(st.columns(len(crops)), enumerate(crops)):
                col.image(crop, caption=f"t={i + 1}", width="stretch")

        if config.model == "hybrid" and config.use_frequency:
            with st.container(border=True):
                st.markdown('<div class="sec"><h3>What the frequency branch saw</h3>'
                            '<span>log(1 + |FFT|) · min–max normalised</span></div>',
                            unsafe_allow_html=True)
                for col, (i, img) in zip(st.columns(len(crops)), enumerate(spectra(crops))):
                    col.image(img, caption=f"t={i + 1}", width="stretch")

st.markdown('<p class="foot">Probabilities are raw sigmoid outputs and have not been calibrated. '
            'Trained on a 400-video DFDC sample on CPU; see the paper for the evaluation.</p>',
            unsafe_allow_html=True)
