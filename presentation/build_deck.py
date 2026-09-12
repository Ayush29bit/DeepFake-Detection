"""Builds the Final Year Project presentation (PPTX + PDF).

Run:  python presentation/build_deck.py
"""

import math
import os

from deck_lib import (AMBER, AMBER_LIGHT, BLUE, BLUE_LIGHT, LINE, MUTED, NAVY, PANEL,
                      SLIDE_H, SLIDE_W, TEAL, TEAL_LIGHT, TEXT, WHITE, Circle, Line, Para,
                      Rect, Slide, Table, Text, blend, render_pdf, render_pptx)

TITLE = ("Deepfake Detection via Hybrid Spatial-Temporal-Frequency Feature Learning "
         "with Transformer-Based Fusion")
TEAM = ["Ayush Tiwari", "Ayushman Patel", "Atharv Shukla", "Bhanupratap Chourasia"]
DATE = "12 September 2026"

L, R = 0.6, 12.75          # content left / right edge
CW = R - L                 # content width
TOP = 1.55                 # content top

TAG_STYLE = {
    "proposed": (TEAL, TEAL_LIGHT),
    "planned": (BLUE, BLUE_LIGHT),
    "concept": (NAVY, PANEL),
    "caution": (AMBER, AMBER_LIGHT),
}

slides = []


# ---------------------------------------------------------------- helpers
def base(title, tag=None, kind="proposed"):
    sd = Slide()
    sd.add(Text(L, 0.42, 9.6, 0.8, [title], size=28, bold=True, color=NAVY, valign="m"))
    sd.add(Rect(L + 0.05, 1.22, 0.6, 0.05, fill=TEAL))
    if tag:
        fg, bg = TAG_STYLE[kind]
        w = 0.16 * len(tag) + 0.35
        sd.add(Rect(R - w, 0.6, w, 0.36, fill=bg, stroke=fg, stroke_w=0.75,
                    paras=[tag.upper()], size=9, color=fg, bold=True))
    sd.add(Line(L, 6.95, R, 6.95, color=LINE, width=0.75))
    sd.add(Text(L, 6.98, 9.5, 0.3, [TITLE], size=8.5, color=MUTED, valign="m"))
    sd._page_slot = Text(R - 1.2, 6.98, 1.2, 0.3, ["#"], size=9, color=MUTED, align="r",
                         valign="m")
    sd.add(sd._page_slot)
    slides.append(sd)
    return sd


def arrow(x1, y1, x2, y2, color=MUTED, w=1.25):
    return Line(x1, y1, x2, y2, color=color, width=w, arrow=True)


def stage(x, y, w, h, text, fill=NAVY, color=WHITE, size=11.5, stroke=None, bold=True):
    return Rect(x, y, w, h, fill=fill, stroke=stroke, paras=[text], size=size, color=color,
                bold=bold)


def soft(x, y, w, h, paras, fill=PANEL, stroke=LINE, size=12, align="l", valign="t",
         inset=0.16, color=TEXT, space_after=4):
    return Rect(x, y, w, h, fill=fill, stroke=stroke, paras=paras, size=size, color=color,
                align=align, valign=valign, inset=inset, space_after=space_after)


def heading(text, size=13, color=NAVY):
    return Para(text, size=size, bold=True, color=color, space_after=6)


def bullet(text, size=None, space_after=None):
    return Para(text, bullet=True, size=size, space_after=space_after)


def caption(x, y, w, text, size=9.5, color=MUTED, align="l", h=0.35):
    return Text(x, y, w, h, [text], size=size, color=color, align=align, valign="t")


def spectrum_grid(sd, x0, y0, n=9, cell=0.3, gap=0.05, c_lo=NAVY, c_hi=TEAL):
    """Decorative log-magnitude-spectrum-like tile grid (bright centre)."""
    c = (n - 1) / 2
    for i in range(n):
        for j in range(n):
            d = math.hypot(i - c, j - c)
            t = math.exp(-((d / 2.4) ** 2))
            col = blend(c_lo, c_hi, 0.06 + 0.9 * t)
            sd.add(Rect(x0 + j * (cell + gap), y0 + i * (cell + gap), cell, cell, fill=col))


def frame_strip(sd, x0, y0, n=8, w=0.3, h=0.2, gap=0.1, stroke=None):
    stroke = stroke or blend(NAVY, WHITE, 0.4)
    for k in range(n):
        sd.add(Rect(x0 + k * (w + gap), y0, w, h, stroke=stroke, stroke_w=0.75))


# ================================================================ 1. Title
sd = Slide(bg=NAVY)
slides.append(sd)
sd.add(Text(0.8, 1.3, 8, 0.35, ["FINAL YEAR MAJOR PROJECT"], size=11, bold=True, color=TEAL))
sd.add(Text(0.8, 1.7, 8.3, 2.4, [TITLE], size=30, bold=True, color=WHITE, valign="t",
            line_spacing=1.1))
sd.add(Rect(0.85, 4.2, 0.7, 0.04, fill=TEAL))
sd.add(Text(0.8, 4.4, 8.3, 0.5, ["   ·   ".join(TEAM)], size=13, color=WHITE, valign="m"))
soft_white = blend(NAVY, WHITE, 0.72)
sd.add(Text(0.8, 5.0, 8.3, 1.4, [
    Para("Department of ____________________________   |   ____________________________ (Institution)",
         space_after=5),
    Para("Faculty Guide: ____________________________", space_after=5),
    Para(f"Presentation Date: {DATE}"),
], size=12, color=soft_white))
spectrum_grid(sd, 9.45, 2.0)
frame_strip(sd, 9.45, 5.45)
sd.add(Text(9.45, 5.75, 3.1, 0.3, ["spatial · temporal · frequency"], size=9,
            color=blend(NAVY, WHITE, 0.45)))

# ================================================================ 2. Team
sd = base("Project Team")
sd.add(caption(L, 1.3, 10, "Final Year Major Project  ·  " + TITLE, size=10.5))
card_w, card_h = 5.85, 1.95
for i, name in enumerate(TEAM):
    cx = L + (i % 2) * (card_w + 0.45)
    cy = 1.85 + (i // 2) * (card_h + 0.35)
    sd.add(Rect(cx, cy, card_w, card_h, fill=PANEL, stroke=LINE))
    initials = "".join(p[0] for p in name.split())
    sd.add(Circle(cx + 1.0, cy + card_h / 2, 0.52, fill=NAVY, paras=[initials], size=16))
    sd.add(Text(cx + 1.8, cy + 0.45, 3.9, 1.1, [
        Para(name, size=20, bold=True, color=NAVY, space_after=6),
        Para("Enrollment No.: ______________", size=11, color=MUTED),
    ], valign="m"))
sd.add(Text(L, 6.5, CW, 0.35, [
    "Department of ____________________   |   ____________________ (Institution)   |   Faculty Guide: ____________________"],
    size=10.5, color=MUTED, align="c", valign="m"))

# ================================================================ 3. Problem
sd = base("Problem Identification")
sd.add(Text(L, TOP, 6.6, 5.2, [
    heading("Why detection is hard, and why it is still an open problem", size=13.5),
    bullet("Deepfake technology enables highly realistic manipulation of facial content.", space_after=8),
    bullet("Human visual inspection is increasingly unreliable for judging authenticity.", space_after=8),
    bullet("Modern generation techniques reduce the obvious visual artifacts that early detectors relied on.", space_after=8),
    bullet("Detection models can become dependent on dataset-specific artifacts rather than general forgery cues.", space_after=8),
    bullet("A detector trained on one manipulation distribution may perform poorly on unseen data: new generators, different compression, lighting or capture conditions.", space_after=8),
    bullet("**Robust and generalizable detection therefore remains an important research problem.**"),
], size=13.5))
# distribution-shift diagram
dx = 7.55
sd.add(stage(dx + 0.9, 1.7, 3.4, 0.85, "Detector trained on known manipulation methods", size=11.5))
sd.add(arrow(dx + 2.0, 2.55, dx + 1.25, 3.2))
sd.add(arrow(dx + 3.2, 2.55, dx + 3.95, 3.2))
sd.add(Rect(dx, 3.2, 2.5, 1.05, fill=BLUE_LIGHT, stroke=LINE, paras=[
    Para("In-domain evaluation", bold=True, color=NAVY, space_after=2),
    "Same dataset, same manipulation methods"], size=10.5, inset=0.1))
sd.add(Rect(dx + 2.7, 3.2, 2.5, 1.05, fill=AMBER_LIGHT, stroke=LINE, paras=[
    Para("Unseen conditions", bold=True, color=AMBER, space_after=2),
    "New generators, compression, lighting"], size=10.5, inset=0.1))
sd.add(arrow(dx + 1.25, 4.25, dx + 1.25, 4.85))
sd.add(arrow(dx + 3.95, 4.25, dx + 3.95, 4.85))
sd.add(Rect(dx, 4.85, 2.5, 0.75, stroke=NAVY, paras=["Typically strong performance"],
            size=11, color=NAVY, bold=True))
sd.add(Rect(dx + 2.7, 4.85, 2.5, 0.75, stroke=AMBER, paras=["Performance may degrade"],
            size=11, color=AMBER, bold=True))
sd.add(caption(dx, 5.8, 5.2, "Illustration of the generalization problem this project targets. "
               "Not an experimental result.", size=9.5))

# ================================================================ 4. Motivation
sd = base("Why Deepfake Detection Matters")
sd.add(caption(L, 1.3, 11, "Reliable detection is one component of a broader response to "
               "synthetic media. Consequences of undetected manipulation:", size=11))
cx0, cy0 = SLIDE_W / 2, 4.15
nodes = [
    ("Misinformation", "Fabricated statements attributed to real people"),
    ("Identity impersonation & fraud", "Face spoofing in verification and social-engineering fraud"),
    ("Reputation damage", "Non-consensual or defamatory synthetic content"),
    ("Political & social manipulation", "Synthetic media used to influence public opinion"),
    ("Loss of trust in digital media", "Authentic footage can be dismissed as fake"),
    ("Cybersecurity implications", "Bypassing biometric checks; enabling targeted attacks"),
]
rx, ry, nw, nh = 4.45, 2.0, 2.6, 0.95
pos = []
for k in range(6):
    ang = math.radians(-90 + 60 * k)
    pos.append((cx0 + rx * math.cos(ang), cy0 + ry * math.sin(ang)))
for (px, py) in pos:
    sd.add(Line(cx0, cy0, px, py, color=LINE, width=1.25))
sd.add(Circle(cx0, cy0, 0.85, fill=NAVY, paras=[Para("Deepfake", size=13), Para("media", size=13)]))
for (px, py), (t, d) in zip(pos, nodes):
    sd.add(Rect(px - nw / 2, py - nh / 2, nw, nh, fill=WHITE, stroke=BLUE, stroke_w=1,
                paras=[Para(t, bold=True, color=NAVY, size=11.5, space_after=2),
                       Para(d, size=9.5, color=MUTED)], inset=0.08))

# ================================================================ 5. Signals
sd = base("Detection Signals in Deepfake Media", tag="Conceptual foundation", kind="concept")
cards = [
    ("SPATIAL", NAVY, "“What does the face look like?”",
     "Individual-frame visual information",
     ["Facial texture and edges", "Blending boundaries and artifacts",
      "Unnatural facial details", "Local visual inconsistencies"],
     "Pretrained CNN / vision backbone applied to each face frame"),
    ("TEMPORAL", BLUE, "“How does the face change across frames?”",
     "Relationships across the frame sequence",
     ["Unnatural or jittery motion", "Frame-to-frame inconsistencies",
      "Blinking and facial movement patterns", "Temporal artifacts"],
     "Transformer encoder over the sequence of frame-level features"),
    ("FREQUENCY", TEAL, "“What hidden pixel-level patterns exist beneath the visual appearance?”",
     "Frequency-domain representation of image information",
     ["Abnormal spectral distributions", "Up-sampling and blending traces",
      "Periodic patterns not visible in pixel space", "Attenuated high-frequency detail"],
     "FFT/DCT transform followed by a neural feature extractor"),
]
cw_, gap = 3.9, 0.225
for i, (name, col, q, defn, ex, cap) in enumerate(cards):
    x = L + i * (cw_ + gap)
    sd.add(Rect(x, TOP, cw_, 4.1, fill=WHITE, stroke=LINE))
    sd.add(Rect(x, TOP, cw_, 0.5, fill=col, paras=[name], size=12.5, color=WHITE, bold=True))
    sd.add(Text(x + 0.1, TOP + 0.6, cw_ - 0.2, 3.9, [
        Para(q, size=13.5, bold=True, color=NAVY, italic=True, space_after=4),
        Para(defn, size=10.5, color=MUTED, space_after=10),
        Para("May reveal:", size=11, bold=True, color=TEXT, space_after=3),
        *[bullet(e, size=11, space_after=2) for e in ex],
        Para("", size=6, space_after=0),
        Para("Captured by (proposed):", size=11, bold=True, color=TEXT, space_after=3),
        Para(cap, size=11, color=TEXT),
    ], inset=0.1))
sd.add(Text(L, 5.9, CW, 0.7, [
    "The three signals are complementary. The proposed framework is designed to combine them "
    "rather than rely on any single one; no single cue is assumed to be present in every deepfake."],
    size=11.5, color=NAVY, valign="m", align="c"))

# ================================================================ 6. Related work
sd = base("Related Work")
sd.add(caption(L, 1.3, 11, "Landscape of prior approaches, grouped by the type of evidence they "
               "exploit. Bracketed numbers refer to the References slides.", size=10.5))
rows = [
    ["Category", "Main idea", "Strength", "Limitation", "Representative work"],
    ["CNN-based detection",
     "Train a convolutional network on face crops to classify real vs. manipulated frames.",
     "Strong in-domain accuracy; simple to train from pretrained backbones.",
     "Tends to learn dataset-specific spatial artifacts; accuracy can drop on unseen manipulations or compression.",
     "MesoNet [4]; CNN baselines in FaceForensics++ [1]"],
    ["Vision Transformer / Transformer-based",
     "Apply self-attention over image patches or CNN feature maps to capture global context.",
     "Models long-range dependencies; flexible attention over regions.",
     "Data-hungry; remains frame-centric unless explicitly extended to video.",
     "ViT [6]; CNN + ViT for video deepfakes [16]"],
    ["Temporal / video-based",
     "Model frame sequences with recurrent or attention modules to find inconsistencies over time.",
     "Uses motion and coherence cues that single frames cannot provide.",
     "Higher compute; sensitive to frame sampling and face-tracking quality.",
     "RNN-based [11], [12]; temporal coherence [13]; eye blinking [14]"],
    ["Frequency-domain",
     "Analyse spectra (FFT/DCT) to expose up-sampling and blending traces left by generators.",
     "Sensitive to generator artifacts that are subtle in pixel space.",
     "Artifacts can be attenuated by compression; not guaranteed for every method.",
     "F3-Net [8]; Frank et al. [10]; Durall et al. [9]"],
    ["Hybrid / multi-stream",
     "Combine several cue types (spatial, frequency, temporal, attention maps) in one model.",
     "Complementary evidence can improve robustness.",
     "More complex; the fusion strategy and its benefit must be validated by ablation.",
     "Multi-attentional [17]; Face X-ray [15]"],
]
sd.add(Table(L, 1.75, CW, [1.55, 2.6, 2.2, 2.6, 2.05], rows, size=11, first_col_bold=True,
             alt_fill=PANEL, min_row_h=0.8, pad=0.09))

# ================================================================ 7. Research gap
sd = base("Research Gap")
gaps = [
    ("Dependence on spatial artifacts", "Single-frame appearance cues are generator-specific and easy to over-fit."),
    ("Limited modeling of temporal consistency", "Many detectors treat video as independent frames."),
    ("Under-utilization of frequency-domain information", "Spectral cues are often ignored or used in isolation."),
    ("Dataset-specific bias", "Benchmarks share capture conditions and generators; models absorb these regularities."),
    ("Weak cross-dataset generalization", "Performance frequently drops when evaluated on a different dataset."),
]
sd.add(Text(L, TOP, 6.2, 0.4, [heading("Limitations observed across existing approaches")]))
for i, (t, d) in enumerate(gaps):
    y = 2.05 + i * 0.9
    sd.add(Rect(L, y + 0.08, 0.12, 0.12, fill=TEAL))
    sd.add(Text(L + 0.25, y - 0.05, 5.9, 0.85, [
        Para(t, bold=True, color=NAVY, size=12.5, space_after=2),
        Para(d, size=11, color=MUTED)]))
qx = 7.3
sd.add(Rect(qx, TOP, R - qx, 2.95, fill=NAVY))
sd.add(Rect(qx, TOP, 0.12, 2.95, fill=TEAL))
sd.add(Text(qx + 0.3, TOP + 0.15, R - qx - 0.5, 2.7, [
    Para("RESEARCH QUESTION", size=10, bold=True, color=TEAL, space_after=8),
    Para("Can combining spatial, temporal, and frequency-domain information through "
         "attention-based feature fusion improve the robustness of deepfake detection?",
         size=17, bold=True, color=WHITE)], valign="m", line_spacing=1.15))
sd.add(soft(qx, 4.75, R - qx, 1.75, [
    Para("HYPOTHESIS  (to be tested, not yet validated)", size=10, bold=True, color=TEAL, space_after=6),
    Para("Complementary cues, fused with attention, may reduce dependence on any single artifact "
         "type and therefore hold up better under distribution shift (new manipulation methods, "
         "compression, capture conditions).", size=12.5)], fill=TEAL_LIGHT, stroke=TEAL, valign="m"))

# ================================================================ 8. Proposed solution
sd = base("Proposed Solution", tag="Proposed")
items = [
    ("Spatial branch", "Extract spatial information from RGB face frames with a pretrained CNN / vision backbone."),
    ("Frequency branch", "Transform face frames into frequency representations (FFT/DCT) and learn discriminative frequency features."),
    ("Temporal modeling", "Learn relationships across the sampled frame sequence with a Transformer encoder."),
    ("Attention-based fusion", "Combine the complementary representations using Transformer-based attention rather than fixed weighting."),
    ("Binary classification", "Classification head producing a REAL / FAKE decision with a confidence score."),
    ("Evaluation protocol", "Measure in-domain performance and, where feasible, cross-dataset generalization; ablate each component."),
]
bw, bh = 3.9, 1.7
for i, (t, d) in enumerate(items):
    x = L + (i % 3) * (bw + 0.225)
    y = TOP + (i // 3) * (bh + 0.22)
    sd.add(Rect(x, y, bw, bh, fill=WHITE, stroke=LINE))
    sd.add(Rect(x, y, 0.07, bh, fill=TEAL))
    sd.add(Text(x + 0.2, y + 0.12, bw - 0.35, bh - 0.2, [
        Para(f"0{i + 1}", size=10, bold=True, color=TEAL, space_after=2),
        Para(t, size=14, bold=True, color=NAVY, space_after=4),
        Para(d, size=11.5)]))
sy = TOP + 2 * (bh + 0.22) + 0.15
sd.add(Rect(L, sy, CW, 1.05, fill=PANEL, stroke=LINE))
sd.add(Text(L + 0.2, sy + 0.08, 1.6, 0.9, [Para("Project status", bold=True, color=NAVY, size=12)],
            valign="m"))
chips = [("CONFIRMED", "Topic and research direction", NAVY, PANEL),
         ("PROPOSED", "Hybrid architecture (this deck)", TEAL, TEAL_LIGHT),
         ("PLANNED", "Preprocessing, training, experiments", BLUE, BLUE_LIGHT),
         ("NOT YET AVAILABLE", "Experimental results", AMBER, AMBER_LIGHT)]
for i, (lab, desc, fg, bg) in enumerate(chips):
    x = L + 1.95 + i * 2.65
    sd.add(Rect(x, sy + 0.22, 2.45, 0.62, fill=bg, stroke=fg, paras=[
        Para(lab, size=9, bold=True, color=fg, space_after=1), Para(desc, size=9.5, color=TEXT)],
        inset=0.06))

# ================================================================ 9. Methodology
sd = base("Methodology", tag="Proposed")
sd.add(Rect(0.45, 1.35, 12.45, 2.45, fill=PANEL, stroke=LINE))
sd.add(Text(0.6, 1.4, 8, 0.3, ["FRAME-LEVEL PROCESSING  ·  applied to each of the N sampled frames"],
            size=9, bold=True, color=TEAL))
r1y, bh1, bw1 = 2.15, 0.8, 1.55
xs = [0.6, 2.57, 4.54, 6.51]
labels = ["Video", "Frame Sampling", "Face Detection & Alignment", "Preprocessing"]
subs = ["input video file", "N evenly spaced frames", "detector configurable",
        "crop · resize · normalize"]
for x, lab, sub in zip(xs, labels, subs):
    sd.add(stage(x, r1y, bw1, bh1, lab, size=11))
    sd.add(caption(x - 0.1, r1y + bh1 + 0.04, bw1 + 0.2, sub, size=8.5, align="c"))
for x in xs[:-1]:
    sd.add(arrow(x + bw1, r1y + bh1 / 2, x + bw1 + 0.42, r1y + bh1 / 2))
sx, sw = 8.48, 1.75
sd.add(stage(sx, 1.72, sw, 0.62, "Spatial Feature Extraction", fill=BLUE, size=10.5))
sd.add(caption(sx - 0.15, 2.36, sw + 0.3, "pretrained CNN / vision backbone", size=8.5, align="c"))
sd.add(stage(sx, 2.78, sw, 0.62, "Frequency Feature Extraction", fill=TEAL, size=10.5))
sd.add(caption(sx - 0.15, 3.42, sw + 0.3, "FFT/DCT → log-magnitude → CNN", size=8.5, align="c"))
sd.add(arrow(8.06, r1y + bh1 / 2, sx, 2.03))
sd.add(arrow(8.06, r1y + bh1 / 2, sx, 3.09))
fx, fw = 10.65, 2.05
sd.add(stage(fx, r1y, fw, bh1, "Frame-level Feature Representation", size=11))
sd.add(caption(fx - 0.1, r1y + bh1 + 0.04, fw + 0.2, "concat + projection, per frame", size=8.5, align="c"))
sd.add(arrow(sx + sw, 2.03, fx, r1y + bh1 / 2))
sd.add(arrow(sx + sw, 3.09, fx, r1y + bh1 / 2))
# connector to row 2
sd.add(Line(fx + fw / 2, r1y + bh1 + 0.3, fx + fw / 2, 3.88, color=NAVY, width=1.5))
sd.add(Line(fx + fw / 2, 3.88, 1.8, 3.88, color=NAVY, width=1.5))
sd.add(arrow(1.8, 3.88, 1.8, 4.75, color=NAVY, w=1.5))
sd.add(Text(2.0, 3.9, 6.5, 0.24, ["N frame-level vectors form one sequence per video"], size=9,
            color=NAVY, bold=True, valign="t", inset=0.02))
sd.add(Rect(0.45, 4.15, 12.45, 2.15, fill=PANEL, stroke=LINE))
sd.add(Text(0.6, 4.2, 8, 0.3, ["VIDEO-LEVEL MODELING  ·  operates on the sequence of N frame representations"],
            size=9, bold=True, color=TEAL))
r2y, bw2 = 4.75, 2.4
xs2 = [0.6, 3.6, 6.6, 9.6]
labels2 = ["Temporal Transformer Encoder", "Attention-based Feature Fusion",
           "Classification Head", "REAL / FAKE + Confidence Score"]
subs2 = ["self-attention across frames + positional encoding",
         "attention-weighted aggregation of frame representations (configuration to be finalized)",
         "linear layer → sigmoid → FAKE probability", "decision threshold 0.5 by default"]
fills2 = [NAVY, NAVY, NAVY, TEAL]
for x, lab, sub, f in zip(xs2, labels2, subs2, fills2):
    sd.add(stage(x, r2y, bw2, bh1, lab, fill=f, size=11.5))
    sd.add(Text(x - 0.1, r2y + bh1 + 0.04, bw2 + 0.2, 0.55, [sub], size=8.5, color=MUTED, align="c"))
for x in xs2[:-1]:
    sd.add(arrow(x + bw2, r2y + bh1 / 2, x + bw2 + 0.6, r2y + bh1 / 2))
sd.add(caption(L, 6.45, CW, "The diagram reflects the proposed design. The exact backbone, "
               "face detector and fusion configuration will be selected during implementation.",
               size=9.5, align="c"))

# ================================================================ 10. Spatial
sd = base("Spatial Feature Extraction", tag="Proposed")
sd.add(Text(L, TOP, 6.6, 0.4, [heading("Face Frame  →  CNN  →  Spatial Embedding")]))
sd.add(Rect(L, 2.15, 1.8, 0.95, fill=BLUE_LIGHT, stroke=LINE, paras=[
    Para("Face frame", bold=True, color=NAVY, space_after=2), Para("RGB crop, e.g. 224 × 224", size=9.5, color=MUTED)],
    size=11.5))
sd.add(arrow(2.4, 2.62, 2.85, 2.62))
sd.add(stage(2.85, 2.15, 2.3, 0.95, "Pretrained CNN / Vision Backbone", fill=BLUE))
sd.add(arrow(5.15, 2.62, 5.6, 2.62))
sd.add(Rect(5.6, 2.15, 1.6, 0.95, fill=NAVY, paras=[
    Para("Spatial embedding", bold=True, color=WHITE, space_after=2), Para("feature vector, per frame", size=9.5, color=blend(NAVY, WHITE, 0.7))],
    size=11.5))
sd.add(Text(L, 3.25, 6.6, 0.6, [
    "Backbone: final architecture to be selected during implementation. An ImageNet-pretrained "
    "ResNet18 is the initial candidate in the project implementation plan, kept configurable."],
    size=9.5, color=AMBER))
sd.add(soft(L, 4.0, 6.6, 2.1, [
    heading("Why spatial information alone may be insufficient", size=12.5),
    bullet("Spatial artifacts are generator- and dataset-specific: a model can over-fit to them and fail on unseen manipulation methods.", size=11.5, space_after=5),
    bullet("Compression, resizing and post-processing weaken fine-grained pixel cues.", size=11.5, space_after=5),
    bullet("A single frame carries no information about motion or consistency over time.", size=11.5),
]))
rx0 = 7.6
sd.add(Table(rx0, TOP, R - rx0, [1.1, 3.0], [
    ["Input", "One face frame (RGB image)"],
    ["Processing", "CNN / vision backbone, pretrained on natural images; classification layer removed"],
    ["Output", "Fixed-length spatial embedding describing the visual characteristics of the frame"],
], size=11, header=False, first_col_bold=True, min_row_h=0.55))
sd.add(Text(rx0, 3.6, R - rx0, 3.2, [
    heading("Examples of information the embedding may capture", size=12.5),
    bullet("Texture of skin and hair regions", size=12, space_after=4),
    bullet("Edges and contour sharpness", size=12, space_after=4),
    bullet("Facial structure and symmetry", size=12, space_after=4),
    bullet("Blending artifacts at face boundaries", size=12, space_after=4),
    bullet("Local inconsistencies in colour and lighting", size=12),
]))

# ================================================================ 11. Frequency
sd = base("Frequency-Domain Feature Extraction", tag="Proposed")
sd.add(Text(L, TOP, 6.5, 3.4, [
    heading("Concept", size=13),
    bullet("An image can be represented in the spatial domain (pixels) or in the frequency domain.", space_after=5),
    bullet("The frequency representation describes how rapidly pixel intensities vary across the image.", space_after=5),
    bullet("Low-frequency components represent smooth structures; high-frequency components capture fine detail and rapid changes.", space_after=5),
    bullet("Deepfake generation and post-processing (up-sampling, blending, re-compression) **may** introduce abnormal frequency patterns.", space_after=5),
    bullet("An FFT or DCT transforms a face frame into a frequency representation (e.g. a log-magnitude spectrum).", space_after=5),
    bullet("A neural network can then learn which frequency patterns are discriminative."),
], size=12))
sd.add(Rect(L, 5.05, 3.15, 1.15, fill=BLUE_LIGHT, stroke=LINE, paras=[
    Para("Low frequency", bold=True, color=NAVY, space_after=2),
    Para("Smooth regions, overall shape and shading", size=10.5)], size=11.5, align="l", inset=0.12))
sd.add(Rect(L + 3.35, 5.05, 3.15, 1.15, fill=TEAL_LIGHT, stroke=LINE, paras=[
    Para("High frequency", bold=True, color=NAVY, space_after=2),
    Para("Edges, texture, noise, up-sampling traces", size=10.5)], size=11.5, align="l", inset=0.12))
sd.add(caption(L, 6.3, 6.5, "Frequency cues can provide additional discriminative information, but "
               "they are not assumed to be present in every deepfake.", size=9.5, color=AMBER))
fx0, fw0 = 8.0, 4.2
steps = [("Face frame", "RGB face crop (same input as the spatial branch)", BLUE_LIGHT, NAVY),
         ("FFT / DCT", "2-D transform; zero frequency shifted to the centre", NAVY, WHITE),
         ("Frequency representation", "log(1 + magnitude), normalized", TEAL_LIGHT, NAVY),
         ("Neural feature extractor", "lightweight CNN on the spectrum image", NAVY, WHITE),
         ("Frequency embedding", "fixed-length vector, per frame", TEAL, WHITE)]
for i, (t, d, f, c) in enumerate(steps):
    y = TOP + 0.05 + i * 0.98
    sd.add(Rect(fx0, y, fw0, 0.68, fill=f, stroke=LINE if f in (BLUE_LIGHT, TEAL_LIGHT) else None,
                paras=[Para(t, bold=True, size=11.5, space_after=1),
                       Para(d, size=9.5, color=c if c == WHITE else MUTED)], color=c))
    if i < len(steps) - 1:
        sd.add(arrow(fx0 + fw0 / 2, y + 0.68, fx0 + fw0 / 2, y + 0.98))

# ================================================================ 12. Temporal
sd = base("Temporal Feature Modeling", tag="Proposed")
sd.add(Text(L, 1.35, 6.4, 0.3, ["A video is a sequence of frames; each frame yields a feature vector"],
            size=10.5, color=MUTED))
fw_, fh_ = 1.1, 0.65
names = ["Frame 1", "Frame 2", "Frame 3", "…", "Frame N"]
for i, n in enumerate(names):
    x = L + i * 1.35
    if n == "…":
        sd.add(Text(x, 1.75, fw_, fh_, ["· · ·"], size=14, color=MUTED, align="c", valign="m"))
        sd.add(Text(x, 2.62, fw_, 0.4, ["· · ·"], size=12, color=MUTED, align="c", valign="m"))
        continue
    sd.add(Rect(x, 1.75, fw_, fh_, fill=WHITE, stroke=BLUE, paras=[n], size=11, color=NAVY, bold=True))
    sd.add(arrow(x + fw_ / 2, 2.4, x + fw_ / 2, 2.62))
    lbl = "f" + ("N" if n == "Frame N" else n[-1])
    sd.add(Rect(x + 0.2, 2.62, fw_ - 0.4, 0.4, fill=BLUE_LIGHT, paras=[lbl], size=10.5, color=NAVY, bold=True))
    if i < 4:
        sd.add(arrow(x + fw_, 1.75 + fh_ / 2, x + 1.35, 1.75 + fh_ / 2))
sd.add(Text(L, 3.06, 6.5, 0.3, ["frame-level feature vectors (spatial + frequency, fused per frame)"],
            size=9.5, color=MUTED, align="c"))
sd.add(arrow(L + 3.25, 3.4, L + 3.25, 3.7, color=NAVY, w=1.5))
sd.add(Rect(L, 3.7, 6.5, 0.95, fill=NAVY, paras=[
    Para("Transformer Encoder", bold=True, size=13, space_after=2),
    Para("positional encoding  +  multi-head self-attention across the frame sequence", size=10)], color=WHITE))
sd.add(arrow(L + 3.25, 4.65, L + 3.25, 4.95, color=NAVY, w=1.5))
sd.add(Rect(L, 4.95, 6.5, 0.8, fill=TEAL_LIGHT, stroke=TEAL, paras=[
    Para("Contextualized frame representations", bold=True, color=NAVY, size=12, space_after=2),
    Para("each frame's representation now depends on every other frame  →  temporal inconsistencies become visible", size=10, color=TEXT)]))
sd.add(caption(L, 5.9, 6.5, "Each element attends to all others, so the model can relate distant "
               "frames directly.", size=9.5))
rx1 = 7.6
sd.add(Text(rx1, TOP, R - rx1, 2.5, [
    heading("Why a Transformer encoder rather than a recurrent model (proposed)", size=12.5),
    bullet("Self-attention models relationships between distant frames directly, without passing information step by step.", size=11.5, space_after=5),
    bullet("Sequence elements are processed in parallel during training.", size=11.5, space_after=5),
    bullet("It provides a flexible mechanism for learning temporal dependencies and for attention-based fusion in the same framework.", size=11.5),
]))
sd.add(soft(rx1, 4.2, R - rx1, 1.95, [
    Para("NOT A LANGUAGE MODEL", size=10, bold=True, color=AMBER, space_after=6),
    Para("Transformer ≠ LLM. The encoder used here has no tokenizer, no vocabulary and no language "
         "objective. It is a standard sequence model (e.g. PyTorch's Transformer encoder) applied to "
         "numeric frame embeddings, as in vision and video transformers [6], [7].", size=11.5)],
    fill=AMBER_LIGHT, stroke=AMBER))

# ================================================================ 13. Fusion
sd = base("Attention-Based Feature Fusion", tag="Proposed")
srcs = [("Spatial features", "visual evidence", NAVY), ("Frequency features", "hidden signal-level evidence", TEAL),
        ("Temporal features", "sequence / motion evidence", BLUE)]
for i, (t, d, f) in enumerate(srcs):
    y = 1.75 + i * 1.1
    sd.add(Rect(L, y, 2.5, 0.8, fill=f, paras=[Para(t, bold=True, size=12, space_after=1),
                                                   Para(d, size=9.5)], color=WHITE))
    sd.add(arrow(L + 2.5, y + 0.4, 3.75, 3.25))
sd.add(Rect(3.75, 2.35, 2.1, 1.8, fill=WHITE, stroke=NAVY, stroke_w=1.5, paras=[
    Para("Attention / Transformer Fusion", bold=True, color=NAVY, size=12, space_after=3),
    Para("input-dependent weighting of evidence", size=9.5, color=MUTED)]))
sd.add(arrow(5.85, 3.25, 6.3, 3.25))
sd.add(Rect(6.3, 2.8, 1.35, 0.9, fill=NAVY, paras=[Para("Fused representation", bold=True, size=10.5)], color=WHITE))
sd.add(caption(6.3, 3.75, 1.5, "→ classifier", size=9.5, align="c"))
sd.add(soft(L, 5.2, 7.05, 1.5, [
    Para("PROPOSED MECHANISM", size=10, bold=True, color=TEAL, space_after=4),
    Para("Spatial and frequency embeddings are fused per frame; the Transformer encoder's self-attention "
         "then lets frame representations interact across the sequence; a video-level representation is "
         "obtained by aggregation (mean pooling or attention-weighted, to be finalized during implementation).",
         size=10.5)], fill=TEAL_LIGHT, stroke=TEAL))
rx2 = 8.0
sd.add(Text(rx2, TOP, R - rx2, 1.6, [
    heading("Why attention rather than fixed weighting", size=12.5),
    Para("Attention allows the model to learn which information source is more relevant for a "
         "particular input, instead of assigning identical importance to every feature source.", size=11.5)]))
sd.add(Text(rx2, 3.25, R - rx2, 0.35, [heading("Intuition: an illustrative example (not a result)", size=11.5)]))
sd.add(Table(rx2, 3.65, R - rx2, [1.3, 1.9, 1.5], [
    ["Evidence source", "Observation for this input", "Attention weight"],
    ["Spatial", "appearance mostly normal", "lower"],
    ["Temporal", "suspicious motion pattern", "higher"],
    ["Frequency", "suspicious spectral pattern", "higher"],
], size=10.5, first_col_bold=True, min_row_h=0.38, align=["l", "l", "c"]))
sd.add(Text(rx2, 5.4, R - rx2, 1.3, [
    "For this example the fusion mechanism can learn that the temporal and frequency evidence is "
    "more informative, and rely on it more than on the spatial evidence."], size=11, color=NAVY))

# ================================================================ 14. Datasets
sd = base("Dataset & Preprocessing Strategy", tag="Planned", kind="planned")
dsets = [
    ("FaceForensics++ [1]", "PRIMARY  ·  training and in-domain evaluation", TEAL,
     "Standard benchmark for deepfake detection research: real videos plus several manipulation methods, enabling reproducible comparison."),
    ("Celeb-DF [2]", "GENERALIZATION  ·  cross-dataset evaluation, if feasible", BLUE,
     "Different deepfake characteristics from the training data; used only for testing (no fine-tuning) to probe generalization."),
    ("DFDC [3]", "OPTIONAL  ·  subject to available compute and time", AMBER,
     "Very large and computationally expensive to preprocess and evaluate; considered only after the above are complete."),
]
for i, (n, role, col, why) in enumerate(dsets):
    y = TOP + i * 1.72
    sd.add(Rect(L, y, 6.4, 1.55, fill=WHITE, stroke=LINE))
    sd.add(Rect(L, y, 0.08, 1.55, fill=col))
    sd.add(Text(L + 0.22, y + 0.1, 6.05, 1.4, [
        Para(n, size=13.5, bold=True, color=NAVY, space_after=2),
        Para(role, size=9.5, bold=True, color=col, space_after=5),
        Para(why, size=10.5)]))
px0 = 7.5
sd.add(Text(px0, TOP - 0.05, R - px0, 0.35, [heading("Preprocessing pipeline (per video)", size=12.5)]))
steps = ["Video loading", "Frame sampling (N frames, evenly spaced)", "Face detection",
         "Face alignment / cropping", "Resize to a fixed resolution", "Normalization",
         "Sequence construction (N face crops in order)", "Label assignment"]
for i, s in enumerate(steps):
    y = 1.95 + i * 0.43
    sd.add(Rect(px0, y, 0.36, 0.34, fill=NAVY, paras=[str(i + 1)], size=10, color=WHITE, bold=True))
    sd.add(Text(px0 + 0.45, y, R - px0 - 0.45, 0.34, [s], size=11.5, valign="m"))
sd.add(soft(px0, 5.5, R - px0, 1.25, [
    Para("Labels: REAL = 0,  FAKE = 1  (assigned per video)", bold=True, color=NAVY, size=11, space_after=3),
    Para("Frames inherit the label of their video; the model predicts at video level from the frame sequence. "
         "Train / validation / test splits are formed at video level so frames of one video never appear in two splits.",
         size=10)], fill=BLUE_LIGHT, stroke=LINE, inset=0.12))

# ================================================================ 15. Implementation plan
sd = base("Implementation Plan", tag="Planned", kind="planned")
phases = [
    ("Dataset acquisition & preprocessing", "video loading, frame sampling, face extraction, dataset/loader", "In progress"),
    ("Baseline CNN detector", "frame-level classifier on pretrained backbone; validates the data pipeline", "Planned"),
    ("Frequency branch", "FFT/DCT → log-magnitude → lightweight CNN embedding", "Planned"),
    ("Temporal Transformer", "positional encoding + Transformer encoder over frame sequences", "Planned"),
    ("Attention-based feature fusion", "per-frame fusion + attention-weighted aggregation", "Planned"),
    ("Training & hyperparameter tuning", "loss, optimizer, validation-based model selection", "Planned"),
    ("Evaluation & cross-dataset testing", "metrics, ablations, Celeb-DF transfer if feasible", "Planned"),
    ("Demo / deployment", "optional minimal Streamlit or FastAPI interface", "Optional"),
]
pw, ph = 2.85, 1.5
for i, (t, d, st) in enumerate(phases):
    x = L + (i % 4) * (pw + 0.25)
    y = TOP + (i // 4) * (ph + 0.25)
    sd.add(Rect(x, y, pw, ph, fill=WHITE, stroke=LINE))
    sd.add(Rect(x, y, pw, 0.34, fill=NAVY, paras=[f"PHASE {i + 1}"], size=9, color=WHITE, bold=True, align="l", inset=0.12))
    stc = {"In progress": TEAL, "Planned": BLUE, "Optional": AMBER}[st]
    sd.add(Text(x + pw - 1.1, y, 1.05, 0.34, [st.upper()], size=8, bold=True, color=WHITE, align="r", valign="m"))
    sd.add(Rect(x + pw - 0.08, y, 0.08, 0.34, fill=stc))
    sd.add(Text(x + 0.05, y + 0.4, pw - 0.1, ph - 0.45, [
        Para(t, size=11.5, bold=True, color=NAVY, space_after=3), Para(d, size=9.5, color=TEXT)]))
    if i % 4 != 3:
        sd.add(arrow(x + pw, y + ph / 2, x + pw + 0.25, y + ph / 2))
ty = TOP + 2 * (ph + 0.25) + 0.05
sd.add(Rect(L, ty, CW, 1.6, fill=PANEL, stroke=LINE))
sd.add(Text(L + 0.2, ty + 0.12, 2.2, 0.6, [Para("Core stack", bold=True, color=NAVY, size=11.5),
                                            Para("confirmed", size=9.5, color=TEAL, bold=True)], valign="m"))
core = ["Python", "PyTorch", "OpenCV", "NumPy", "SciPy (where required)", "scikit-learn"]
for i, c in enumerate(core):
    sd.add(Rect(L + 2.5 + i * 1.62, ty + 0.2, 1.5, 0.45, fill=WHITE, stroke=NAVY, paras=[c], size=10, color=NAVY, bold=True))
sd.add(Text(L + 0.2, ty + 0.85, 2.2, 0.6, [Para("Optional tools", bold=True, color=NAVY, size=11.5),
                                            Para("only if actually used", size=9.5, color=AMBER, bold=True)], valign="m"))
opt = ["Streamlit / FastAPI (demo interface)", "TensorBoard / Weights & Biases (tracking)",
       "Face detector library (configurable)"]
for i, c in enumerate(opt):
    sd.add(Rect(L + 2.5 + i * 3.25, ty + 0.93, 3.1, 0.45, fill=WHITE, stroke=LINE, paras=[c], size=10, color=MUTED))

# ================================================================ 16. Training
sd = base("Training & Experimental Design", tag="Planned", kind="planned")
steps = ["Video", "Sampled frames", "Face crops", "Feature extraction", "Temporal / fusion model",
         "REAL / FAKE prediction", "Loss calculation", "Backward pass", "Parameter update"]
n = len(steps)
gap_, sw_ = 0.2, (CW - 8 * 0.2) / 9
for i, s in enumerate(steps):
    x = L + i * (sw_ + gap_)
    f = NAVY if i < 6 else BLUE
    sd.add(stage(x, TOP + 0.1, sw_, 0.75, s, fill=f, size=9.5))
    if i < n - 1:
        sd.add(arrow(x + sw_, TOP + 0.475, x + sw_ + gap_, TOP + 0.475, w=1))
sd.add(Line(R - sw_ / 2, TOP + 0.85, R - sw_ / 2, TOP + 1.1, color=MUTED, width=1))
sd.add(Line(R - sw_ / 2, TOP + 1.1, L + sw_ / 2, TOP + 1.1, color=MUTED, width=1))
sd.add(arrow(L + sw_ / 2, TOP + 1.1, L + sw_ / 2, TOP + 0.85, w=1))
sd.add(Text(L, TOP + 1.15, CW, 0.3, ["backpropagation of the loss, then a parameter update; repeated over mini-batches and epochs, with validation after each epoch"],
            size=9.5, color=MUTED, align="c"))
sd.add(Text(L, 3.25, 6.2, 3.5, [
    heading("Training components (planned)", size=12.5),
    bullet("Binary cross-entropy loss on the video-level logit (REAL = 0, FAKE = 1)", size=11.5, space_after=5),
    bullet("Adam / AdamW optimizer", size=11.5, space_after=5),
    bullet("Transfer learning: pretrained backbone for the spatial branch where appropriate", size=11.5, space_after=5),
    bullet("Held-out validation set for model selection and hyperparameter choices", size=11.5, space_after=5),
    bullet("Early stopping / checkpointing of the best validation model, if implemented", size=11.5, space_after=5),
    bullet("Modest augmentation (flip, small rotation, mild colour jitter) applied before the frequency transform", size=11.5),
]))
ex0 = 7.2
sd.add(Text(ex0, 3.25, R - ex0, 0.4, [heading("Experiment design: Baseline vs. Proposed", size=12.5)]))
sd.add(Rect(ex0, 3.75, 2.65, 2.0, fill=PANEL, stroke=LINE, paras=[
    Para("BASELINE", size=10, bold=True, color=MUTED, space_after=4),
    Para("CNN-based detector", bold=True, color=NAVY, size=12, space_after=4),
    Para("Frame-level classification; per-frame outputs averaged over the sampled frames. No frequency branch, no temporal model.", size=10)],
    align="l", valign="t", inset=0.14))
sd.add(Rect(ex0 + 2.9, 3.75, 2.65, 2.0, fill=TEAL_LIGHT, stroke=TEAL, paras=[
    Para("PROPOSED", size=10, bold=True, color=TEAL, space_after=4),
    Para("Hybrid model", bold=True, color=NAVY, size=12, space_after=4),
    Para("Spatial + frequency + temporal Transformer with attention-based fusion; one prediction per video.", size=10)],
    align="l", valign="t", inset=0.14))
sd.add(Text(ex0, 5.9, R - ex0, 0.8, [
    "Both models are trained and evaluated on identical video-level splits with the same loss and "
    "metrics, so results are directly comparable."], size=10.5, color=NAVY))

# ================================================================ 17. Evaluation
sd = base("Evaluation Framework", tag="Results not yet available", kind="caution")
metrics = [("Accuracy", "share of videos classified correctly"),
           ("Precision", "of videos flagged FAKE, how many are fake"),
           ("Recall", "of fake videos, how many are flagged"),
           ("F1-score", "harmonic mean of precision and recall"),
           ("ROC-AUC", "threshold-independent separability, where appropriate")]
mw = (CW - 4 * 0.2) / 5
for i, (m, d) in enumerate(metrics):
    x = L + i * (mw + 0.2)
    sd.add(Rect(x, TOP, mw, 0.85, fill=WHITE, stroke=LINE, paras=[
        Para(m, bold=True, color=NAVY, size=12, space_after=2), Para(d, size=9.5, color=MUTED)], inset=0.08))
hdr = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
ph_ = "—"
sd.add(Table(L, 2.7, CW, [2.6, 1, 1, 1, 1, 1], [
    hdr, ["Baseline CNN"] + [ph_] * 5, ["Spatial + Frequency"] + [ph_] * 5,
    ["Spatial + Temporal"] + [ph_] * 5, ["Full Proposed Model"] + [ph_] * 5],
    size=11, first_col_bold=True, min_row_h=0.36, align=["l", "c", "c", "c", "c", "c"]))
sd.add(Text(L, 4.55, CW, 0.35, [
    "In-domain evaluation on the FaceForensics++ held-out test split, video-level predictions. "
    "**Results will be populated after experimental evaluation.**"], size=10, color=AMBER))
sd.add(Text(L, 5.0, CW, 0.3, [heading("Cross-dataset evaluation (if implemented)", size=11.5)]))
sd.add(Table(L, 5.35, CW, [1.7, 3.3, 1, 1, 1, 1, 1], [
    ["Train", "Test", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    ["FaceForensics++", "FaceForensics++ (in-domain)", ph_, ph_, ph_, ph_, ph_],
    ["FaceForensics++", "Celeb-DF (cross-dataset, no fine-tuning)", ph_, ph_, ph_, ph_, ph_]],
    size=10.5, min_row_h=0.34, align=["l", "l", "c", "c", "c", "c", "c"]))
sd.add(caption(L, 6.5, CW, "Every reported number will be accompanied by the model checkpoint, "
               "data split and configuration used to produce it.", size=9.5))

# ================================================================ 18. Ablation
sd = base("Ablation Study", tag="Planned", kind="planned")
sd.add(Text(L, 1.32, 11, 0.3, ["Each component is added in turn so that its individual contribution "
                                "can be measured on the same data split."], size=10.5, color=MUTED))
Y, N_ = "Yes", "–"
sd.add(Table(L, 1.75, CW, [0.6, 2.4, 0.9, 1.0, 0.95, 1.4, 3.6], [
    ["Exp.", "Configuration", "Spatial", "Frequency", "Temporal", "Attention-based fusion", "What it isolates"],
    ["A", "Spatial only", Y, N_, N_, N_, "Value of the spatial branch alone (frame features averaged)"],
    ["B", "Spatial + Frequency", Y, Y, N_, N_, "Contribution of frequency-domain features"],
    ["C", "Spatial + Temporal", Y, N_, Y, N_, "Contribution of temporal modeling"],
    ["D", "Spatial + Frequency + Temporal", Y, Y, Y, N_, "Complementarity of all three cues with simple aggregation"],
    ["E", "Full model with Transformer-based fusion", Y, Y, Y, Y, "Added value of attention-based fusion over simple aggregation"],
], size=10.5, first_col_bold=True, min_row_h=0.42, alt_fill=PANEL,
    align=["c", "l", "c", "c", "c", "c", "l"]))
sd.add(soft(L, 5.05, 5.9, 1.45, [
    heading("Purpose", size=12),
    Para("Determine whether each component actually contributes to performance, rather than assuming "
         "that a more complex model is better. Components that do not help will be reported as such.", size=11)]))
sd.add(soft(L + 6.2, 5.05, CW - 6.2, 1.45, [
    heading("Protocol", size=12),
    Para("Identical video-level splits, training schedule and metrics for every configuration; "
         "results reported on the held-out test split. Configurations D and E differ only in how "
         "frame representations are aggregated.", size=11)]))

# ================================================================ 19. Expected contribution
sd = base("Expected Contribution", tag="Expected outcome  ·  research objectives", kind="caution")
sd.add(Text(L, 1.32, 11, 0.3, ["The project aims to investigate whether hybrid feature learning can:"],
            size=11.5, color=MUTED))
objs = [("Improve robustness", "to compression, quality variation and other nuisance factors"),
        ("Capture complementary forgery cues", "appearance, motion and spectral evidence in one model"),
        ("Reduce dependence on a single feature type", "so that failure of one cue does not fail the detector"),
        ("Improve generalization", "to different manipulation distributions, tested cross-dataset"),
        ("Provide a stronger research baseline", "with ablations that future work can build on")]
ow = (CW - 4 * 0.2) / 5
for i, (t, d) in enumerate(objs):
    x = L + i * (ow + 0.2)
    sd.add(Rect(x, 1.75, ow, 2.1, fill=WHITE, stroke=LINE))
    sd.add(Rect(x, 1.75, ow, 0.06, fill=TEAL))
    sd.add(Text(x + 0.05, 1.95, ow - 0.1, 1.85, [
        Para(f"OBJECTIVE {i + 1}", size=9, bold=True, color=TEAL, space_after=6),
        Para(t, size=12.5, bold=True, color=NAVY, space_after=6),
        Para(d, size=10.5, color=TEXT)]))
sd.add(soft(L, 4.25, CW, 1.9, [
    Para("WHAT THIS PROJECT DOES NOT CLAIM", size=10, bold=True, color=AMBER, space_after=6),
    bullet("It does not claim to “solve” deepfakes; detection is an evolving problem as generation methods change.", size=11, space_after=4),
    bullet("It does not claim state-of-the-art performance; any such claim requires experimental evidence that does not yet exist.", size=11, space_after=4),
    bullet("Improvements listed above are hypotheses. They will be confirmed or rejected by the planned experiments and ablations.", size=11)],
    fill=AMBER_LIGHT, stroke=AMBER))

# ================================================================ 20. Limitations
sd = base("Limitations & Challenges")
lims = [
    ("Computational requirements", "Video preprocessing and sequence models are expensive; frame count, resolution and dataset size must be budgeted against available GPU time."),
    ("Dataset bias", "Benchmarks share actors, capture set-ups and generators; a model may learn these regularities instead of forgery cues."),
    ("Compression and video quality variations", "Re-encoding attenuates high-frequency artifacts and can invalidate cues learned on higher-quality data."),
    ("Generalization to unseen manipulation methods", "New generators may leave different traces; cross-dataset testing probes but cannot guarantee transfer."),
    ("False positives / false negatives", "Both error types carry real cost (wrongly discrediting genuine footage vs. missing a forgery); thresholds must be reported, not hidden."),
    ("Highly sophisticated deepfakes", "Some forgeries may leave few detectable traces in any of the three signals."),
    ("Dependence on face detection / alignment", "Detector failures, occlusion, profile views and multiple faces propagate errors into every downstream branch."),
]
lw = (CW - 0.3) / 2
for i, (t, d) in enumerate(lims):
    x = L + (i % 2) * (lw + 0.3)
    y = TOP + (i // 2) * 1.3
    sd.add(Rect(x, y, lw, 1.15, fill=PANEL, stroke=LINE))
    sd.add(Rect(x, y, 0.07, 1.15, fill=BLUE))
    sd.add(Text(x + 0.2, y + 0.08, lw - 0.35, 1.05, [
        Para(t, bold=True, color=NAVY, size=12, space_after=3), Para(d, size=10)], valign="m"))

# ================================================================ 21. Future scope
sd = base("Future Scope")
sd.add(Text(L, 1.32, 11, 0.3, ["Directions beyond the current project scope; none are claimed as part of this work."],
            size=10.5, color=MUTED))
fut = [
    ("Audio-visual detection", "Combine lip-sync and voice consistency with the visual branches."),
    ("Multimodal models", "Joint reasoning over video, audio and metadata."),
    ("Real-time detection", "Streaming inference with bounded latency per frame window."),
    ("Lightweight edge deployment", "Compressed models for mobile and on-device screening."),
    ("Adversarial robustness", "Resistance to perturbations designed to fool detectors."),
    ("Explainable AI", "Attention and saliency maps to show which frames or regions drove a decision."),
    ("Emerging generation techniques", "Diffusion-based and other new synthesis methods."),
    ("Larger cross-dataset evaluation", "Broader benchmarks, including DFDC where compute allows."),
]
fw2, fh2 = (CW - 3 * 0.25) / 4, 1.95
for i, (t, d) in enumerate(fut):
    x = L + (i % 4) * (fw2 + 0.25)
    y = 1.75 + (i // 4) * (fh2 + 0.25)
    sd.add(Rect(x, y, fw2, fh2, fill=WHITE, stroke=LINE))
    sd.add(Text(x + 0.15, y + 0.15, fw2 - 0.3, fh2 - 0.3, [
        Para(f"0{i + 1}", size=16, bold=True, color=TEAL, space_after=6),
        Para(t, size=12.5, bold=True, color=NAVY, space_after=5),
        Para(d, size=10.5)]))

# ================================================================ 22. Conclusion
sd = base("Conclusion")
sd.add(Text(L, TOP, CW, 3.3, [
    bullet("Deepfake detection is an important computer vision and cybersecurity problem.", size=15, space_after=10),
    bullet("Existing detectors can perform well in-domain yet struggle to generalize to unseen data.", size=15, space_after=10),
    bullet("The proposed system combines spatial, temporal and frequency-domain information.", size=15, space_after=10),
    bullet("Transformer-based attention is used for sequence modeling and feature fusion.", size=15, space_after=10),
    bullet("The project will evaluate, through ablations and cross-dataset tests, whether this hybrid architecture improves robustness.", size=15),
], color=TEXT))
sd.add(Rect(0, 5.05, SLIDE_W, 1.65, fill=NAVY))
sd.add(Rect(L, 5.55, 0.08, 0.65, fill=TEAL))
sd.add(Text(L + 0.3, 5.05, CW - 0.3, 1.65, [
    Para("“From detecting how a face looks, to how it moves, to the hidden patterns beneath it.”",
         size=19, bold=True, color=WHITE, italic=True)], valign="m"))

# ================================================================ 23-24. References
refs = [
    "A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “FaceForensics++: Learning to Detect Manipulated Facial Images,” in Proc. IEEE/CVF Int. Conf. on Computer Vision (ICCV), 2019.",
    "Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics,” in Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2020.",
    "B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, and C. C. Ferrer, “The DeepFake Detection Challenge (DFDC) Dataset,” arXiv:2006.07397, 2020.",
    "D. Afchar, V. Nozick, J. Yamagishi, and I. Echizen, “MesoNet: A Compact Facial Video Forgery Detection Network,” in Proc. IEEE Int. Workshop on Information Forensics and Security (WIFS), 2018.",
    "K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016.",
    "A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” in Proc. Int. Conf. on Learning Representations (ICLR), 2021.",
    "A. Vaswani et al., “Attention Is All You Need,” in Advances in Neural Information Processing Systems (NeurIPS), 2017.",
    "Y. Qian, G. Yin, L. Sheng, Z. Chen, and J. Shao, “Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues,” in Proc. European Conf. on Computer Vision (ECCV), 2020.",
    "R. Durall, M. Keuper, and J. Keuper, “Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions,” in Proc. IEEE/CVF CVPR, 2020.",
    "J. Frank, T. Eule, A. Fischer, K. Rieck, D. Kolossa, and T. Holz, “Leveraging Frequency Analysis for Deep Fake Image Recognition,” in Proc. Int. Conf. on Machine Learning (ICML), 2020.",
    "E. Sabir, J. Cheng, A. Jaiswal, W. AbdAlmageed, I. Masi, and P. Natarajan, “Recurrent Convolutional Strategies for Face Manipulation Detection in Videos,” in Proc. IEEE/CVF CVPR Workshops, 2019.",
    "D. Güera and E. J. Delp, “Deepfake Video Detection Using Recurrent Neural Networks,” in Proc. IEEE Int. Conf. on Advanced Video and Signal Based Surveillance (AVSS), 2018.",
    "Y. Zheng, J. Bao, D. Chen, M. Zeng, and F. Wen, “Exploring Temporal Coherence for More General Video Face Forgery Detection,” in Proc. IEEE/CVF ICCV, 2021.",
    "Y. Li, M.-C. Chang, and S. Lyu, “In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking,” in Proc. IEEE WIFS, 2018.",
    "L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen, and B. Guo, “Face X-ray for More General Face Forgery Detection,” in Proc. IEEE/CVF CVPR, 2020.",
    "D. A. Coccomini, N. Messina, C. Gennaro, and F. Falchi, “Combining EfficientNet and Vision Transformers for Video Deepfake Detection,” in Proc. Int. Conf. on Image Analysis and Processing (ICIAP), 2022.",
    "H. Zhao, W. Zhou, D. Chen, T. Wei, W. Zhang, and N. Yu, “Multi-Attentional Deepfake Detection,” in Proc. IEEE/CVF CVPR, 2021.",
    "R. Tolosana, R. Vera-Rodriguez, J. Fierrez, A. Morales, and J. Ortega-Garcia, “DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection,” Information Fusion, vol. 64, pp. 131–148, 2020.",
    "L. Verdoliva, “Media Forensics and DeepFakes: An Overview,” IEEE Journal of Selected Topics in Signal Processing, vol. 14, no. 5, pp. 910–932, 2020.",
    "S.-Y. Wang, O. Wang, R. Zhang, A. Owens, and A. A. Efros, “CNN-Generated Images Are Surprisingly Easy to Spot… For Now,” in Proc. IEEE/CVF CVPR, 2020.",
]
for part, chunk in enumerate((refs[:10], refs[10:])):
    sd = base(f"References ({part + 1}/2)")
    start = part * 10
    paras = []
    for i, r in enumerate(chunk):
        paras.append(Para(f"[{start + i + 1}]  {r}", size=11.5, space_after=8))
    sd.add(Text(L, TOP - 0.1, CW, 5.4, paras, line_spacing=1.1))
    if part == 1:
        sd.add(caption(L, 6.55, CW, "Citation style: IEEE. Bracketed numbers are used on the Related Work, "
                       "Temporal Modeling and Dataset slides.", size=9.5))

# ================================================================ 25. Q&A
sd = Slide(bg=NAVY)
slides.append(sd)
sd.add(Text(0.8, 2.3, 8.5, 1.2, ["Questions & Discussion"], size=40, bold=True, color=WHITE, valign="m"))
sd.add(Rect(0.85, 3.6, 0.7, 0.04, fill=TEAL))
sd.add(Text(0.8, 3.8, 8.5, 0.5, ["Thank you"], size=16, color=TEAL, valign="m"))
sd.add(Text(0.8, 4.5, 8.5, 1.0, [Para(TITLE, size=11, color=blend(NAVY, WHITE, 0.6), space_after=6),
                                   Para("   ·   ".join(TEAM), size=11, color=blend(NAVY, WHITE, 0.6))]))
spectrum_grid(sd, 10.2, 2.35, n=7, cell=0.3, gap=0.05)


# ---------------------------------------------------------------- finalize
def finalize():
    total = len(slides)
    for i, sd in enumerate(slides):
        slot = getattr(sd, "_page_slot", None)
        if slot is not None:
            slot.paras = [f"{i + 1} / {total}"]


if __name__ == "__main__":
    finalize()
    out = os.path.dirname(os.path.abspath(__file__))
    base_name = "DeepFake_Detection_FYP_Presentation"
    render_pptx(slides, os.path.join(out, base_name + ".pptx"))
    render_pdf(slides, os.path.join(out, base_name + ".pdf"), title=TITLE)
    print(f"{len(slides)} slides written to {out}")
