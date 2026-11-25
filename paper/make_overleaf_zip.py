"""Package the paper for upload to Overleaf.

Run from the repository root:
    python paper/make_overleaf_zip.py

Writes paper_overleaf.zip with main.tex, figures/*.png and data/*.csv. Paths
inside the zip use forward slashes: a zip built with Windows backslashes
unpacks on Overleaf as files literally named "figures\\x.png", so every image
and plot is missing and references compile to "??".
"""

import glob
import os
import zipfile

PAPER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(PAPER), "paper_overleaf.zip")

files = ["main.tex"] + sorted(glob.glob("figures/*.png", root_dir=PAPER)) \
    + sorted(glob.glob("data/*.csv", root_dir=PAPER))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in files:
        zf.write(os.path.join(PAPER, name), arcname=name.replace(os.sep, "/"))

with zipfile.ZipFile(OUT) as zf:
    names = zf.namelist()
assert all("\\" not in n for n in names), names
for needed in ("figures/roc_curves.png", "figures/confusion_matrices.png",
               "figures/fft_examples.png", "figures/frame_sequence.png",
               "data/dfdc_hybrid.csv", "data/dfdc_baseline_5ep.csv"):
    assert needed in names, needed
print(f"wrote {OUT}:", ", ".join(names))
