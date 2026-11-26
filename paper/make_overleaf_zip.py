"""Package the paper for upload to Overleaf.

Run from the repository root:
    python paper/make_overleaf_zip.py

Writes paper_overleaf.zip with main.tex and the figures it includes. Paths
inside the zip use forward slashes: a zip built with Windows backslashes
unpacks on Overleaf as files literally named "figures\\x.png", so every image
and plot is missing and references compile to "??".
"""

import os
import re
import zipfile

PAPER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(PAPER), "paper_overleaf.zip")

with open(os.path.join(PAPER, "main.tex"), encoding="utf-8") as fh:
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", fh.read())
files = ["main.tex"] + figures
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in files:
        zf.write(os.path.join(PAPER, name), arcname=name.replace(os.sep, "/"))

with zipfile.ZipFile(OUT) as zf:
    names = zf.namelist()
assert all("\\" not in n for n in names), names
assert figures and all(f in names for f in figures), (figures, names)
print(f"wrote {OUT}:", ", ".join(names))
