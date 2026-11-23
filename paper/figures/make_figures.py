"""Build the paper's image figures from cached DFDC face crops.

Run from the repository root after preprocessing:
    python paper/figures/make_figures.py

Writes paper/figures/fft_examples.png and paper/figures/frame_sequence.png.
The spectra come from models.frequency.fft_log_magnitude, the exact transform
the network uses, so the figure shows what the frequency branch actually sees.
"""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from models.frequency import fft_log_magnitude  # noqa: E402
from preprocessing.dataset import normalise  # noqa: E402

CACHE = "data/processed/{}_n8_s224_haar.npy"
OUT = os.path.dirname(os.path.abspath(__file__))
# Test-split videos (seed 42): two REAL, two FAKE.
EXAMPLES = [("dfdc__ahqqqilsxt", "REAL"), ("dfdc__dhxctgyoqj", "REAL"),
            ("dfdc__ekhacizpah", "FAKE"), ("dfdc__ajwpjhrbcv", "FAKE")]


def label(img, text):
    out = cv2.copyMakeBorder(img, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def spectrum(crop_rgb):
    x = normalise(crop_rgb[None])                       # (1, 3, H, W), as in training
    mag = fft_log_magnitude(x, normalised_input=True)[0, 0].numpy()
    assert np.isfinite(mag).all() and mag.min() >= 0 and mag.max() <= 1
    return cv2.cvtColor((mag * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def main():
    pad = np.full((252, 8, 3), 255, np.uint8)
    rows = []
    for vid, cls in EXAMPLES:
        crop = np.load(CACHE.format(vid))[3]            # a middle frame
        rows.append(np.hstack([label(crop, f"{cls}: face crop"), pad,
                               label(spectrum(crop), "log|FFT| (centred)")]))
    grid = np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in rows])
    cv2.imwrite(os.path.join(OUT, "fft_examples.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    seq = np.load(CACHE.format("dfdc__ahqqqilsxt"))
    strip = np.hstack([np.hstack([f, np.full((224, 4, 3), 255, np.uint8)]) for f in seq])
    cv2.imwrite(os.path.join(OUT, "frame_sequence.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print("wrote fft_examples.png and frame_sequence.png to", OUT)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
