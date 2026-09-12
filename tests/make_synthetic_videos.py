"""Generate a tiny synthetic video set so the pipeline can be exercised offline.

Two frame sources:

* ``olivetti`` (default) - real face photographs from scikit-learn's Olivetti
  faces dataset, animated across frames. Using real faces means the face
  detector is genuinely exercised rather than always hitting its fallback.
* ``drawn`` - crude drawn faces, used when Olivetti cannot be downloaded.

In both cases the REAL/FAKE distinction is a synthetic marker (a fine
checkerboard overlay on the FAKE clips), NOT real manipulation. These clips
exist only to prove the code path runs end to end. Any metric produced from
them is a smoke-test artefact and says nothing about deepfake detection
performance.

    python -m tests.make_synthetic_videos --out data/raw/synthetic --per-class 6
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

FRAME_SIZE = (240, 320)   # height, width


def add_high_freq(frame: np.ndarray, strength: int = 14) -> np.ndarray:
    """Overlay a 1-pixel checkerboard: a synthetic high-frequency marker."""
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx + yy) % 2) * strength).astype(np.int16)
    return np.clip(frame.astype(np.int16) + checker[..., None], 0, 255).astype(np.uint8)


def load_olivetti_faces() -> Optional[np.ndarray]:
    """Return (400, 64, 64) uint8 real face images, or None if unavailable."""
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    try:
        from sklearn.datasets import fetch_olivetti_faces

        data = fetch_olivetti_faces(shuffle=False)
    except Exception as exc:
        print(f"[synthetic] Olivetti faces unavailable ({exc}); using drawn faces")
        return None
    return (data.images * 255).astype(np.uint8)


def compose_photo_frame(face: np.ndarray, t: float, rng: np.random.Generator,
                        size: Tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """Paste a real face onto a plain background, drifting slightly over time."""
    h, w = size
    canvas = np.full((h, w), 190, dtype=np.uint8)
    face_size = 200
    resized = cv2.resize(face, (face_size, face_size), interpolation=cv2.INTER_CUBIC)
    x = int((w - face_size) / 2 + 10 * np.sin(t * 2 * np.pi))
    y = int((h - face_size) / 2 + 6 * np.cos(t * 2 * np.pi))
    x, y = max(0, min(x, w - face_size)), max(0, min(y, h - face_size))
    canvas[y:y + face_size, x:x + face_size] = resized
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    noise = rng.normal(0, 2, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def draw_face_frame(t: float, rng: np.random.Generator,
                    size: Tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """Draw one frame containing a simple face-like pattern (offline fallback)."""
    h, w = size
    frame = np.full((h, w, 3), 210, dtype=np.uint8)
    cx = int(w / 2 + 8 * np.sin(t * 2 * np.pi))
    cy = int(h / 2 + 4 * np.cos(t * 2 * np.pi))
    face_w, face_h = int(w * 0.22), int(h * 0.30)

    cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360, (190, 165, 145), -1)
    eye_dy, eye_dx = int(face_h * 0.25), int(face_w * 0.40)
    eye_r = max(3, int(face_w * 0.14))
    for sign in (-1, 1):
        cv2.circle(frame, (cx + sign * eye_dx, cy - eye_dy), eye_r, (35, 30, 30), -1)
    cv2.ellipse(frame, (cx, cy + int(face_h * 0.45)),
                (int(face_w * 0.45), int(face_h * 0.14)), 0, 0, 180, (70, 45, 45), 2)
    noise = rng.normal(0, 3, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def write_clip(path: str, n_frames: int, fps: int, high_freq: bool, seed: int,
               face: Optional[np.ndarray]) -> None:
    rng = np.random.default_rng(seed)
    h, w = FRAME_SIZE
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)
            frame = (compose_photo_frame(face, t, rng) if face is not None
                     else draw_face_frame(t, rng))
            if high_freq:
                frame = add_high_freq(frame)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def build_dataset(out_dir: str, per_class: int = 6, frames: int = 30, fps: int = 10,
                  seed: int = 0, source: str = "olivetti") -> Tuple[str, str]:
    """Write `per_class` REAL and `per_class` FAKE clips. Returns (real_dir, fake_dir)."""
    faces = load_olivetti_faces() if source == "olivetti" else None

    real_dir = os.path.join(out_dir, "real")
    fake_dir = os.path.join(out_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # One distinct identity per clip, so a video-level split separates identities.
    picks: List[Optional[np.ndarray]] = []
    if faces is not None:
        for i in range(2 * per_class):
            picks.append(faces[(i * 10) % len(faces)])
    else:
        picks = [None] * (2 * per_class)

    for i in range(per_class):
        n = frames + (i * 7) % 23        # varying lengths exercise dynamic sampling
        write_clip(os.path.join(real_dir, f"real_{i:02d}.mp4"), n, fps,
                   high_freq=False, seed=seed + i, face=picks[i])
        write_clip(os.path.join(fake_dir, f"fake_{i:02d}.mp4"), n, fps,
                   high_freq=True, seed=seed + 100 + i, face=picks[per_class + i])
    return real_dir, fake_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/synthetic")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="olivetti", choices=["olivetti", "drawn"])
    args = parser.parse_args()

    real_dir, fake_dir = build_dataset(args.out, args.per_class, args.frames,
                                       seed=args.seed, source=args.source)
    print(f"smoke-test clips written to:\n  {real_dir}\n  {fake_dir}")
    print("NOTE: the REAL/FAKE marker is synthetic, not real manipulation. "
          "Metrics from these clips are smoke-test artefacts only.")


if __name__ == "__main__":
    main()
