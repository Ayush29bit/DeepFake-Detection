"""Video loading and deterministic frame sampling (OpenCV)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np


def sample_indices(total_frames: int, num_frames: int, jitter: bool = False,
                   rng: Optional[np.random.Generator] = None) -> List[int]:
    """Return `num_frames` evenly spaced frame indices for a video.

    Works for any video length: indices are computed from `total_frames`, never
    hardcoded. If the video has fewer frames than requested, indices repeat
    (clamped to the last valid frame) so the returned sequence always has
    length `num_frames`.

    `jitter` moves each index randomly within its own segment. It is intended
    for training-time augmentation only; evaluation and inference must leave it
    off so sampling is deterministic (see docs/datasets.md).
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    if total_frames < num_frames:
        base = np.linspace(0, total_frames - 1, num=num_frames)
        return [int(round(i)) for i in base]

    edges = np.linspace(0, total_frames, num=num_frames + 1)
    if jitter:
        rng = rng or np.random.default_rng()
        idx = [int(rng.integers(int(np.floor(edges[i])),
                                max(int(np.floor(edges[i])) + 1,
                                    int(np.ceil(edges[i + 1])))))
               for i in range(num_frames)]
    else:
        idx = [int((edges[i] + edges[i + 1]) / 2.0) for i in range(num_frames)]
    return [min(max(i, 0), total_frames - 1) for i in idx]


class VideoReader:
    """Thin OpenCV wrapper: frame count plus random/sequential frame access.

    Frames are returned as RGB uint8 arrays of shape (H, W, 3).
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"could not open video: {self.path}")
        reported = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = reported if reported > 0 else self._count_frames()
        if self.frame_count <= 0:
            raise IOError(f"video has no readable frames: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def _count_frames(self) -> int:
        count = 0
        while True:
            ok = self.cap.grab()
            if not ok:
                break
            count += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return count

    def read_indices(self, indices: Sequence[int]) -> List[np.ndarray]:
        """Read the given frame indices, in ascending order, as RGB arrays.

        Seeking is unreliable for some codecs, so a failed seek-and-read falls
        back to a sequential scan. Frames that still cannot be decoded reuse the
        previously decoded frame (or, for a leading failure, are back-filled
        once a frame is available).
        """
        wanted = list(indices)
        frames: List[Optional[np.ndarray]] = [None] * len(wanted)
        order = sorted(range(len(wanted)), key=lambda i: wanted[i])

        last: Optional[np.ndarray] = None
        for pos in order:
            target = int(wanted[pos])
            frame = self._read_single(target)
            if frame is None:
                frame = last
            else:
                last = frame
            frames[pos] = frame

        if last is None:
            raise IOError(f"could not decode any frame from: {self.path}")
        return [f if f is not None else last for f in frames]

    def _read_single(self, index: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(index + 1):
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    return None
                if i == index:
                    break
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_sampled_frames(path: str, num_frames: int = 8, jitter: bool = False,
                        rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """Open a video and return `num_frames` uniformly sampled RGB frames."""
    with VideoReader(path) as reader:
        indices = sample_indices(reader.frame_count, num_frames, jitter=jitter, rng=rng)
        return reader.read_indices(indices)
