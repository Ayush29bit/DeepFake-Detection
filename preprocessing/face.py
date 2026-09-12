"""Face detection and cropping for sampled video frames.

Detector choice is configurable (docs/implementation.md leaves it open):

* ``haar``  - OpenCV's bundled Haar cascade. Default: no extra dependency.
* ``mtcnn`` - MTCNN via ``facenet-pytorch``; used only if that package is
  installed, otherwise construction raises with an explanatory message.
* ``none``  - no detection, always centre-crop (useful for smoke tests).

Fallback policy when no face is found in a frame (applied in this order, and
identical at training and inference time so the two stay consistent):

1. Use the detected face box.
2. Otherwise reuse the most recent detected box from the same video
   (nearest preceding frame).
3. Otherwise reuse the next detected box in the same video (back-fill), which
   covers videos whose first frames have no detection.
4. Otherwise centre-crop the frame to a square.

Frames are never dropped, so the sequence length is always ``num_frames`` and a
single failed frame cannot break the video pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


def square_box_with_margin(box: Box, width: int, height: int,
                           margin: float = 0.25) -> Box:
    """Expand a box by `margin`, make it square, and clamp it to the frame."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(x2 - x1, y2 - y1) * (1.0 + margin)
    size = min(size, min(width, height))
    half = size / 2.0
    cx = min(max(cx, half), width - half)
    cy = min(max(cy, half), height - half)
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def centre_crop_box(width: int, height: int) -> Box:
    size = min(width, height)
    x1 = (width - size) // 2
    y1 = (height - size) // 2
    return (x1, y1, x1 + size, y1 + size)


class FaceExtractor:
    """Detect the largest face in a frame and return a resized square crop."""

    def __init__(self, detector: str = "haar", image_size: int = 224,
                 margin: float = 0.25, device: str = "cpu"):
        self.detector_name = detector
        self.image_size = int(image_size)
        self.margin = float(margin)
        self._cascade = None
        self._mtcnn = None

        if detector == "haar":
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                raise RuntimeError(f"could not load Haar cascade from {path}")
            self._cascade = cascade
        elif detector == "mtcnn":
            try:
                from facenet_pytorch import MTCNN
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "face_detector='mtcnn' requires facenet-pytorch "
                    "(pip install facenet-pytorch), or use --face-detector haar"
                ) from exc
            self._mtcnn = MTCNN(keep_all=True, device=device)
        elif detector != "none":
            raise ValueError(f"unknown face detector: {detector!r}")

    # ---------------------------------------------------------------- detect
    def detect(self, frame_rgb: np.ndarray) -> Optional[Box]:
        """Return the largest detected face box, or None."""
        if self.detector_name == "none":
            return None
        if self._cascade is not None:
            return self._detect_haar(frame_rgb)
        return self._detect_mtcnn(frame_rgb)

    def _detect_haar(self, frame_rgb: np.ndarray) -> Optional[Box]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(x + w), int(y + h))

    def _detect_mtcnn(self, frame_rgb: np.ndarray) -> Optional[Box]:  # pragma: no cover
        from PIL import Image

        boxes, _ = self._mtcnn.detect(Image.fromarray(frame_rgb))
        if boxes is None or len(boxes) == 0:
            return None
        best = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return tuple(int(round(v)) for v in best)  # type: ignore[return-value]

    # ------------------------------------------------------------------ crop
    def crop(self, frame_rgb: np.ndarray, box: Optional[Box]) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if box is None:
            box = centre_crop_box(w, h)
        else:
            box = square_box_with_margin(box, w, h, self.margin)
        x1, y1, x2, y2 = box
        patch = frame_rgb[max(y1, 0):y2, max(x1, 0):x2]
        if patch.size == 0:
            patch = frame_rgb
        return cv2.resize(patch, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_AREA)

    def extract(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Crop a single frame. Returns (crop, whether a face was detected)."""
        box = self.detect(frame_rgb)
        return self.crop(frame_rgb, box), box is not None

    def extract_sequence(self, frames: Sequence[np.ndarray]
                         ) -> Tuple[List[np.ndarray], int]:
        """Crop a whole frame sequence, applying the documented fallback chain.

        Returns (crops, number_of_frames_with_a_detected_face).
        """
        boxes: List[Optional[Box]] = [self.detect(f) for f in frames]
        detected = sum(b is not None for b in boxes)

        filled: List[Optional[Box]] = list(boxes)
        last: Optional[Box] = None
        for i, b in enumerate(filled):          # forward fill (previous frame)
            if b is None:
                filled[i] = last
            else:
                last = b
        nxt: Optional[Box] = None
        for i in range(len(filled) - 1, -1, -1):  # back-fill (next frame)
            if filled[i] is None:
                filled[i] = nxt
            else:
                nxt = filled[i]

        crops = [self.crop(frame, box) for frame, box in zip(frames, filled)]
        return crops, detected
