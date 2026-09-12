"""End-to-end inference: a video file in, REAL/FAKE plus confidence out.

    video -> uniform frame sampling -> face crops -> model -> sigmoid -> decision

The probability comes straight from the model's sigmoid output. Nothing here
invents or adjusts a score.

Example:
    python -m inference.predict --checkpoint checkpoints/hybrid_best.pt \
        --video data/raw/fake/clip01.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from preprocessing.dataset import normalise
from preprocessing.face import FaceExtractor
from preprocessing.video import load_sampled_frames


@dataclass
class Prediction:
    """One video-level prediction."""
    video: str
    label: str              # "REAL" or "FAKE"
    fake_probability: float  # sigmoid(logit): P(FAKE)
    confidence: float        # probability assigned to the predicted class
    logit: float
    frames_used: int
    faces_detected: int      # frames where the detector found a face
    face_crops: Optional[np.ndarray] = None  # (T, H, W, 3) uint8, if requested

    def summary(self) -> str:
        return (f"Prediction: {self.label}\n"
                f"Confidence: {self.confidence * 100:.1f}%\n"
                f"P(FAKE):    {self.fake_probability:.4f}\n"
                f"Frames:     {self.frames_used} "
                f"({self.faces_detected} with a detected face)")


def extract_face_sequence(video_path: str, num_frames: int, image_size: int,
                          detector: str = "haar", margin: float = 0.25):
    """Sample frames deterministically and return (crops (T,H,W,3) uint8, n_detected)."""
    frames = load_sampled_frames(video_path, num_frames=num_frames, jitter=False)
    extractor = FaceExtractor(detector=detector, image_size=image_size, margin=margin)
    crops, detected = extractor.extract_sequence(frames)
    return np.stack(crops).astype(np.uint8), detected


@torch.no_grad()
def predict_video(video_path: str, model, config, device=None,
                  return_crops: bool = False) -> Prediction:
    """Run the full pipeline on one video with an already-loaded model."""
    device = device or config.resolve_device()
    model.eval()

    crops, detected = extract_face_sequence(
        video_path, config.num_frames, config.image_size,
        detector=config.face_detector, margin=config.face_margin)

    batch = normalise(crops).unsqueeze(0).to(device)   # (1, T, 3, H, W)
    logit = model(batch)
    logit_value = float(logit.reshape(-1)[0])
    fake_prob = float(torch.sigmoid(torch.tensor(logit_value)))

    is_fake = fake_prob >= config.threshold
    label = "FAKE" if is_fake else "REAL"
    confidence = fake_prob if is_fake else 1.0 - fake_prob

    return Prediction(video=video_path, label=label, fake_probability=fake_prob,
                      confidence=confidence, logit=logit_value,
                      frames_used=int(crops.shape[0]), faces_detected=int(detected),
                      face_crops=crops if return_crops else None)


def predict_from_checkpoint(video_path: str, checkpoint: str, device: str = "auto",
                            return_crops: bool = False) -> Prediction:
    """Load a checkpoint and predict one video. Convenience wrapper for the UI."""
    from evaluation.evaluate import load_checkpoint

    model, saved, _ckpt, resolved = load_checkpoint(checkpoint)
    if device != "auto":
        resolved = torch.device(device)
        model.to(resolved)
    return predict_video(video_path, model, saved, device=resolved,
                         return_crops=return_crops)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict REAL/FAKE for a video with a trained checkpoint.")
    parser.add_argument("--video", required=True, help="path to a video file")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    result = predict_from_checkpoint(args.video, args.checkpoint, device=args.device)
    print(result.summary())


if __name__ == "__main__":
    main()
