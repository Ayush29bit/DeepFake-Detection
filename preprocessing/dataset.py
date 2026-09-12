"""PyTorch Dataset over face-cropped video frame sequences.

An item is (frames, label):
    frames -> float tensor (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), normalised
              with ImageNet statistics
    label  -> float tensor scalar, REAL = 0.0, FAKE = 1.0

Videos are decoded lazily, one at a time, inside ``__getitem__`` - the dataset
never holds the video corpus in RAM. Extracted face crops are cached to
``data/processed`` as small uint8 ``.npy`` arrays so repeated epochs and reruns
skip video decoding and face detection.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.face import FaceExtractor
from preprocessing.splits import VideoItem
from preprocessing.video import load_sampled_frames
from training.config import IMAGENET_MEAN, IMAGENET_STD, Config


def normalise(frames_uint8: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 RGB -> (T, 3, H, W) float tensor, ImageNet-normalised."""
    tensor = torch.from_numpy(np.ascontiguousarray(frames_uint8)).float().div_(255.0)
    tensor = tensor.permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of `normalise`, returning values in [0, 1]. Shape (..., 3, H, W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def augment_sequence(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation applied consistently to every frame of a sequence.

    Horizontal flip, small rotation and mild brightness/contrast jitter only -
    docs/datasets.md deliberately avoids heavy blur/compression augmentation,
    which would destroy the high-frequency content the frequency branch reads.
    Augmentation happens here, on the spatial crop, before the model computes
    the FFT, so both branches see the same augmented frame.
    """
    import cv2

    out = frames
    if rng.random() < 0.5:
        out = out[:, :, ::-1, :]

    angle = float(rng.uniform(-5.0, 5.0))
    if abs(angle) > 0.1:
        h, w = out.shape[1:3]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        out = np.stack([cv2.warpAffine(f, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
                        for f in out])

    brightness = float(rng.uniform(-16.0, 16.0))
    contrast = float(rng.uniform(0.9, 1.1))
    out = np.clip(out.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    return out


class VideoFaceDataset(Dataset):
    """Dataset of (face-crop sequence, label) pairs, one item per video."""

    def __init__(self, items: Sequence[VideoItem], config: Config, train: bool = False,
                 extractor: Optional[FaceExtractor] = None):
        self.items: List[VideoItem] = list(items)
        self.config = config
        self.train = train
        self.cache_dir = config.cache_dir if config.use_cache else ""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._extractor = extractor
        self.failures: List[str] = []

    # Built lazily so the dataset survives being pickled to DataLoader workers.
    @property
    def extractor(self) -> FaceExtractor:
        if self._extractor is None:
            self._extractor = FaceExtractor(
                detector=self.config.face_detector,
                image_size=self.config.image_size,
                margin=self.config.face_margin,
            )
        return self._extractor

    def __len__(self) -> int:
        return len(self.items)

    def _cache_path(self, item: VideoItem) -> str:
        cfg = self.config
        key = f"{item.video_id}_n{cfg.num_frames}_s{cfg.image_size}_{cfg.face_detector}"
        return os.path.join(self.cache_dir, key + ".npy")

    def load_crops(self, item: VideoItem) -> Tuple[np.ndarray, int]:
        """Return (T, H, W, 3) uint8 face crops for a video, using the cache.

        Frame-index jitter is only applied for training, and jittered sequences
        are never cached (the cache always holds the deterministic sampling).
        """
        cfg = self.config
        jitter = self.train and cfg.train_jitter
        cache_path = self._cache_path(item) if self.cache_dir else ""

        if cache_path and not jitter and os.path.exists(cache_path):
            return np.load(cache_path), -1

        rng = np.random.default_rng() if jitter else None
        frames = load_sampled_frames(item.path, num_frames=cfg.num_frames,
                                     jitter=jitter, rng=rng)
        crops, detected = self.extractor.extract_sequence(frames)
        stacked = np.stack(crops).astype(np.uint8)

        if cache_path and not jitter:
            # The temp name must end in .npy: np.save appends the suffix itself
            # when it is missing, which would leave os.replace with no source.
            tmp = f"{cache_path}.{os.getpid()}.tmp.npy"
            try:
                np.save(tmp, stacked)
                os.replace(tmp, cache_path)
            except OSError as exc:   # a cache problem must not lose the crops
                print(f"[dataset] WARNING: could not cache {cache_path} ({exc})")
        return stacked, detected

    def __getitem__(self, index: int):
        item = self.items[index]
        cfg = self.config
        try:
            crops, _ = self.load_crops(item)
        except Exception as exc:  # a single unreadable video must not kill a run
            self.failures.append(f"{item.path}: {exc}")
            print(f"[dataset] WARNING: failed to read {item.path} ({exc}); "
                  f"substituting a blank sequence")
            crops = np.zeros((cfg.num_frames, cfg.image_size, cfg.image_size, 3),
                             dtype=np.uint8)

        if self.train and cfg.augment:
            crops = augment_sequence(crops, np.random.default_rng())

        frames = normalise(crops)
        label = torch.tensor(float(item.label), dtype=torch.float32)
        return frames, label


def build_dataloader(dataset: VideoFaceDataset, config: Config,
                     shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
