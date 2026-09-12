"""Video-level train/validation/test splitting.

Splits are made over whole videos, never over frames: every frame sampled from
a video lands in exactly one split, so no video leaks across sets
(docs/datasets.md).
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from training.config import FAKE, REAL

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


@dataclass(frozen=True)
class VideoItem:
    path: str
    label: int          # REAL = 0, FAKE = 1
    video_id: str       # unique key, also used as the preprocessing cache key


def _video_id(path: str, label: int) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    return f"{'fake' if label == FAKE else 'real'}__{parent}__{stem}"


def discover_videos(real_dir: str, fake_dir: str, limit: int = 0) -> List[VideoItem]:
    """Collect videos from a REAL directory and a FAKE directory (recursively).

    `limit` > 0 caps the number of videos taken per class, which keeps smoke
    runs and quick experiments small.
    """
    items: List[VideoItem] = []
    for directory, label in ((real_dir, REAL), (fake_dir, FAKE)):
        if not directory:
            continue
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"video directory not found: {directory}")
        found: List[str] = []
        for root, _dirs, files in os.walk(directory):
            for name in sorted(files):
                if name.lower().endswith(VIDEO_EXTENSIONS):
                    found.append(os.path.join(root, name))
        found.sort()
        if limit > 0:
            found = found[:limit]
        items.extend(VideoItem(p, label, _video_id(p, label)) for p in found)
    return items


def load_manifest(path: str, limit: int = 0) -> List[VideoItem]:
    """Load videos from a CSV manifest with columns: path,label[,video_id].

    `label` accepts 0/1 or the strings REAL/FAKE (case-insensitive).
    """
    items: List[VideoItem] = []
    counts = {REAL: 0, FAKE: 0}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = str(row["label"]).strip().upper()
            label = REAL if raw in ("0", "REAL") else FAKE if raw in ("1", "FAKE") else None
            if label is None:
                raise ValueError(f"unrecognised label in manifest: {row['label']!r}")
            if limit > 0 and counts[label] >= limit:
                continue
            counts[label] += 1
            video_path = row["path"].strip()
            items.append(VideoItem(video_path, label,
                                   row.get("video_id") or _video_id(video_path, label)))
    return items


def write_manifest(items: Sequence[VideoItem], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "video_id"])
        for item in items:
            writer.writerow([item.path, item.label, item.video_id])


def split_videos(items: Sequence[VideoItem], train_ratio: float = 0.70,
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 seed: int = 42) -> Dict[str, List[VideoItem]]:
    """Split videos into train/val/test, stratified by class, seeded.

    Shuffling is done per class with a fixed seed, so the split is reproducible
    and class balance is preserved as closely as the counts allow.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    splits: Dict[str, List[VideoItem]] = {"train": [], "val": [], "test": []}
    for label in (REAL, FAKE):
        group = sorted([i for i in items if i.label == label], key=lambda i: i.video_id)
        random.Random(seed).shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Guarantee a non-empty val/test whenever there are enough videos.
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])
    return splits


def split_summary(splits: Dict[str, List[VideoItem]]) -> str:
    lines = []
    for name in ("train", "val", "test"):
        group = splits.get(name, [])
        real = sum(1 for i in group if i.label == REAL)
        fake = sum(1 for i in group if i.label == FAKE)
        lines.append(f"  {name:<5} videos={len(group):<5} REAL={real:<5} FAKE={fake}")
    return "\n".join(lines)


def assert_no_leakage(splits: Dict[str, List[VideoItem]]) -> None:
    """Raise if any video id appears in more than one split."""
    seen: Dict[str, str] = {}
    for name, group in splits.items():
        for item in group:
            if item.video_id in seen:
                raise AssertionError(
                    f"video {item.video_id!r} appears in both "
                    f"{seen[item.video_id]!r} and {name!r} splits")
            seen[item.video_id] = name


def build_items(config) -> List[VideoItem]:
    """Resolve the configured dataset source into a list of videos."""
    if config.manifest:
        return load_manifest(config.manifest, limit=config.limit)
    if config.real_dir or config.fake_dir:
        return discover_videos(config.real_dir, config.fake_dir, limit=config.limit)
    raise ValueError("no dataset source configured: pass --manifest or "
                     "--real-dir/--fake-dir")
