"""Measure parameter count and CPU inference time for trained checkpoints.

Reports two times per model: the network forward pass on one 8-frame clip,
and the full single-video pipeline (decode, face crop, forward) from
inference.predict. Run it on an otherwise idle machine.

Example:
    python -m evaluation.timing --video data/raw/demo_videos/1_REAL_ahqqqilsxt.mp4 \
        checkpoints/dfdc_baseline_5ep_best.pt checkpoints/dfdc_hybrid_best.pt
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from evaluation.evaluate import load_checkpoint
from inference.predict import predict_video


def timed(fn, repeats: int, warmup: int = 2) -> float:
    """Median wall-clock seconds of fn() over `repeats` runs, after warm-up."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--video", required=True, help="video for the end-to-end timing")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    print(f"torch threads: {torch.get_num_threads()}")
    print("| Checkpoint | Parameters | Forward, 1 clip (ms) | Full pipeline, 1 video (s) |")
    print("|---|---:|---:|---:|")
    for path in args.checkpoints:
        model, saved, _ckpt, device = load_checkpoint(path, device=torch.device("cpu"))
        model.eval()
        clip = torch.randn(1, saved.num_frames, 3, saved.image_size, saved.image_size)
        with torch.no_grad():
            forward = timed(lambda: model(clip), args.repeats)
        pipeline = timed(lambda: predict_video(args.video, model, saved, device=device),
                         max(3, args.repeats // 5), warmup=1)
        params = sum(p.numel() for p in model.parameters())
        print(f"| {os.path.basename(path)} | {params / 1e6:.2f} M | {forward * 1000:.0f} "
              f"| {pipeline:.2f} |")


if __name__ == "__main__":
    main()
