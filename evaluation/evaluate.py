"""Evaluate a trained checkpoint on a held-out split.

The split is rebuilt from the same seed and ratios stored in the checkpoint's
config, so the evaluated videos are exactly the ones the model never trained on.

Example:
    python -m evaluation.evaluate --checkpoint checkpoints/hybrid_best.pt \
        --real-dir data/raw/real --fake-dir data/raw/fake --split test
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
import torch.nn as nn

from evaluation.metrics import (compute_metrics, format_metrics, save_metrics,
                                youden_threshold)
from models.hybrid import build_model
from preprocessing.dataset import VideoFaceDataset, build_dataloader
from preprocessing.splits import assert_no_leakage, build_items, split_videos
from training.config import Config, add_config_args, config_from_args


def load_checkpoint(path: str, override: Config = None, device=None):
    """Load a checkpoint and rebuild its model.

    Data-related settings (dataset paths, batch size, workers, device) come from
    `override`; model architecture settings come from the checkpoint, so a model
    is always rebuilt exactly as it was trained.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = Config(**ckpt["config"])
    if override is not None:
        for field in ("real_dir", "fake_dir", "manifest", "limit", "batch_size",
                      "num_workers", "device", "cache_dir", "use_cache", "threshold"):
            setattr(saved, field, getattr(override, field))
    device = device or saved.resolve_device()
    model = build_model(saved)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, saved, ckpt, device


def predict_split(model, saved: Config, group, device):
    """Return (mean loss, y_true, y_prob) over a list of videos, in order."""
    loader = build_dataloader(VideoFaceDataset(group, saved, train=False),
                              saved, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0
    y_true, y_prob = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            logits = model(frames)
            total_loss += float(criterion(logits, labels)) * labels.size(0)
            n += labels.size(0)
            y_true.extend(labels.cpu().int().tolist())
            y_prob.extend(torch.sigmoid(logits).cpu().float().tolist())
    return total_loss / max(n, 1), y_true, y_prob


def evaluate_checkpoint(checkpoint: str, config: Config, split: str = "test",
                        output: str = "", val_threshold: bool = False
                        ) -> Dict[str, object]:
    model, saved, ckpt, device = load_checkpoint(checkpoint, override=config)

    items = build_items(saved)
    splits = split_videos(items, saved.train_ratio, saved.val_ratio,
                          saved.test_ratio, seed=saved.seed)
    assert_no_leakage(splits)
    if split not in splits:
        raise ValueError(f"unknown split {split!r}; expected train/val/test")
    group = splits[split]
    if not group:
        raise RuntimeError(f"the {split!r} split is empty for this dataset")

    threshold_source = "config"
    if val_threshold:
        _, v_true, v_prob = predict_split(model, saved, splits["val"], device)
        saved.threshold = youden_threshold(v_true, v_prob)
        threshold_source = "youden_on_val"
        print(f"[evaluate] threshold {saved.threshold:.4f} chosen on the val split")

    loss, y_true, y_prob = predict_split(model, saved, group, device)
    metrics = compute_metrics(y_true, y_prob, threshold=saved.threshold)
    metrics["loss"] = loss

    title = (f"[evaluate] model={saved.model} split={split} "
             f"checkpoint={os.path.basename(checkpoint)} "
             f"(trained to epoch {ckpt.get('epoch')})")
    print(format_metrics(metrics, title=title))

    out_path = output or os.path.join(saved.run_dir, "eval",
                                      f"{os.path.splitext(os.path.basename(checkpoint))[0]}"
                                      f"_{split}{'_valthr' if val_threshold else ''}.json")
    save_metrics(metrics, out_path, extra={
        "checkpoint": checkpoint, "split": split, "model": saved.model,
        "backbone": saved.backbone, "seed": saved.seed,
        "use_frequency": saved.use_frequency, "use_temporal": saved.use_temporal,
        "threshold_source": threshold_source,
        "predictions": [{"video_id": i.video_id, "label": t, "fake_probability": p}
                        for i, t, p in zip(group, y_true, y_prob)],
    })
    print(f"[evaluate] metrics written to {out_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on a held-out split.")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="", help="where to write metrics JSON")
    parser.add_argument("--val-threshold", action="store_true",
                        help="pick the decision threshold on the val split (Youden's J)")
    add_config_args(parser)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, config_from_args(args), args.split, args.output,
                        args.val_threshold)


if __name__ == "__main__":
    main()
