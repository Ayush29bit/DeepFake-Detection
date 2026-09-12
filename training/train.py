"""Training / validation loop for both the baseline and the hybrid model.

Both models expose the same interface - forward((B, T, 3, H, W)) -> (B,) logit -
so a single loop, loss and metric set covers them, keeping the comparison in
docs/evaluation.md apples-to-apples.

Example:
    python -m training.train --model hybrid --real-dir data/raw/real \
        --fake-dir data/raw/fake --num-epochs 5 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import compute_metrics, format_metrics, save_metrics
from models.hybrid import build_model
from preprocessing.dataset import VideoFaceDataset, build_dataloader
from preprocessing.splits import (assert_no_leakage, build_items, split_summary,
                                  split_videos, write_manifest)
from training.config import Config, add_config_args, config_from_args

HIGHER_IS_BETTER = {"f1", "accuracy", "roc_auc", "precision", "recall"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model: nn.Module, loader, criterion, device, optimizer=None
              ) -> Tuple[float, List[int], List[float]]:
    """One pass over a loader. Trains when `optimizer` is given, else evaluates.

    Returns (mean loss, true labels, predicted FAKE probabilities).
    """
    training = optimizer is not None
    model.train(training)

    total_loss, total_items = 0.0, 0
    y_true: List[int] = []
    y_prob: List[float] = []

    with torch.set_grad_enabled(training):
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch = labels.size(0)
            total_loss += float(loss.detach()) * batch
            total_items += batch
            y_true.extend(labels.detach().cpu().int().tolist())
            y_prob.extend(torch.sigmoid(logits.detach()).cpu().float().tolist())

    mean_loss = total_loss / max(total_items, 1)
    return mean_loss, y_true, y_prob


def evaluate(model: nn.Module, loader, criterion, device,
             threshold: float = 0.5) -> Dict[str, object]:
    """Run one no-grad pass and return loss plus classification metrics."""
    loss, y_true, y_prob = run_epoch(model, loader, criterion, device, optimizer=None)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = loss
    return metrics


def is_better(metric_name: str, candidate: float, best: float) -> bool:
    if metric_name in HIGHER_IS_BETTER:
        return candidate > best
    return candidate < best


def train(config: Config) -> Dict[str, object]:
    set_seed(config.seed)
    device = config.resolve_device()

    run_name = config.run_name or f"{config.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(config.run_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------ data + splits
    items = build_items(config)
    if not items:
        raise RuntimeError("no videos found; check --real-dir/--fake-dir/--manifest")
    splits = split_videos(items, config.train_ratio, config.val_ratio,
                          config.test_ratio, seed=config.seed)
    assert_no_leakage(splits)
    print(f"[train] {len(items)} videos, split at the video level (seed={config.seed}):")
    print(split_summary(splits))
    for name, group in splits.items():
        write_manifest(group, os.path.join(run_dir, f"split_{name}.csv"))

    train_loader = build_dataloader(
        VideoFaceDataset(splits["train"], config, train=True), config, shuffle=True)
    val_loader = build_dataloader(
        VideoFaceDataset(splits["val"], config, train=False), config, shuffle=False)

    # ------------------------------------------------------------- model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model={config.model} device={device} trainable_params={n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["epoch", "train_loss", "train_acc", "val_loss",
                                 "val_acc", "val_precision", "val_recall", "val_f1",
                                 "val_roc_auc", "seconds"])

    best_metric = -float("inf") if config.best_metric in HIGHER_IS_BETTER else float("inf")
    best_epoch = -1
    ckpt_path = os.path.join(config.checkpoint_dir, f"{run_name}_best.pt")
    history: List[Dict[str, object]] = []
    val_metrics: Dict[str, object] = {}

    # ------------------------------------------------------------- epochs
    for epoch in range(1, config.num_epochs + 1):
        started = time.time()
        train_loss, y_true, y_prob = run_epoch(model, train_loader, criterion,
                                               device, optimizer)
        train_metrics = compute_metrics(y_true, y_prob, threshold=config.threshold)
        val_metrics = evaluate(model, val_loader, criterion, device, config.threshold)
        elapsed = time.time() - started

        print(f"[epoch {epoch}/{config.num_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['f1']:.4f} ({elapsed:.1f}s)")

        auc = val_metrics["roc_auc"]
        with open(history_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                epoch, f"{train_loss:.6f}", f"{train_metrics['accuracy']:.6f}",
                f"{val_metrics['loss']:.6f}", f"{val_metrics['accuracy']:.6f}",
                f"{val_metrics['precision']:.6f}", f"{val_metrics['recall']:.6f}",
                f"{val_metrics['f1']:.6f}", "" if auc is None else f"{auc:.6f}",
                f"{elapsed:.2f}"])
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_metrics["loss"], "val_f1": val_metrics["f1"]})

        candidate = val_metrics.get(config.best_metric)
        if candidate is None:
            candidate = val_metrics["loss"]
        if is_better(config.best_metric, float(candidate), best_metric):
            best_metric, best_epoch = float(candidate), epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "best_metric": {"name": config.best_metric, "value": best_metric},
            }, ckpt_path)
            print(f"          saved best checkpoint -> {ckpt_path} "
                  f"(val_{config.best_metric}={best_metric:.4f})")

    # ------------------------------------------------------------ wrap up
    print(f"\n[train] best epoch {best_epoch} (val_{config.best_metric}={best_metric:.4f})")
    print(format_metrics(val_metrics, title="[train] final-epoch validation metrics"))

    save_metrics(val_metrics, os.path.join(run_dir, "val_metrics_final_epoch.json"),
                 extra={"run_name": run_name, "model": config.model,
                        "best_epoch": best_epoch, "checkpoint": ckpt_path,
                        "note": "metrics from the final epoch's validation pass"})
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)

    return {"run_dir": run_dir, "checkpoint": ckpt_path, "best_epoch": best_epoch,
            "best_metric": best_metric, "history": history,
            "final_val_metrics": val_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ResNet18 baseline or the hybrid deepfake detector.")
    add_config_args(parser)
    config = config_from_args(parser.parse_args())
    train(config)


if __name__ == "__main__":
    main()
