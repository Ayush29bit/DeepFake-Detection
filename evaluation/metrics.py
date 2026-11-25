"""Video-level classification metrics (scikit-learn).

All metrics are computed from one prediction per video, on a held-out split -
never on training data (docs/evaluation.md).
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true: Sequence[int], y_prob: Sequence[float],
                    threshold: float = 0.5) -> Dict[str, object]:
    """Accuracy / precision / recall / F1 / ROC-AUC plus a confusion matrix.

    FAKE (label 1) is the positive class. ROC-AUC is reported as None when the
    split contains a single class, where it is undefined.
    """
    y_true = np.asarray(list(y_true)).astype(int)
    y_prob = np.asarray(list(y_prob)).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    # ROC-AUC is undefined when the split holds a single class. Depending on the
    # scikit-learn version that either raises or returns nan; both mean "None".
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(roc_auc_score(y_true, y_prob))
        roc_auc: Optional[float] = None if np.isnan(value) else value
    except ValueError:
        roc_auc = None

    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"real": int((y_true == 0).sum()), "fake": int((y_true == 1).sum())},
    }


def youden_threshold(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Threshold maximising TPR - FPR. Pick it on validation, never on test."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob))
    best = int(np.argmax(tpr - fpr))
    return float(min(thresholds[best], 1.0))  # roc_curve's first threshold is inf


def bootstrap_ci(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, tuple]:
    """Percentile bootstrap intervals for ROC-AUC and F1 over videos.

    Resamples videos with replacement within each class, so every resample keeps
    the split's REAL/FAKE counts and ROC-AUC stays defined even with few REAL
    videos. Returns {"roc_auc": (low, high), "f1": (low, high)}.
    """
    y_true = np.asarray(list(y_true)).astype(int)
    y_prob = np.asarray(list(y_prob)).astype(float)
    real, fake = np.flatnonzero(y_true == 0), np.flatnonzero(y_true == 1)
    if len(real) == 0 or len(fake) == 0:
        raise ValueError("bootstrap_ci needs both classes in the split")
    rng = np.random.default_rng(seed)
    aucs, f1s = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(real, len(real)), rng.choice(fake, len(fake))])
        aucs[i] = roc_auc_score(y_true[idx], y_prob[idx])
        f1s[i] = f1_score(y_true[idx], (y_prob[idx] >= threshold).astype(int), zero_division=0)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"roc_auc": tuple(float(v) for v in np.percentile(aucs, q)),
            "f1": tuple(float(v) for v in np.percentile(f1s, q))}


def format_metrics(metrics: Dict[str, object], title: str = "") -> str:
    """Render a metrics dict as a short text block, including the matrix."""
    cm = metrics.get("confusion_matrix", {})
    auc = metrics.get("roc_auc")
    lines = []
    if title:
        lines.append(title)
    if "loss" in metrics and metrics["loss"] is not None:
        lines.append(f"  loss       {metrics['loss']:.4f}")
    lines += [
        f"  accuracy   {metrics['accuracy']:.4f}",
        f"  precision  {metrics['precision']:.4f}",
        f"  recall     {metrics['recall']:.4f}",
        f"  f1         {metrics['f1']:.4f}",
        f"  roc_auc    {'n/a (single class in split)' if auc is None else f'{auc:.4f}'}",
        f"  videos     {metrics['n']} "
        f"(REAL={metrics['support']['real']}, FAKE={metrics['support']['fake']})",
        "  confusion matrix (rows = true, cols = predicted)",
        "                pred REAL  pred FAKE",
        f"    true REAL   {cm.get('tn', 0):>9}  {cm.get('fp', 0):>9}",
        f"    true FAKE   {cm.get('fn', 0):>9}  {cm.get('tp', 0):>9}",
    ]
    return "\n".join(lines)


def save_metrics(metrics: Dict[str, object], path: str,
                 extra: Optional[Dict[str, object]] = None) -> str:
    """Write metrics to JSON, optionally merged with run metadata."""
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
