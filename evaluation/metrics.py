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
