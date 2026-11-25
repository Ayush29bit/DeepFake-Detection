"""Print a Markdown comparison table from saved evaluation JSON files.

Every number comes from a file written by ``evaluation.evaluate``; nothing is
typed in by hand (docs/evaluation.md).

Example:
    python -m evaluation.compare runs/eval/dfdc_baseline_best_test.json \
        runs/eval/dfdc_hybrid_best_test.json
"""

from __future__ import annotations

import argparse
import json
import os

from evaluation.metrics import bootstrap_ci


def describe(m: dict) -> str:
    if m.get("model") == "baseline":
        return "ResNet18 baseline (frame-logit mean)"
    parts = ["spatial"]
    if m.get("use_frequency"):
        parts.append("frequency")
    parts.append("Transformer" if m.get("use_temporal") else "mean-pool")
    return " + ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="metrics JSON from evaluation.evaluate")

    parser.add_argument("--ci", action="store_true",
                        help="add 95%% stratified bootstrap intervals for F1 and ROC-AUC")
    args = parser.parse_args()

    print("| Model | Checkpoint | Split | Threshold | n | Accuracy | Precision | Recall "
          "| F1 | ROC-AUC | TN/FP/FN/TP |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        cm = m["confusion_matrix"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        f1 = f"{m['f1']:.3f}"
        if args.ci and m["roc_auc"] is not None:
            ci = bootstrap_ci([p["label"] for p in m["predictions"]],
                              [p["fake_probability"] for p in m["predictions"]], m["threshold"])
            f1 += " [{:.3f}, {:.3f}]".format(*ci["f1"])
            auc += " [{:.3f}, {:.3f}]".format(*ci["roc_auc"])
        source = " (val)" if m.get("threshold_source") == "youden_on_val" else ""
        print(f"| {describe(m)} | {os.path.basename(m['checkpoint'])} | {m['split']} "
              f"| {m['threshold']:.3f}{source} | {m['n']} | {m['accuracy']:.3f} | {m['precision']:.3f} "
              f"| {m['recall']:.3f} | {f1} | {auc} "
              f"| {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} |")


if __name__ == "__main__":
    main()
