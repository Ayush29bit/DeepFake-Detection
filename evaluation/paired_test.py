"""Paired DeLong test between two models evaluated on the same test split.

Reads two metrics JSON files written by ``evaluation.evaluate`` and matches
their per-video predictions by video id, so both AUCs come from exactly the
same videos.

Example:
    python -m evaluation.paired_test runs/eval/dfdc_hybrid_best_test.json \
        runs/eval/dfdc_baseline_5ep_best_test.json
"""

from __future__ import annotations

import argparse
import json
import os

from evaluation.metrics import delong_test


def load_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return {p["video_id"]: p for p in json.load(fh)["predictions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", help="metrics JSON of model A")
    parser.add_argument("b", help="metrics JSON of model B")
    args = parser.parse_args()

    a, b = load_predictions(args.a), load_predictions(args.b)
    if set(a) != set(b):
        raise SystemExit("the two files were evaluated on different videos (different seeds?)")
    ids = sorted(a)
    if any(a[i]["label"] != b[i]["label"] for i in ids):
        raise SystemExit("labels disagree between the two files")
    r = delong_test([a[i]["label"] for i in ids],
                    [a[i]["fake_probability"] for i in ids],
                    [b[i]["fake_probability"] for i in ids])
    print(f"| {os.path.basename(args.a)} | {os.path.basename(args.b)} | {len(ids)} "
          f"| {r['auc_a']:.3f} | {r['auc_b']:.3f} | {r['z']:+.2f} | {r['p_value']:.3f} |")


if __name__ == "__main__":
    main()
