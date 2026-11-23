"""Build a training manifest from a DFDC-style ``metadata.json``.

DFDC ships one ``metadata.json`` per folder mapping each video to its label and,
for fakes, the ``original`` real video it was made from. That original becomes
the manifest ``group``, so ``split_videos`` keeps every fake in the same split
as its source (docs/datasets.md).

Example:
    python -m preprocessing.make_manifest \
        --metadata data/raw/dfdc/train_sample_videos/metadata.json \
        --output data/raw/dfdc/manifest.csv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from preprocessing.splits import VideoItem, write_manifest
from training.config import FAKE, REAL


def dfdc_items(metadata_path: str) -> List[VideoItem]:
    folder = os.path.dirname(metadata_path)
    with open(metadata_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    items = []
    for name in sorted(meta):
        entry = meta[name]
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            print(f"[manifest] WARNING: {path} listed in metadata but missing; skipped")
            continue
        label = FAKE if entry["label"].upper() == "FAKE" else REAL
        source = entry.get("original") if label == FAKE else name
        stem = os.path.splitext(name)[0]
        items.append(VideoItem(path, label, f"dfdc__{stem}",
                               os.path.splitext(source or name)[0]))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", required=True, help="DFDC metadata.json")
    parser.add_argument("--output", required=True, help="manifest CSV to write")
    args = parser.parse_args()
    items = dfdc_items(args.metadata)
    write_manifest(items, args.output)
    real = sum(i.label == REAL for i in items)
    print(f"[manifest] {len(items)} videos (REAL={real}, FAKE={len(items) - real}, "
          f"source groups={len({i.group_key for i in items})}) -> {args.output}")


if __name__ == "__main__":
    main()
