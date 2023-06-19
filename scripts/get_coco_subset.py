#!/usr/bin/env python
"""Download a 2k-image subset of COCO val2017 into data/coco_val2017/.

We pull only the image IDs listed in configs/coco_subset.json. We do not
download annotations -- we ship our own minimal annotations in that JSON.
"""
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "configs" / "coco_subset.json"
OUT = ROOT / "data" / "coco_val2017"
URL_TEMPLATE = "http://images.cocodataset.org/val2017/{name}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SUBSET) as f:
        cfg = json.load(f)
    names = set()
    for entry in cfg.get("attr_swap_pairs", []) + cfg.get("count_samples", []):
        names.add(entry["file_name"])
    n = len(names)
    print(f"downloading {n} images to {OUT}")
    for i, name in enumerate(sorted(names), 1):
        dst = OUT / name
        if dst.exists():
            continue
        url = URL_TEMPLATE.format(name=name)
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"  failed {name}: {e}", file=sys.stderr)
        if i % 50 == 0:
            print(f"  {i}/{n}")
    print("done")


if __name__ == "__main__":
    main()
