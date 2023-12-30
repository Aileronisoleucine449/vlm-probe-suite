"""COCO val2017 subset loader.

We hold a fixed 2000-image subset list under configs/coco_subset.json. Each
image has a small attribute annotation produced by parsing COCO captions for
color+object patterns. Not perfect; we filter aggressively.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from PIL import Image
from .base import Dataset, PairSample, CountSample


class CocoSubset(Dataset):
    name = "coco-subset"

    def __init__(self, root: str = "data/coco_val2017", subset_json: str = "configs/coco_subset.json"):
        self.root = Path(root)
        with open(subset_json) as f:
            self.subset = json.load(f)

    def iter_pairs(self, probe_name: str):
        anns = self.subset.get("attr_swap_pairs", [])
        for a in anns:
            img_path = self.root / a["file_name"]
            if not img_path.exists():
                continue
            yield PairSample(
                id=a["id"],
                image=Image.open(img_path),
                positive_caption=a["positive"],
                negative_caption=a["negative"],
            )

    def iter_count_samples(self):
        for a in self.subset.get("count_samples", []):
            img_path = self.root / a["file_name"]
            if not img_path.exists():
                continue
            yield CountSample(
                id=a["id"],
                image=Image.open(img_path),
                object_class=a["object_class"],
                count=int(a["count"]),
            )

# Anns are derived from captions; not all COCO images have usable color+object patterns.
