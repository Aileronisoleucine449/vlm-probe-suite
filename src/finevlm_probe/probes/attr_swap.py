"""attr-swap probe.

Given an image with two objects of different colors and a positive caption like
"red cube next to blue ball", we form a hard negative by swapping the colors:
"blue cube next to red ball". We check whether the model assigns higher
similarity to the positive than the swapped negative.
"""
from __future__ import annotations
import logging
from collections import defaultdict
from .base import Probe

log = logging.getLogger(__name__)


class AttrSwapProbe(Probe):
    name = "attr-swap"

    def prepare(self, dataset):
        self.samples = list(dataset.iter_pairs("attr-swap"))

    def step(self, sample, model):
        img = sample.image
        pos = sample.positive_caption
        neg = sample.negative_caption
        s_pos = model.score(img, pos)
        s_neg = model.score(img, neg)
        correct = s_pos > s_neg
        return {
            "sample_id": sample.id,
            "s_pos": s_pos,
            "s_neg": s_neg,
            "correct": int(correct),
            "margin": s_pos - s_neg,
        }

    def summarize(self, records):
        records = list(records)
        n = len(records)
        if n == 0:
            return {"n": 0, "accuracy": float("nan")}
        acc = sum(r["correct"] for r in records) / n
        margin = sum(r["margin"] for r in records) / n
        # crude per-color stratification if available
        by_color = defaultdict(list)
        for r in records:
            color = r.get("color_pair", "unk")
            by_color[color].append(r["correct"])
        return {
            "n": n,
            "accuracy": acc,
            "mean_margin": margin,
            "per_color_accuracy": {k: sum(v)/len(v) for k, v in by_color.items()},
        }
