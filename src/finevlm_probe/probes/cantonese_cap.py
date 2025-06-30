"""cantonese-cap: top-1 retrieval over Cantonese captions.

We form a pool of K=4 candidate Cantonese captions per image (1 positive,
3 hard negatives from the same object class) and ask the model to rank them.
"""
from __future__ import annotations
from .base import Probe


class CantoneseCapProbe(Probe):
    name = "cantonese-cap"
    K = 4

    def prepare(self, dataset):
        self.samples = list(dataset.iter_canto_samples())

    def step(self, sample, model):
        scores = [model.score(sample.image, c) for c in sample.candidates]
        pred = scores.index(max(scores))
        return {"sample_id": sample.id, "pred": pred,
                "gold": sample.positive_idx,
                "correct": int(pred == sample.positive_idx)}

    def summarize(self, records):
        records = list(records)
        n = len(records)
        if n == 0:
            return {"n": 0, "accuracy": float("nan")}
        return {"n": n, "accuracy": sum(r["correct"] for r in records) / n}

# updated docstring; K was 4 from the start but earlier doc said 5
