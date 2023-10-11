"""count-coarse: 1 / 3 / 5+ object count discrimination."""
from __future__ import annotations
from collections import Counter, defaultdict
from .base import Probe


def _bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "3"
    return "5+"


class CountCoarseProbe(Probe):
    name = "count-coarse"
    BUCKETS = ("1", "3", "5+")
    TEMPLATES = ("a photo of one {obj}", "a photo of three {obj}", "a photo of five or more {obj}")

    def prepare(self, dataset):
        self.samples = list(dataset.iter_count_samples())

    def step(self, sample, model):
        scores = []
        for tpl in self.TEMPLATES:
            txt = tpl.format(obj=sample.object_class)
            scores.append(model.score(sample.image, txt))
        pred_idx = max(range(3), key=lambda i: scores[i])
        pred = self.BUCKETS[pred_idx]
        gold = _bucket(sample.count)
        return {
            "sample_id": sample.id,
            "object_class": sample.object_class,
            "gold": gold,
            "pred": pred,
            "correct": int(pred == gold),
        }

    def summarize(self, records):
        records = list(records)
        n = len(records)
        if n == 0:
            return {"n": 0, "accuracy": float("nan")}
        # macro-F1 over buckets
        tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
        for r in records:
            if r["pred"] == r["gold"]:
                tp[r["gold"]] += 1
            else:
                fp[r["pred"]] += 1
                fn[r["gold"]] += 1
        f1s = []
        for b in self.BUCKETS:
            p = tp[b] / (tp[b] + fp[b]) if (tp[b] + fp[b]) else 0.0
            rc = tp[b] / (tp[b] + fn[b]) if (tp[b] + fn[b]) else 0.0
            f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
            f1s.append(f1)
        return {
            "n": n,
            "accuracy": sum(r["correct"] for r in records) / n,
            "macro_f1": sum(f1s) / len(f1s),
            "per_bucket": dict(Counter(r["gold"] for r in records)),
        }
