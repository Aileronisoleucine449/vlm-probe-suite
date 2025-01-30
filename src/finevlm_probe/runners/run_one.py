"""Run a single (model, probe, dataset) combination and write JSON."""
from __future__ import annotations
import argparse
import json
import logging
import os
import time
from pathlib import Path

from ..models import get_model
from ..probes import get_probe
from ..datasets import get_dataset

log = logging.getLogger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap samples for dev/debug")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("model=%s probe=%s dataset=%s", args.model, args.probe, args.dataset)

    model = get_model(args.model, device=args.device)
    probe = get_probe(args.probe)
    dataset = get_dataset(args.dataset)

    t0 = time.time()
    probe.prepare(dataset)
    samples = probe.samples
    if args.max_samples:
        samples = samples[:args.max_samples]

    records = []
    for i, s in enumerate(samples):
        records.append(probe.step(s, model))
        if (i + 1) % 100 == 0:
            log.info("  %d/%d", i + 1, len(samples))

    summary = probe.summarize(records)
    elapsed = time.time() - t0

    out = {
        "model": model.name,
        "probe": probe.name,
        "dataset": dataset.name,
        "summary": summary,
        "n_records": len(records),
        "elapsed_s": elapsed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote %s in %.1fs", args.out, elapsed)


if __name__ == "__main__":
    main()

# if the caller already configured logging, basicConfig is a no-op (good)
