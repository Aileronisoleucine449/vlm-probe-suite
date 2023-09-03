"""Sweep runner: dispatch one run per (model, probe, dataset) cell from YAML."""
from __future__ import annotations
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--out", default="runs/")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpus", default=None, help="comma list; defaults to CUDA_VISIBLE_DEVICES")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = cfg["models"]
    probes = cfg["probes"]
    datasets = cfg.get("datasets", ["coco-subset"])

    gpu_list = (args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")
    idx = 0
    for m in models:
        for pr in probes:
            for ds in datasets:
                tag = f"{m}__{pr}__{ds}.json"
                out_path = Path(args.out) / tag
                if out_path.exists():
                    log.info("skip (exists): %s", out_path)
                    continue
                gpu = gpu_list[idx % len(gpu_list)]
                idx += 1
                cmd = [
                    sys.executable, "-m", "finevlm_probe.runners.run_one",
                    "--model", m, "--probe", pr, "--dataset", ds,
                    "--out", str(out_path),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                log.info("$ CUDA_VISIBLE_DEVICES=%s %s", gpu, " ".join(cmd))
                if args.dry_run:
                    continue
                subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
