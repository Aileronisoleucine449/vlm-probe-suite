"""Smoke test for the aggregator output formatting."""
import json
import subprocess
import sys
from pathlib import Path


def test_aggregate_markdown(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    json.dump({"model": "clip_vit_b32", "probe": "attr-swap", "dataset": "coco-subset",
               "summary": {"accuracy": 0.61}, "n_records": 100, "elapsed_s": 1.0}, open(f1, "w"))
    json.dump({"model": "clip_vit_b32", "probe": "count-coarse", "dataset": "coco-subset",
               "summary": {"accuracy": 0.42}, "n_records": 100, "elapsed_s": 1.0}, open(f2, "w"))
    r = subprocess.run(
        [sys.executable, "-m", "finevlm_probe.reporting.aggregate",
         str(f1), str(f2), "--format", "markdown"],
        capture_output=True, text=True, check=True,
    )
    assert "attr-swap" in r.stdout
    assert "0.610" in r.stdout
