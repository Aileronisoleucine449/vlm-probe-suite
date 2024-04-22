#!/usr/bin/env bash
set -euo pipefail
python -m finevlm_probe.runners.sweep configs/resolution_sweep.yaml --out runs/table3/
python -m finevlm_probe.reporting.aggregate runs/table3/*.json --format markdown > tables/table3.md
