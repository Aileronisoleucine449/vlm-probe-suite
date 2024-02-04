#!/usr/bin/env bash
# Reproduces Table 2 in the report.
set -euo pipefail
python -m finevlm_probe.runners.sweep configs/main_sweep.yaml --out runs/table2/
python -m finevlm_probe.reporting.aggregate runs/table2/*.json --format markdown > tables/table2.md
echo "wrote tables/table2.md"
