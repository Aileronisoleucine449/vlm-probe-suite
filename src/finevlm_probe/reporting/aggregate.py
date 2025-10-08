"""Aggregate per-cell JSON outputs into a single table (markdown or LaTeX)."""
from __future__ import annotations
import argparse
import glob
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    p.add_argument("--metric", default="accuracy")
    args = p.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))

    cells = defaultdict(dict)  # (model, probe) -> metric
    probes = set()
    models = []

    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        m, pr = r["model"], r["probe"]
        if m not in models:
            models.append(m)
        probes.add(pr)
        val = r["summary"].get(args.metric, float("nan"))
        cells[(m, pr)] = val

    probes = sorted(probes)
    if args.format == "markdown":
        header = "| model | " + " | ".join(probes) + " |"
        sep = "|---" * (len(probes) + 1) + "|"
        print(header); print(sep)
        for m in models:
            row = [m] + [f"{cells.get((m, pr), float('nan')):.3f}" for pr in probes]
            print("| " + " | ".join(row) + " |")
    else:
        # latex
        print("\\begin{tabular}{l" + "c" * len(probes) + "}")
        print("\\toprule")
        print("model & " + " & ".join(probes) + " \\\\")
        print("\\midrule")
        for m in models:
            row = [m] + [f"{cells.get((m, pr), float('nan')):.3f}" for pr in probes]
            print(" & ".join(row) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
