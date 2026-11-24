"""
Aggregate runs written by scripts/run_prior_ablation.sh.

Expects   logs/prior_ablation/<dataset>/<setting>/seed<k>/metrics.json
Prints    mean ± std over seeds for the test metrics, one block per dataset.

Usage (from the rmf/ directory):
    python scripts/summarize_runs.py [--root logs/prior_ablation] [--csv results.csv]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = ["test/mmd-001", "test/mmd_lap-001", "test/mmd-002", "test/mmd_lap-002",
           "test/mmd-004", "test/mmd_lap-004", "test/mmd_prior", "test/mmd_lap_prior"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/prior_ablation")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.root, "*", "*", "seed*", "metrics.json"))):
        parts = path.split(os.sep)
        dataset, setting = parts[-4], parts[-3]
        with open(path) as f:
            runs[(dataset, setting)].append(json.load(f))
    if not runs:
        print(f"No runs found under {args.root}")
        return

    rows = []
    for dataset in sorted({d for d, _ in runs}):
        print(f"\n## {dataset}")
        header = f"{'setting':34s} {'n':>2s} " + " ".join(f"{m.split('/')[-1]:>17s}" for m in METRICS)
        print(header)
        for (d, setting), res in sorted(runs.items()):
            if d != dataset:
                continue
            cells = []
            for m in METRICS:
                vals = np.array([r[m] for r in res if m in r], dtype=float)
                if len(vals) == 0:
                    cells.append(f"{'-':>17s}")
                    continue
                mean, std = np.nanmean(vals), np.nanstd(vals)
                n_nan = int(np.isnan(vals).sum())
                cells.append(f"{mean:8.4f}±{std:6.4f}" + ("*" if n_nan else " "))
                rows.append(dict(dataset=dataset, setting=setting, metric=m, mean=mean, std=std,
                                 n=len(vals), n_nan=n_nan))
            print(f"{setting:34s} {len(res):2d} " + " ".join(cells))
    print("\n* = some seeds gave NaN (negative MMD^2 under the geodesic Gaussian kernel)")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
