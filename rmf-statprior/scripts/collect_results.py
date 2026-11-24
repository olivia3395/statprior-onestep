"""
Collect test_metrics.json files written by MeanFlowModule.on_test_end and print a summary
table (mean ± std over seeds) for each (dataset, prior setting, method).

    python scripts/collect_results.py                 # scans logs/
    python scripts/collect_results.py --root logs --csv results.csv
"""
import argparse, glob, json, os
import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="logs")
ap.add_argument("--csv", default=None)
a = ap.parse_args()

rows = []
for f in glob.glob(os.path.join(a.root, "**", "test_metrics.json"), recursive=True):
    r = json.load(open(f))
    cfg_path = os.path.join(os.path.dirname(f), ".hydra", "config.yaml")
    if os.path.exists(cfg_path):
        r["seed"] = yaml.safe_load(open(cfg_path)).get("seed")
    r["run_dir"] = os.path.dirname(f)
    rows.append(r)
if not rows:
    raise SystemExit(f"no test_metrics.json under {a.root}")

df = pd.DataFrame(rows)
def _label(r):
    if r["prior_type"] == "uniform":
        return "uniform"
    if r["prior_type"] == "gmm":
        return f"gmm(K={r.get('prior_k_selected')},s={r.get('prior_cov_scale')},{r['prior_coupling']})"
    return f"cluster(K={r['prior_k']},s={r['prior_kappa_scale']},{r['prior_coupling']})"
df["prior"] = df.apply(_label, axis=1)
metrics = [c for c in ["test/mmd-prior", "test/mmd-001", "test/mmd-002", "test/mmd-004",
                       "test/mmdlap-prior", "test/mmdlap-001", "test/mmdlap-004"] if c in df]
g = df.groupby(["dataset", "my_method", "prior"])[metrics]
summary = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
summary["n_seeds"] = g.size()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(summary.to_string())
if a.csv:
    df.to_csv(a.csv, index=False)
    print(f"\nall runs written to {a.csv}")
