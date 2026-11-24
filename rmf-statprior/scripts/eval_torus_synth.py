"""
Evaluation of the dimension-scaling study (synthetic torus mixtures tsynth_d*, exact densities).

For every run and for the prior / 1 / 2 / 4 sampling steps it reports
  gap_per_dim  (E_data[log p] - E_samples[log p]) / d  in nats per dimension (0 = perfect);
  below_floor  fraction of samples whose true density is below exp(-20) x uniform (clipped there);
  kl_per_dim   KL(p || uniform) / d of the dataset (how concentrated the data are).
Samples outside the support are counted with log p clipped at (uniform log-density - 20 nats) so that the
mean stays finite; off_support shows how many there are.

    python scripts/eval_torus_synth.py                 # all tsynth_d* runs under logs/
    python scripts/eval_torus_synth.py --reps 5 --csv dim_eval.csv
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import torus_synth as TS
from eval_synth import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--csv", default="dim_eval.csv")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(2026)

    rows, truth = [], {}
    for f in sorted(glob.glob(os.path.join(a.root, "*", "test_metrics.json"))):
        meta = json.load(open(f))
        name = meta.get("dataset", "")
        d = TS.dim_of(name)
        if d is None:
            continue
        run = os.path.dirname(f)
        model, ckpt, cfg = load_model(run, device, a.ckpt)
        if model is None or cfg.trainer.max_epochs < a.min_epochs:
            continue
        prior = "uniform" if meta["prior_type"] == "uniform" else \
            f"gmm(K={meta.get('prior_k_selected')},s={meta.get('prior_cov_scale')},{meta.get('prior_coupling')})"
        print(f"[eval] {os.path.basename(run)} {name} {prior} seed={cfg.seed} ckpt={ckpt}", flush=True)
        if name not in truth:
            xt = TS.sample(name, 20000, rng)
            ll_true = TS.logpdf(name, xt).mean()
            truth[name] = (ll_true, ll_true + d * math.log(2 * math.pi))
        ll_true, kl = truth[name]
        floor = -d * math.log(2 * math.pi) - 20.0

        def evaluate(x, label):
            lp = TS.logpdf(name, x)
            below = (~np.isfinite(lp)) | (lp < floor)       # density below exp(-20) x uniform: effectively off the data
            lp = np.maximum(np.nan_to_num(lp, neginf=floor), floor)
            rows.append(dict(d=d, dataset=name, prior=prior, seed=cfg.seed, steps=label,
                             gap_per_dim=(ll_true - lp.mean()) / d, below_floor=below.mean(), kl_per_dim=kl / d))

        with torch.no_grad():
            for _ in range(a.reps):
                z = model._prior_samples((a.n, d), device)
                evaluate(z.detach().cpu().double().numpy() % (2 * math.pi), "prior")
                for s in a.steps:
                    out = torch.cat([model.sample_batch(steps=s, sz=min(4096, a.n - j)) for j in range(0, a.n, 4096)])
                    evaluate(out.detach().cpu().double().numpy() % (2 * math.pi), f"{s}-step")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no tsynth_d* runs found")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    g = df.groupby(["d", "steps", "prior"])
    table = g[["gap_per_dim", "below_floor", "kl_per_dim"]].mean()
    table["gap_sd"] = g["gap_per_dim"].std()
    table["n_evals"] = g.size()
    print("\nLog-likelihood lost per dimension (nats; 0 = perfect). below_floor: fraction of samples whose density is"
          "\nbelow exp(-20) times the uniform density; their log-density is clipped there, so when below_floor is"
          "\nlarge, gap_per_dim is only a lower bound on the true loss.")
    print(table.round(4).to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
