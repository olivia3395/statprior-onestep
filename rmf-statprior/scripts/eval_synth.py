"""
Large-sample re-evaluation of trained runs on the synthetic sphere datasets
(vMF blobs synth_k*, rings synth_ring*, arcs synth_arc*; ground truth in synth_shapes.py).

Why: with 1000 test points, the original MMD (gaussian geodesic kernel, gamma=1)
has a noise floor of ~0.04 +- 0.015, which is larger than the differences between
methods. For synthetic data the true distribution is known, so we can
  * draw as many fresh ground-truth samples as we like,
  * use kernels at several length scales (gamma=1 is ~1 rad and cannot see the
    structure of concentrated data),
  * report the exact noise floor next to each number,
  * compute the true log-likelihood of generated samples.

No retraining: loads the best checkpoint of every run under --root.

    python scripts/eval_synth.py                      # all synth runs under logs/
    python scripts/eval_synth.py --n 10000 --csv synth_eval.csv
    python scripts/eval_synth.py --only 'ring|arc' --csv shapes_eval.csv   # just the new shapes

Uses last.ckpt when present (fixed training budget); --ckpt best uses the val-MMD-selected one.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make `gfm` importable

import hydra
import numpy as np
import pandas as pd
import torch

import geodesic as GD  # numerically stable float64 distances
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_shapes as S

# ------------------------------------------------------------------------ MMD
KERNELS = [("gauss", 1.0), ("lap", 1.0), ("lap", 10.0), ("lap", 50.0)]


def _kmean(a, b, kind, gamma, same, block=2048):
    """Mean of k(a_i, b_j) over all pairs (excluding i==j if same), blockwise, in float64 with stable distances."""
    total, count = 0.0, 0
    for i in range(0, len(a), block):
        ai = a[i : i + block]
        d = GD.sphere_dist(ai, b)
        k = torch.exp(-gamma * d**2) if kind == "gauss" else torch.exp(-gamma * d)
        if same:
            idx = torch.arange(ai.shape[0], device=a.device)
            k[idx, idx + i] = 0.0
        total += k.sum().item()
        count += k.numel() - (ai.shape[0] if same else 0)
    return total / count


def mmd2_unbiased(x, y, kind, gamma):
    return _kmean(x, x, kind, gamma, True) + _kmean(y, y, kind, gamma, True) - 2 * _kmean(x, y, kind, gamma, False)


# ------------------------------------------------------------------ main loop
def load_model(run_dir, device, which="last"):
    """which='last' uses last.ckpt when it exists (fixed training budget, no noisy selection);
    otherwise falls back to the best-by-val-MMD checkpoint."""
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    model = hydra.utils.instantiate(cfg.model)
    ckdir = os.path.join(run_dir, "checkpoints")
    last = os.path.join(ckdir, "last.ckpt")
    best = sorted(f for f in glob.glob(os.path.join(ckdir, "*.ckpt")) if not f.endswith("last.ckpt"))
    path = last if (which == "last" and os.path.exists(last)) else (best[-1] if best else None)
    if path is None:
        return None, None, cfg
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval(), os.path.basename(path), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--n", type=int, default=10000, help="generated and ground-truth sample size")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--floor_reps", type=int, default=5)
    ap.add_argument("--csv", default="synth_eval.csv")
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--reps", type=int, default=1, help="independent draws of generated samples per run")
    ap.add_argument("--only", default=None, help="regex on dataset name, e.g. 'ring|arc'")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(2026)

    rows, floors = [], {}
    run_dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(a.root, "*", "test_metrics.json")))
    for run_dir in run_dirs:
        meta = json.load(open(os.path.join(run_dir, "test_metrics.json")))
        name = meta.get("dataset", "")
        if S.parse(name)[0] is None or (a.only and not re.search(a.only, name)):
            continue
        model, ckpt, cfg = load_model(run_dir, device, a.ckpt)
        if model is None:
            continue
        if meta["prior_type"] == "uniform":
            prior = "uniform"
        elif meta["prior_type"] == "gmm":
            prior = f"gmm(K={meta.get('prior_k_selected')},s={meta.get('prior_cov_scale')},{meta['prior_coupling']})"
        else:
            prior = f"cluster(K={meta['prior_k']})"
        print(f"[eval] {os.path.basename(run_dir)} {name} prior={prior} seed={cfg.seed} ckpt={ckpt}")

        truth = torch.tensor(S.sample(name, a.n, rng), dtype=torch.float32, device=device)

        # noise floor: fresh truth vs fresh truth, same sample size (cached per kappa)
        if name not in floors:
            reps = {f"{k}{g:g}": [] for k, g in KERNELS}
            for _ in range(a.floor_reps):
                t2 = torch.tensor(S.sample(name, a.n, rng), dtype=torch.float32, device=device)
                for k, g in KERNELS:
                    reps[f"{k}{g:g}"].append(mmd2_unbiased(t2, truth, k, g))
            floors[name] = {key: float(np.std(v)) for key, v in reps.items()}
            floors[name]["ll_true"] = float(S.logpdf(name, truth.cpu().double().numpy()).mean())

        def evaluate(samples, label):
            r = dict(dataset=name, prior=prior, seed=cfg.seed, steps=label, ckpt=ckpt)
            for k, g in KERNELS:
                r[f"mmd2_{k}{g:g}"] = mmd2_unbiased(samples, truth, k, g)
            r["ll_gen"] = float(S.logpdf(name, samples.cpu().double().numpy()).mean())
            rows.append(r)

        with torch.no_grad():
            for rep in range(a.reps):
                evaluate(model._prior_samples((a.n, 3), device).float(), "prior")
                for s in a.steps:
                    out = torch.cat([model.sample_batch(steps=s, sz=min(4096, a.n - i)) for i in range(0, a.n, 4096)])
                    evaluate(out.float(), f"{s}-step")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    if not rows:
        raise SystemExit("no synthetic runs with checkpoints found")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    metric_cols = [c for c in df.columns if c.startswith("mmd2_")] + ["ll_gen"]
    g = df.groupby(["dataset", "steps", "prior"])[metric_cols]
    table = g.mean()
    table[metric_cols[:-1]] *= 1e3  # MMD^2 in units of 1e-3
    print("\nMMD^2 (unbiased, x1e-3; ~0 = indistinguishable from truth) and mean true log-likelihood of samples")
    print(table.round(3).to_string())
    sd = g.std()
    sd[metric_cols[:-1]] *= 1e3
    print("\nstandard deviation over seeds x repeated sample draws:")
    print(sd.round(3).to_string())
    print("\nNoise floor: std of MMD^2 between two independent true samples of the same size (x1e-3),")
    print("and mean true log-likelihood of real samples (target for ll_gen):")
    fl = pd.DataFrame(floors).T
    fl[[c for c in fl.columns if c != "ll_true"]] *= 1e3
    print(fl.round(3).to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
