"""
Evaluation of trained runs on the REAL Earth datasets (volcano, earthquake, flood, fire).

The true distribution is unknown, so we compare generated samples with the held-out
test split (reconstructed exactly: EarthDataModule splits with a fixed seed 42) and
calibrate every number against a noise floor built from real data only.

Reported per run and number of steps (prior / 1 / 2 / 4):
  mmd2_*   unbiased MMD^2 (x1e-3) between n_gen generated points and the test split,
           for the paper's kernel (gauss, gamma=1) and Laplace kernels at 3 scales.
  nn_ratio memorisation check: mean distance from generated points to their nearest
           TRAINING point, divided by the same quantity for TEST points. ~1 = fine,
           << 1 = samples hug the training data (memorisation).
Noise floor: std of MMD^2 between the test split and random real subsets drawn from
the rest of the data (a model that reproduces the data distribution scores ~0 +- floor).

    python scripts/eval_real.py                       # all Earth runs under logs/
    python scripts/eval_real.py --only fire --n_gen 20000
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
import numpy as np
import pandas as pd
import torch

import geodesic as GD  # numerically stable float64 distances

from eval_synth import KERNELS, _kmean, load_model  # same kernels / checkpoint logic

EARTH = {"volcano", "earthquake", "flood", "fire"}


def mmd2_unbiased(x, y, kind, gamma):
    return _kmean(x, x, kind, gamma, True) + _kmean(y, y, kind, gamma, True) - 2 * _kmean(x, y, kind, gamma, False)


def nn_mean(a, ref, block=1024):
    """Mean geodesic distance from each point of a to its nearest point in ref
    (mean rather than median: some Earth datasets contain repeated coordinates)."""
    d = []
    for i in range(0, len(a), block):
        d.append(GD.sphere_dist(a[i : i + block], ref).min(1).values)
    return torch.cat(d).mean().item()


def split_tensors(cfg):
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    test = dm.get_test_tensor().float()
    val = torch.stack([torch.as_tensor(dm.data_val[i]) for i in range(len(dm.data_val))]).float()
    tr = dm.data_train.dset
    train = torch.stack([torch.as_tensor(tr[i]) for i in range(len(tr))]).float()
    return train[..., :3], val[..., :3], test[..., :3]



def method_label(method, cfg):
    """Objective label; for alpha-Flow the curriculum length (model.epoch) is part of the configuration."""
    method = str(method)
    return f"{method}(ke={cfg.model.get('epoch', 100)})" if method == "alpha-f" else method

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--n_gen", type=int, default=20000)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--floor_reps", type=int, default=20)
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--only", default=None, help="regex on dataset name")
    ap.add_argument("--min_epochs", type=int, default=20, help="skip smoke-test runs trained for fewer epochs")
    ap.add_argument("--csv", default="real_eval.csv")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="reference split: 'val' for model selection (e.g. the prior width s), 'test' for reporting")
    ap.add_argument("--reps", type=int, default=1, help="independent draws of generated samples per run")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)

    rows, floors = [], {}
    for f in sorted(glob.glob(os.path.join(a.root, "*", "test_metrics.json"))):
        run_dir = os.path.dirname(f)
        meta = json.load(open(f))
        name = meta.get("dataset", "")
        if name not in EARTH or (a.only and not re.search(a.only, name)):
            continue
        model, ckpt, cfg = load_model(run_dir, device, a.ckpt)
        if model is None or cfg.trainer.max_epochs < a.min_epochs:
            continue
        if meta["prior_type"] == "uniform":
            prior = "uniform"
        elif meta["prior_type"] == "gmm":
            prior = f"gmm(K={meta.get('prior_k_selected')},s={meta.get('prior_cov_scale')},{meta.get('prior_coupling')})"
        else:
            prior = f"cluster(K={meta['prior_k']})"
        print(f"[eval] {os.path.basename(run_dir)} {name} prior={prior} seed={cfg.seed} "
              f"epochs={cfg.trainer.max_epochs} method={method_label(cfg.model.get('my_method', 'mf'), cfg)} ckpt={ckpt}")

        train, val, test = split_tensors(cfg)
        if a.split == "val":
            test = val  # compare with the validation split (model selection)
        train, test = train.to(device), test.to(device)

        if name not in floors:
            # real-vs-real floor: split ALL real points at random into A (test-set size) and B (the rest);
            # MMD^2(A, B) is what a model reproducing the data distribution would score.
            allx = torch.cat([train, val.to(device), test])
            n_t = len(test)
            reps = {f"{k}{gm:g}": [] for k, gm in KERNELS}
            for _ in range(a.floor_reps):
                perm = torch.randperm(len(allx), generator=g).to(device)
                A, B = allx[perm[:n_t]], allx[perm[n_t:]]
                for k, gm in KERNELS:
                    reps[f"{k}{gm:g}"].append(mmd2_unbiased(B, A, k, gm))
            floors[name] = {key: float(np.std(v)) for key, v in reps.items()}
            floors[name]["n_test"] = n_t
            floors[name]["_test_nn"] = nn_mean(test, train)

        def evaluate(samples, label):
            r = dict(dataset=name, epochs=int(cfg.trainer.max_epochs), method=method_label(cfg.model.get("my_method", "mf"), cfg),
                     prior=prior, seed=cfg.seed, steps=label, ckpt=ckpt)
            for k, gm in KERNELS:
                r[f"mmd2_{k}{gm:g}"] = mmd2_unbiased(samples, test, k, gm)
            r["nn_ratio"] = nn_mean(samples, train) / max(floors[name]["_test_nn"], 1e-12)
            rows.append(r)

        with torch.no_grad():
            for rep in range(a.reps):
                evaluate(model._prior_samples((a.n_gen, 3), device).float(), "prior")
                for s in a.steps:
                    out = torch.cat([model.sample_batch(steps=s, sz=min(4096, a.n_gen - i)) for i in range(0, a.n_gen, 4096)])
                    evaluate(out.float(), f"{s}-step")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no Earth runs found")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = [c for c in df.columns if c.startswith("mmd2_")] + ["nn_ratio"]
    grp = df.groupby(["dataset", "epochs", "method", "steps", "prior"])[cols]
    table = grp.mean()
    table[cols[:-1]] *= 1e3
    table["n_evals"] = grp.size()
    print("\nMMD^2 vs held-out test split (unbiased, x1e-3; ~0 = as good as real data) and memorisation ratio")
    print("mean over seeds x repeated sample draws (n_evals = seeds * reps):")
    print(table.round(3).to_string())
    sd = grp.std()
    sd[cols[:-1]] *= 1e3
    print("\nstandard deviation over seeds x repeated sample draws:")
    print(sd.round(3).to_string())
    fl = pd.DataFrame(floors).T.drop(columns=["_test_nn"])
    fl[[c for c in fl.columns if c != "n_test"]] *= 1e3
    print("\nNoise floor: std of MMD^2 between the test split and real data (x1e-3):")
    print(fl.round(3).to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
