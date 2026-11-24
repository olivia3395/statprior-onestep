"""
Rigorous evaluation of trained runs on SO(3) (cone, fisher24, line, swiss) and on the
flat torus (protein Ramachandran data: General, Glycine, Proline, Pre-Pro; RNA 7-D).

Compared with the training-time metric this script
  * reconstructs the exact held-out test split (same seeding as training),
  * generates many samples (default 20000) instead of |test| samples,
  * reports unbiased MMD^2 for several kernels:
        paper   the kernel used in the papers: geodesic Gaussian exp(-d^2) on SO(3); on the
                torus the code base averages 1-D kernels per angle, i.e. an ADDITIVE kernel that
                only sees marginals (kept for comparison with the published numbers)
        lap{g}  joint Laplace kernels at several scales:
                SO(3): exp(-g * angle);  torus: prod_i exp(-g * |wrap(dx_i)|), positive definite
                and sensitive to the dependence between angles
  * calibrates each number with a real-vs-real noise floor, and
  * checks memorisation (nn_ratio, see eval_real.py).

    python scripts/eval_manifold.py                          # every SO(3) and torus run under logs/
    python scripts/eval_manifold.py --only "cone|fisher24"   # subset (regex on dataset name)
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch

import geodesic as GD  # numerically stable float64 distances

from eval_synth import load_model

SO3_SETS = {"cone", "fisher24", "line", "swiss", "peak"}
TORUS_SETS = {"General", "Glycine", "Proline", "Pre-Pro", "RNA"}
GAMMAS = {"so3": (1.0, 5.0, 20.0), "torus": (1.0, 3.0, 10.0)}


# ------------------------------------------------------------------- distances
def _wrap(d):
    return torch.remainder(d + math.pi, 2 * math.pi) - math.pi


def kernel_block(a, b, space, kind, gamma):
    """k(a_i, b_j) for a block of a against all of b."""
    if space == "so3":
        if kind == "paper":                                   # geodesic Gaussian of the RMF protocol (not PD)
            return torch.exp(-gamma * GD.so3_angle(a, b) ** 2)
        # chordal Laplace kernel exp(-gamma ||R - R'||_F / sqrt 2): the Euclidean Laplace kernel of R^9 restricted
        # to SO(3), hence positive definite and characteristic. ||R - R'||_F = 2 sqrt 2 sin(theta / 2), so for
        # small angles it coincides with exp(-gamma theta). (The geodesic Laplace kernel exp(-gamma theta) is NOT
        # positive definite on SO(3): its even Peter-Weyl coefficients become negative at high degree.)
        return torch.exp(-gamma * GD.so3_chord(a, b) / math.sqrt(2.0))
    diff = _wrap(a[:, None, :] - b[None, :, :]).abs()  # (na, nb, dim)
    if kind == "paper":  # additive kernel of the code base: mean over angles of exp(-d_i^2)
        return torch.exp(-gamma * diff**2).mean(-1)
    return torch.exp(-gamma * diff.sum(-1))  # product of 1-D Laplace kernels (joint, PD)


def kmean(a, b, space, kind, gamma, same, block=None):
    block = block or (512 if space == "torus" else 2048)
    a, b = a.double(), b.double()
    tot, cnt = 0.0, 0
    for i in range(0, len(a), block):
        k = kernel_block(a[i : i + block], b, space, kind, gamma)
        if same:
            idx = torch.arange(k.shape[0], device=a.device)
            k[idx, idx + i] = 0.0
        tot += k.sum().item()
        cnt += k.numel() - (k.shape[0] if same else 0)
    return tot / cnt


def mmd2(x, y, space, kind, gamma):
    return kmean(x, x, space, kind, gamma, True) + kmean(y, y, space, kind, gamma, True) \
        - 2 * kmean(x, y, space, kind, gamma, False)


def nn_mean(a, ref, space, block=2048):
    out = []
    for i in range(0, len(a), block):
        ai = a[i : i + block]
        if space == "so3":
            d = GD.so3_angle(ai, ref)
        else:
            d = _wrap(ai[:, None] - ref[None]).norm(dim=-1)
        out.append(d.min(1).values)
    return torch.cat(out).mean().item()


def kernels(space):
    return [("paper", 1.0)] + [("lap", g) for g in GAMMAS[space]]


# ----------------------------------------------------------------------- data
def split_tensors(cfg):
    if cfg.get("seed") is not None:  # SO(3) data are shuffled with the run seed before the fixed split
        L.seed_everything(cfg.seed, workers=True)
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    test = dm.get_test_tensor().float()
    tr = getattr(dm.data_train, "dset", dm.data_train)
    train = torch.stack([torch.as_tensor(tr[i]) for i in range(len(tr))]).float()
    val = torch.stack([torch.as_tensor(dm.data_val[i]) for i in range(len(dm.data_val))]).float()
    return train, val, test


def dataset_name(meta):
    return meta.get("dataset", "")



def method_label(method, cfg):
    """Objective label; for alpha-Flow the curriculum length (model.epoch) is part of the configuration."""
    method = str(method)
    return f"{method}(ke={cfg.model.get('epoch', 100)})" if method == "alpha-f" else method

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--n_gen", type=int, default=20000)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--floor_reps", type=int, default=10)
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--only", default=None)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--csv", default="manifold_eval.csv")
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
        name = dataset_name(meta)
        space = "so3" if name in SO3_SETS else "torus" if (name in TORUS_SETS or str(name).startswith("pepwin")) else None
        if space is None or (a.only and not re.search(a.only, name)):
            continue
        model, ckpt, cfg = load_model(run_dir, device, a.ckpt)
        if model is None or cfg.trainer.max_epochs < a.min_epochs:
            continue
        if meta["prior_type"] == "uniform":
            prior = "uniform"
        elif meta["prior_type"] == "gmm":
            prior = f"gmm(K={meta.get('prior_k_selected')},s={meta.get('prior_cov_scale')},{meta['prior_coupling']})"
        else:
            prior = meta["prior_type"]
        print(f"[eval] {os.path.basename(run_dir)} {name} prior={prior} seed={cfg.seed} "
              f"epochs={cfg.trainer.max_epochs} method={method_label(meta.get('my_method'), cfg)} ckpt={ckpt}")

        train, val, test = split_tensors(cfg)
        if a.split == "val":
            test = val  # compare with the validation split (model selection)
        train, test = train.to(device), test.to(device)
        key = (name, cfg.seed) if space == "so3" else (name, None)  # SO(3) split depends on the seed
        if key not in floors:
            allx = torch.cat([train, val.to(device), test])
            n_t = len(test)
            reps = {f"{k}{gm:g}": [] for k, gm in kernels(space)}
            for _ in range(a.floor_reps):
                perm = torch.randperm(len(allx), generator=g).to(device)
                A, B = allx[perm[:n_t]], allx[perm[n_t : n_t + a.n_gen]]
                for k, gm in kernels(space):
                    reps[f"{k}{gm:g}"].append(mmd2(B, A, space, k, gm))
            ref_train = train[torch.randperm(len(train), generator=g)[:50000].to(device)]
            floors[key] = {kk: float(np.std(v)) for kk, v in reps.items()}
            floors[key].update(n_test=n_t, _ref=ref_train, _test_nn=nn_mean(test, ref_train, space))

        fl = floors[key]

        def evaluate(samples, label):
            r = dict(dataset=name, epochs=int(cfg.trainer.max_epochs), method=method_label(meta.get("my_method"), cfg),
                     prior=prior, seed=cfg.seed, steps=label, ckpt=ckpt)
            for k, gm in kernels(space):
                r[f"mmd2_{k}{gm:g}"] = mmd2(samples, test, space, k, gm)
            r["nn_ratio"] = nn_mean(samples, fl["_ref"], space) / max(fl["_test_nn"], 1e-12)
            rows.append(r)

        with torch.no_grad():
            shape = (a.n_gen,) + tuple(cfg.model.in_shape)
            for rep in range(a.reps):
                z0 = model._prior_samples(shape, device).float()
                evaluate(z0.reshape((a.n_gen,) + tuple(test.shape[1:])), "prior")
                for s in a.steps:
                    out = torch.cat([model.sample_batch(steps=s, sz=min(4096, a.n_gen - i)) for i in range(0, a.n_gen, 4096)])
                    evaluate(out.float().reshape((a.n_gen,) + tuple(test.shape[1:])), f"{s}-step")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no SO(3) / torus runs found")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for name, part in df.groupby("dataset"):
        cols = [c for c in part.columns if c.startswith("mmd2_") and part[c].notna().any()] + ["nn_ratio"]
        grp = part.groupby(["dataset", "epochs", "method", "steps", "prior"])[cols]
        table = grp.mean()
        table[cols[:-1]] *= 1e3
        table["n_evals"] = grp.size()
        print(f"\n=== {name}: unbiased MMD^2 vs held-out test split (x1e-3; ~0 = as good as real data)")
        print("mean over seeds x repeated sample draws:")
        print(table.round(3).to_string())
        sd = grp.std()
        sd[cols[:-1]] *= 1e3
        print("standard deviation:")
        print(sd.round(3).to_string())
    print("\nNoise floor: std of MMD^2 between the test split and real data (x1e-3):")
    fl = pd.DataFrame({f"{k[0]}" + (f" (seed {k[1]})" if k[1] is not None else ""):
                       {kk: v for kk, v in d.items() if not kk.startswith("_")} for k, d in floors.items()}).T
    fl[[c for c in fl.columns if c != "n_test"]] *= 1e3
    print(fl.round(3).to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
