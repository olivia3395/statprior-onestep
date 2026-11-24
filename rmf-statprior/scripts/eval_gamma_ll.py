"""
Two complements to the MMD tables, on existing checkpoints (no training):

 (A) Bandwidth sensitivity. One-step MMD^2 against the test split for a grid of kernel bandwidths gamma, with the
     real-vs-real noise floor at each gamma. Kernels are those of the paper: chordal Laplace on SO(3), geodesic
     Laplace on S^2, product Laplace on the torus (all positive definite and characteristic).

 (B) Log-likelihood proxies, in nats per point. With a geodesic KDE of bandwidth h (chosen by validation
     log-likelihood of the test-split KDE):
       precision  = mean log p_test(x),   x ~ generated     (do samples lie where the data are?)
       coverage   = mean log p_gen(y),    y ~ test split    (does the model cover all of the data?)
     For coverage, p_gen is a Parzen estimate on the generated samples whose bandwidth is chosen by cross-validation
     on the generated samples themselves (a fixed bandwidth would reward over-concentrated models).
     Both are reported relative to a perfect model, i.e. training points in place of generated samples (processed
     identically), so that 0 = as good as real data and negative values are nats lost per point.

Runs: main-table settings only, with the main training budget of each dataset, and the newest run per
(dataset, prior, seed), as in the other evaluation scripts.

    python scripts/eval_gamma_ll.py --datasets cone Pre-Pro fire --csv gamma_ll.csv > eval_gamma_ll.log 2>&1
"""
import argparse
import glob
import json
import math
import os
import sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT)
sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import eval_manifold as EM
import eval_real as ER
import kde_baseline as KB
from eval_synth import load_model

GAMMAS = {"so3": [0.3, 1, 3, 5, 10, 20, 50, 100], "sphere": [0.3, 1, 3, 10, 30, 50, 100, 200],
          "torus": [0.1, 0.3, 1, 3, 10, 30]}
DIM = {"so3": 3, "sphere": 2}
# main-table settings: uniform base, statistical base with posterior coupling, and with independent coupling
KEEP = {"cone": ("gmm(K=3,s=2.0,", ), "Pre-Pro": ("gmm(K=8,s=2.0,", ), "fire": ("gmm(K=32,s=2.0,", )}
EPOCHS = {"cone": 200, "Pre-Pro": 1000, "fire": 100}          # training budget of the main tables
HGRID = np.exp(np.linspace(np.log(0.003), np.log(1.0), 24))


def space_of(name):
    if name in ER.EARTH:
        return "sphere"
    return "so3" if name in EM.SO3_SETS else "torus"


def mmd2(x, y, space, gamma):
    if space == "sphere":
        return ER.mmd2_unbiased(x, y, "lap", gamma)
    return EM.mmd2(x, y, space, "lap", gamma)


def splits(cfg, space):
    return ER.split_tensors(cfg) if space == "sphere" else EM.split_tensors(cfg)


def parzen_coverage(sample, test, space, d, g):
    """Coverage proxy: mean log-density of the test split under a Parzen estimate fitted on `sample`, with the
    bandwidth chosen by held-out log-likelihood within `sample` (80/20 split)."""
    perm = torch.randperm(len(sample), generator=g).to(sample.device)
    n_fit = int(0.8 * len(sample))
    fit, held = sample[perm[:n_fit]], sample[perm[n_fit:]]
    h = float(HGRID[int(np.argmax([KB.kde_loglik(held, fit, hh, space, d) for hh in HGRID]))])
    return KB.kde_loglik(test, fit, h, space, d), h


def label(meta, cfg):
    if meta["prior_type"] == "uniform":
        return "uniform"
    return f"gmm(K={meta.get('prior_k_selected')},s={meta.get('prior_cov_scale')},{meta.get('prior_coupling')})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--datasets", nargs="+", default=["cone", "Pre-Pro", "fire"])
    ap.add_argument("--n_gen", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--floor_reps", type=int, default=10)
    ap.add_argument("--csv", default="gamma_ll.csv")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    rows, ref = [], {}

    chosen = {}
    for f in sorted(glob.glob(os.path.join(a.root, "*", "test_metrics.json"))):
        meta = json.load(open(f))
        name = EM.dataset_name(meta) if "dataset" not in meta else meta["dataset"]
        if name not in a.datasets:
            continue
        run = os.path.dirname(f)
        cfg0 = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
        if str(cfg0.model.get("my_method", "mf")) != "mf":
            continue
        if name in EPOCHS and int(cfg0.trainer.max_epochs) != EPOCHS[name]:
            continue
        prior = label(meta, cfg0)
        if prior != "uniform" and not prior.startswith(KEEP.get(name, ("gmm(",))):
            continue
        chosen[(name, prior, int(cfg0.seed))] = (run, meta)       # sorted by time: the newest run wins
    print("runs used:", *[f"{k[0]} {k[1]} seed={k[2]}: {os.path.basename(v[0])}" for k, v in sorted(chosen.items())],
          sep="\n  ", flush=True)

    for (name, prior, _), (run, meta) in sorted(chosen.items(), key=lambda kv: kv[1][0]):
        space = space_of(name)
        model, ckpt, cfg = load_model(run, device, "last")
        if model is None:
            continue
        print(f"[eval] {os.path.basename(run)} {name} {prior} seed={cfg.seed}", flush=True)
        train, val, test = [t.to(device).float() for t in splits(cfg, space)]
        key = (name, cfg.seed) if space == "so3" else (name, None)
        d = DIM.get(space, int(np.prod(test.shape[1:])))

        if key not in ref:          # per split: noise floors per gamma, KDE bandwidth, perfect-model log-likelihoods
            allx = torch.cat([train, val, test])
            fl = {gm: [] for gm in GAMMAS[space]}
            for _ in range(a.floor_reps):
                perm = torch.randperm(len(allx), generator=g).to(device)
                A, B = allx[perm[:len(test)]], allx[perm[len(test):len(test) + a.n_gen]]
                for gm in GAMMAS[space]:
                    fl[gm].append(mmd2(B, A, space, gm))
            grid = np.exp(np.linspace(np.log(0.005), np.log(1.0), 24))
            vq = val[torch.randperm(len(val), generator=g)[:1500].to(device)]
            h = float(grid[int(np.argmax([KB.kde_loglik(vq.double(), test.double(), hh, space, d) for hh in grid]))])
            tr = train[torch.randperm(len(train), generator=g)[:a.n_gen].to(device)].double()
            cov_ref, h_ref = parzen_coverage(tr, test.double(), space, d, g)
            ref[key] = dict(floor={gm: float(np.std(v)) for gm, v in fl.items()}, h=h,
                            prec=KB.kde_loglik(tr, test.double(), h, space, d), cov=cov_ref, h_cov_ref=h_ref)
            print(f"   floors and KDE bandwidth h={h:.4f} rad", flush=True)
        R = ref[key]

        with torch.no_grad():
            for rep in range(a.reps):
                out = torch.cat([model.sample_batch(steps=1, sz=min(4096, a.n_gen - i)) for i in range(0, a.n_gen, 4096)])
                x = out.float().reshape((a.n_gen,) + tuple(test.shape[1:])).to(device)
                cov, h_gen = parzen_coverage(x.double(), test.double(), space, d, g)
                r = dict(dataset=name, prior=prior, seed=cfg.seed, rep=rep, run=os.path.basename(run),
                         epochs=int(cfg.trainer.max_epochs), h=R["h"], h_cov=h_gen, h_cov_ref=R["h_cov_ref"],
                         d_precision=KB.kde_loglik(x.double(), test.double(), R["h"], space, d) - R["prec"],
                         d_coverage=cov - R["cov"])
                for gm in GAMMAS[space]:
                    v = mmd2(x, test, space, gm)
                    r[f"mmd2_{gm:g}"] = v
                    r[f"floor_{gm:g}"] = R["floor"][gm]
                    r[f"ratio_{gm:g}"] = v / max(R["floor"][gm], 1e-12)
                rows.append(r)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 250)
    print("\n(A) one-step MMD^2 in multiples of the noise-floor s.d., per kernel bandwidth gamma (mean over seeds x draws)")
    for name, part in df.groupby("dataset"):
        cols = [c for c in part.columns if c.startswith("ratio_") and part[c].notna().any()]
        print(f"--- {name}"); print(part.groupby("prior")[cols].mean().round(2).to_string())
    print("\n(B) log-likelihood proxies relative to a perfect model (nats per point; 0 = as good as real data)")
    print(df.groupby(["dataset", "prior"])[["d_precision", "d_coverage"]].agg(["mean", "std"]).round(3).to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
