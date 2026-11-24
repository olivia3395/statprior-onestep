"""
Backbone fragments of k consecutive residues generated in torsion space (pepwin_k10: T^20, pepwin_k20: T^40),
rebuilt in 3D with ideal geometry (scripts/backbone.py) and compared with held-out fragments.

Rows: one-step samples of each trained model, samples of its prior alone, and three references computed from
the training split: a kernel density estimate (bandwidth by validation likelihood), a residue-shuffled control
(each residue's (phi, psi) drawn from the training marginal of that position, independently of its neighbours),
and the training fragments themselves (what a perfect model would score).

    python scripts/eval_backbone.py --datasets pepwin_k10 pepwin_k20 --reps 3 --csv backbone_eval.csv
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

import backbone as BB
import eval_manifold as EM
import kde_baseline as KB
from eval_synth import load_model


def one_step(model, cfg, n, device):
    with torch.no_grad():
        out = torch.cat([model.sample_batch(steps=1, sz=min(4096, n - j)) for j in range(0, n, 4096)])
    return out.detach().cpu().double().numpy() % (2 * math.pi)


def prior_samples(model, cfg, n, device):
    with torch.no_grad():
        z = model._prior_samples((n,) + tuple(cfg.model.in_shape), device)
    return z.detach().cpu().double().numpy() % (2 * math.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--datasets", nargs="+", default=["pepwin_k10", "pepwin_k20"])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--csv", default="backbone_eval.csv")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    rows, refs_done = [], set()

    for f in sorted(glob.glob(os.path.join(a.root, "*", "test_metrics.json"))):
        meta = json.load(open(f))
        name = meta.get("dataset", "")
        if name not in a.datasets:
            continue
        run = os.path.dirname(f)
        model, ckpt, cfg = load_model(run, device, a.ckpt)
        if model is None or cfg.trainer.max_epochs < a.min_epochs:
            continue
        prior = "uniform" if meta["prior_type"] == "uniform" else \
            f"{meta['prior_type']}(K={meta.get('prior_k_selected')},s={meta.get('prior_cov_scale')},{meta.get('prior_coupling')})"
        tr, va, te = [t.double().numpy() for t in EM.split_tensors(cfg)]
        print(f"[eval] {os.path.basename(run)} {name} {prior} seed={cfg.seed} ckpt={ckpt} (train {len(tr)}, test {len(te)})",
              flush=True)

        if name not in refs_done:                         # references, computed once per dataset
            refs_done.add(name)
            rows.append(dict(dataset=name, method="training data (perfect model)", rep=0, **BB.compare(tr[: a.n], te)))
            for r in range(a.reps):
                sh = tr[rng.integers(len(tr), size=(a.n,))].reshape(a.n, -1, 2)
                for j in range(sh.shape[1]):              # independent residues, correct per-position marginals
                    sh[:, j] = tr[rng.integers(len(tr), size=a.n)].reshape(a.n, -1, 2)[:, j]
                rows.append(dict(dataset=name, method="residue-shuffled control", rep=r, **BB.compare(sh.reshape(a.n, -1), te)))
            d = tr.shape[1]
            trt, vat = torch.tensor(tr), torch.tensor(va[:1000])
            centers = trt[torch.randperm(len(trt))[:3000]]
            grid = np.exp(np.linspace(np.log(0.02), np.log(1.0), 20))
            h = float(grid[int(np.argmax([KB.kde_loglik(vat, centers, g, "torus", d) for g in grid]))])
            gen = torch.Generator().manual_seed(0)
            for r in range(a.reps):
                x = KB.kde_sample(trt, a.n, h, "torus", gen).numpy()
                rows.append(dict(dataset=name, method=f"KDE (h={h:.3f})", rep=r, **BB.compare(x, te)))
            print(f"   references done (KDE bandwidth {h:.3f} rad)", flush=True)

        for r in range(a.reps):
            rows.append(dict(dataset=name, method=f"RMF 1-step, {prior}", seed=cfg.seed, rep=r,
                             **BB.compare(one_step(model, cfg, a.n, device), te)))
            rows.append(dict(dataset=name, method=f"prior alone, {prior}", seed=cfg.seed, rep=r,
                             **BB.compare(prior_samples(model, cfg, a.n, device), te)))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no pepwin runs found")
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 220)
    cols = ["clash", "helix_run", "helix", "sheet", "W1_e2e"]
    g = df.groupby(["dataset", "method"])[cols]
    print("\nGenerated fragments vs held-out fragments (rebuilt in 3D with ideal geometry). Reference values of held-out")
    print("fragments: clash_ref, helix_run_ref, ... in the CSV. Lower clash and W1_e2e are better; helix_run, helix and")
    print("sheet should match the held-out values.")
    print(g.mean().round(3).to_string())
    print("\nheld-out reference:", df.groupby("dataset")[["clash_ref", "helix_run_ref", "helix_ref", "sheet_ref"]].first().round(3).to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
