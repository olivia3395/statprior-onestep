"""
Displacement figure: where one network evaluation moves samples of the prior, for trained models on a
Ramachandran dataset. Each segment joins a prior sample z to its one-step image T(z) along the shortest path on
the torus; the background is the density of held-out data.

Run from rmf/ (GPU or CPU):
    python scripts/plot_displacement.py --dataset Pre-Pro --epochs 1000 --out fig_displacement.pdf
"""
import argparse
import math
import os
import sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT)
sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style (scripts/style.py)
import numpy as np
import torch
from matplotlib.collections import LineCollection

import plot_density as PD
from eval_synth import load_model

C_UNI, C_STAT = style.UNIFORM, style.STAT


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pairs(run, device, n, ckpt, seed):
    """Prior samples z and their one-step images T(z), computed exactly as in MeanFlowModule.sample_batch."""
    model, _, cfg = load_model(run, device, ckpt)
    torch.manual_seed(seed)
    with torch.no_grad():
        z = model._prior_samples((n,) + tuple(cfg.model.in_shape), device)
        shape = (n,) + (1,) * (z.dim() - 1)
        t = torch.ones(shape, device=device); r = torch.zeros(shape, device=device)
        x = model.manifold.project(model.manifold.exp(z, -(t - r) * model.forward(z, r, t)))
    return z.cpu().double().numpy()[:, :2], x.cpu().double().numpy()[:, :2], cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--dataset", default="Pre-Pro")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--coupling", default="soft")
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--out", default="fig_displacement.pdf")
    a = ap.parse_args()
    PD.MIN_EPOCHS = a.min_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    style.apply()

    runs = PD.pick_runs(a.root, a.dataset, a.epochs, a.coupling)
    if "uniform" not in runs or "gmm" not in runs:
        raise SystemExit(f"need both priors for {a.dataset}, found {sorted(runs)}")
    zu, xu, cfg = pairs(runs["uniform"], device, a.n, a.ckpt, 0)
    zs, xs, _ = pairs(runs["gmm"], device, a.n, a.ckpt, 0)
    test = PD.test_split(cfg, a.dataset).double().numpy()[:, :2]
    Dt = PD.density_torus(test)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
    for ax, (z, x, c, title) in zip(axes, [(zu, xu, C_UNI, style.TITLE_UNIFORM),
                                          (zs, xs, C_STAT, style.TITLE_STAT)]):
        PD.draw_torus(ax, Dt, Dt.max())
        for im in ax.get_images():
            im.set_cmap(style.DENSITY); im.set_alpha(0.5)
        d = wrap(x - z)                                   # shortest displacement on the torus
        zd, ed = np.degrees(wrap(z)), np.degrees(wrap(z) + d)
        segs = np.stack([zd, ed], 1)
        ax.add_collection(LineCollection(segs, colors=c, linewidths=0.6, alpha=0.7))
        ax.scatter(zd[:, 0], zd[:, 1], s=3, color=c, alpha=0.8, lw=0)
        ax.scatter(ed[:, 0], ed[:, 1], s=4, color=style.DATA, lw=0)
        dist = np.linalg.norm(d, axis=1).mean()
        ax.set_title(f"{title}\nmean displacement {dist:.2f} rad", color="#6b6b6b" if c == C_UNI else c)
        ax.set_xlabel(r"$\phi$", fontsize=7); ax.set_xticks([-180, 0, 180]); ax.set_yticks([-180, 0, 180])
    axes[0].set_ylabel(r"$\psi$", fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=160, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
