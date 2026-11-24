"""
Figure 1 (teaser): what one-step generation does with the uniform prior vs our statistical prior,
on one real dataset.

    data (held-out)  |  start: uniform noise        --1 network step-->  RMF result
                     |  start: statistical prior    --1 network step-->  our result
                        (fitted to the data by EM + BIC)

Run from rmf/ (GPU recommended):
    python scripts/plot_teaser.py --dataset Pre-Pro --epochs 1000 --out fig1_teaser.pdf
    python scripts/plot_teaser.py --dataset fire --epochs 100 --out fig1_teaser_fire.pdf
"""
import argparse
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
from matplotlib.patches import FancyArrowPatch

import plot_density as PD
from eval_synth import load_model

C_UNI, C_STAT = "#6b6b6b", style.STAT   # text-safe grey, statistical blue


def prior_and_step(run, device, n, ckpt):
    model, _, cfg = load_model(run, device, ckpt)
    with torch.no_grad():
        shape = (n,) + tuple(cfg.model.in_shape)
        z = model._prior_samples(shape, device)
        out = torch.cat([model.sample_batch(steps=1, sz=min(4096, n - j)) for j in range(0, n, 4096)])
    to_np = lambda t: t.detach().cpu().double().numpy().reshape(n, -1)
    return to_np(z), to_np(out), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--dataset", default="Pre-Pro")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--coupling", default="soft")
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--out", default="fig1_teaser.pdf")
    a = ap.parse_args()
    PD.MIN_EPOCHS = a.min_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    style.apply()

    name = a.dataset
    runs = PD.pick_runs(a.root, name, a.epochs, a.coupling)
    if "uniform" not in runs or "gmm" not in runs:
        raise SystemExit(f"need both priors for {name}, found {sorted(runs)}")
    zu, xu, cfg = prior_and_step(runs["uniform"], device, a.n, a.ckpt)
    zs, xs, _ = prior_and_step(runs["gmm"], device, a.n, a.ckpt)
    test = PD.test_split(cfg, name).double().numpy().reshape(-1, xu.shape[1])
    earth = name in PD.EARTH

    if earth:
        dens = lambda x: PD.density_lonlat(x)
        world = PD.load_world()
        def draw(ax, D, vmax, pts=None):
            PD.draw_earth(ax, D, pts if pts is not None else test, world, vmax, pts is not None)
    else:
        dens = lambda x: PD.density_torus(x)
        def draw(ax, D, vmax, pts=None):
            PD.draw_torus(ax, D, vmax)
            ax.set_xticks([]); ax.set_yticks([])

    Dt, Dzu, Dxu, Dzs, Dxs = [dens(x) for x in (test, zu, xu, zs, xs)]
    vmax = max(D.max() for D in (Dt, Dxu, Dxs, Dzs))

    w, h = (7.6, 3.3) if earth else (7.0, 3.6)
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.15, 0.38, 1, 0.55, 1, 0.42], wspace=0.05, hspace=0.35)
    ax_d = fig.add_subplot(gs[:, 0])
    ax_zu, ax_xu = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])
    ax_zs, ax_xs = fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 4])

    if earth:
        draw(ax_d, None, vmax, pts=test)
    else:
        draw(ax_d, Dt, vmax)
    draw(ax_zu, Dzu, vmax); draw(ax_xu, Dxu, vmax)
    draw(ax_zs, Dzs, vmax); draw(ax_xs, Dxs, vmax)

    ax_d.set_title(f"Data ({PD.NICE.get(name, name)})")
    ax_zu.set_title("Start: uniform prior", color=C_UNI)
    ax_xu.set_title("One-step samples", color=C_UNI)
    ax_zs.set_title("Start: statistical prior", color=C_STAT)
    ax_xs.set_title("One-step samples", color=C_STAT)
    for ax, c in [(ax_zu, C_UNI), (ax_xu, C_UNI), (ax_zs, C_STAT), (ax_xs, C_STAT)]:
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(0.9)

    def arrow(p0, p1, text, color, text_xy, ha="center"):
        fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12,
                                           lw=1.4, color=color))
        fig.text(*text_xy, text, ha=ha, va="center", fontsize=7, color=color)

    def mid_right(ax, dy=0.5):
        b = ax.get_position(); return (b.x1 + 0.006, b.y0 + dy * b.height)

    def mid_left(ax, dy=0.5):
        b = ax.get_position(); return (b.x0 - 0.006, b.y0 + dy * b.height)

    for (za, xa, c) in [(ax_zu, ax_xu, C_UNI), (ax_zs, ax_xs, C_STAT)]:
        p0, p1 = mid_right(za), mid_left(xa)
        arrow(p0, p1, "1 network\nevaluation", c, ((p0[0] + p1[0]) / 2, p0[1] + 0.07))
    bd, bz = ax_d.get_position(), ax_zs.get_position()
    y = max(bd.y0 + 0.02, bz.y0 + 0.5 * bz.height)          # horizontal arrow at the height of the prior panel
    p0, p1 = (bd.x1 + 0.006, y), (bz.x0 - 0.006, y)
    arrow(p0, p1, "fit by\nEM + BIC", C_STAT, ((p0[0] + p1[0]) / 2, y + 0.07))

    bu, bs = ax_xu.get_position(), ax_xs.get_position()
    fig.text(bu.x1 + 0.015, bu.y0 + bu.height / 2, "uniform\nprior", ha="left", va="center",
             fontsize=8, color=C_UNI, weight="bold")
    fig.text(bs.x1 + 0.015, bs.y0 + bs.height / 2, "statistical\nprior (ours)", ha="left", va="center", fontsize=8,
             color=C_STAT, weight="bold")
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=160, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
