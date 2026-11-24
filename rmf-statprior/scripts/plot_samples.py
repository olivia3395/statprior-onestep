"""
Qualitative figure: held-out data vs one-step samples with the uniform prior vs the statistical
prior, one row per dataset.

    sphere (Earth / synthetic): Mollweide scatter
    SO(3): images of the three body axes R e_x, R e_y, R e_z on a Mollweide map (three colours),
           as in the RMF paper
    torus: Ramachandran-style 2-D histogram of the first two angles (degrees)

Run from rmf/ on a machine with the trained runs (GPU recommended):
    python ../paper/figs/plot_samples.py --datasets fire cone fisher24 swiss Glycine
    python scripts/plot_samples.py --datasets fire cone --coupling hot --out figs/fig6_samples.pdf
For each dataset and prior it uses the run with the most epochs (latest if tied).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style (scripts/style.py)
import numpy as np
import torch

from eval_synth import load_model  # noqa: E402

SO3_SETS = {"cone", "fisher24", "line", "swiss", "peak"}
TORUS_SETS = {"General", "Glycine", "Proline", "Pre-Pro", "RNA"}


def space_of(name):
    return "so3" if name in SO3_SETS else "torus" if name in TORUS_SETS else "sphere"


def pick_runs(root, name, coupling, min_epochs):
    """Return {'uniform': run_dir, 'gmm': run_dir} for a dataset."""
    best = {}
    for f in glob.glob(os.path.join(root, "*", "test_metrics.json")):
        meta = json.load(open(f))
        if meta.get("dataset") != name or meta.get("prior_type") not in ("uniform", "gmm"):
            continue
        if meta["prior_type"] == "gmm" and coupling and meta.get("prior_coupling") != coupling:
            continue
        run = os.path.dirname(f)
        cfg_file = os.path.join(run, ".hydra", "config.yaml")
        if not os.path.exists(cfg_file):
            continue
        from omegaconf import OmegaConf
        from plot_density import MAIN_EPOCHS, is_main_setting
        cfg = OmegaConf.load(cfg_file)
        ep = int(cfg.trainer.max_epochs)
        if ep < min_epochs or not is_main_setting(cfg, meta):
            continue
        if name in MAIN_EPOCHS and ep != MAIN_EPOCHS[name]:
            continue
        key = (ep, os.path.basename(run))
        if meta["prior_type"] not in best or key > best[meta["prior_type"]][0]:
            best[meta["prior_type"]] = (key, run)
    return {k: v[1] for k, v in best.items()}


def test_split(cfg, space):
    if space == "sphere":
        from eval_real import split_tensors as st
        _, _, test = st(cfg)
    else:
        from eval_manifold import split_tensors as st
        _, _, test = st(cfg)
    return test


AXES_RGB = ["#c44e52", "#55a868", "#4c72b0"]   # images of e_x, e_y, e_z
NICE = {"fire": "Fire", "flood": "Flood", "earthquake": "Earthquake", "volcano": "Volcano", "cone": "Cone",
        "fisher24": "Fisher", "line": "Line", "swiss": "Swiss roll"}


def draw(ax, pts, space, color):
    pts = pts.detach().cpu().double().numpy()
    if space == "sphere":
        lat = np.arcsin(np.clip(pts[:, 2], -1, 1)); lon = np.arctan2(pts[:, 1], pts[:, 0])
        ax.scatter(lon, lat, s=0.3, c=color, alpha=0.5, linewidths=0, rasterized=True)
    elif space == "so3":
        R = pts.reshape(-1, 3, 3)
        for j, c in enumerate(AXES_RGB):
            v = R[:, :, j]
            lat = np.arcsin(np.clip(v[:, 2], -1, 1)); lon = np.arctan2(v[:, 1], v[:, 0])
            ax.scatter(lon, lat, s=0.2, c=c, alpha=0.35, linewidths=0, rasterized=True)
    else:
        ang = (np.degrees(pts[:, :2]) + 180) % 360 - 180
        ax.hist2d(ang[:, 0], ang[:, 1], bins=90, range=[[-180, 180], [-180, 180]],
                  cmap=style.DENSITY, norm=matplotlib.colors.LogNorm(), rasterized=True)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)


def _unit_points(pts, space):
    """Return a list of (xyz array, colour) to draw on the unit sphere."""
    pts = pts.detach().cpu().double().numpy()
    if space == "sphere":
        return [(pts.reshape(-1, 3), None)]
    R = pts.reshape(-1, 3, 3)
    return [(R[:, :, j], c) for j, c in enumerate(AXES_RGB)]


def draw3d(ax, pts, space, color, elev=20, azim=45):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#ececec",
                    alpha=0.35, linewidth=0, shade=False, rasterized=True)
    for xyz, c in _unit_points(pts, space):
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1.2, c=c or color, alpha=0.6,
                   linewidths=0, depthshade=True, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--datasets", nargs="+", default=["fire", "cone", "fisher24", "swiss"])
    ap.add_argument("--coupling", default=None, help="restrict statistical-prior runs to this coupling, e.g. soft or hot")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--out", default="fig6_samples.pdf")
    ap.add_argument("--style", choices=["map", "3d"], default="map",
                    help="map = Mollweide projection (whole sphere); 3d = sphere as in the RMF paper")
    ap.add_argument("--elev", type=float, default=20)
    ap.add_argument("--azim", type=float, default=45)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    style.apply()

    rows = []
    for name in a.datasets:
        runs = pick_runs(a.root, name, a.coupling, a.min_epochs)
        if runs:
            rows.append((name, runs))
        else:
            print(f"[skip] no runs for {name}")
    grid = []  # one entry per dataset: (name, space, [(title, pts, colour), ...])
    for name, runs in rows:
        space = space_of(name)
        panels, cfg_any = {}, None
        for prior in ["uniform", "gmm"]:
            if prior not in runs:
                panels[prior] = None; continue
            model, ckpt, cfg = load_model(runs[prior], device, "last")
            cfg_any = cfg
            with torch.no_grad():
                out = torch.cat([model.sample_batch(steps=1, sz=min(4096, a.n - j)) for j in range(0, a.n, 4096)])
            panels[prior] = out
            print(f"[{name}] {prior}: {os.path.basename(runs[prior])} ({ckpt})")
            del model
        test = test_split(cfg_any, space)
        test = test[torch.randperm(len(test))[: a.n]]
        grid.append((NICE.get(name, name), space, [(style.TITLE_DATA, test, style.DATA),
                                   (style.TITLE_UNIFORM, panels["uniform"], style.UNIFORM),
                                   (style.TITLE_STAT, panels["gmm"], style.STAT)]))

    if a.style == "3d":
        # layout of the RMF paper: rows = methods, columns = datasets
        nr, nc = 3, len(grid)
        fig = plt.figure(figsize=(1.9 * nc + 1.0, 1.9 * nr))
        for j, (name, space, cols) in enumerate(grid):
            for i, (title, pts, color) in enumerate(cols):
                if space == "torus":
                    ax = fig.add_subplot(nr, nc, i * nc + j + 1)
                    draw(ax, pts, space, color) if pts is not None else None
                else:
                    ax = fig.add_subplot(nr, nc, i * nc + j + 1, projection="3d")
                    if pts is not None:
                        draw3d(ax, pts, space, color, a.elev, a.azim)
                    else:
                        ax.set_axis_off()
                if pts is None:
                    ax.text2D(0.5, 0.5, "missing", ha="center", transform=ax.transAxes) if space != "torus" \
                        else ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
                if i == 0:
                    ax.set_title(name)
                if j == 0:
                    (ax.text2D if space != "torus" else ax.text)(
                        -0.12, 0.5, title.replace(" (", "\n("), rotation=90, va="center", ha="right",
                        transform=ax.transAxes)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.01, wspace=0.0, hspace=0.02)
        fig.savefig(a.out, dpi=300, bbox_inches="tight")
        fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=160, bbox_inches="tight")
        print("written", a.out)
        return

    fig = plt.figure(figsize=(7.0, 1.25 * len(grid) + 0.3))
    for i, (name, space, cols) in enumerate(grid):
        for j, (title, pts, color) in enumerate(cols):
            proj = None if space == "torus" else "mollweide"
            ax = fig.add_subplot(len(grid), 3, 3 * i + j + 1, projection=proj)
            if pts is None:
                ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
            else:
                draw(ax, pts, space, color)
            if proj:
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.grid(True, lw=0.3, alpha=0.5)
            else:
                ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\psi$") if j == 0 else None
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.text(-0.30 if space == "torus" else -0.08, 0.5, name, rotation=90, va="center", ha="right",
                        transform=ax.transAxes)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.02, wspace=0.04, hspace=0.06)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=160, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
