"""
Density maps in the style of the RMF paper (their Figs. 3 and 8):
  * Earth datasets: density of one-step samples on a world map, with held-out data overlaid;
  * torus datasets: Ramachandran-style density of (phi, psi).
Columns: held-out data | RMF (uniform prior) | RMF + statistical prior (ours).

Run from rmf/ (GPU recommended; CPU works with smaller --n):
    python scripts/plot_density.py --datasets volcano earthquake flood fire --out fig_earth_density.pdf
    python scripts/plot_density.py --datasets Glycine Proline Pre-Pro General --out fig_rama.pdf
Options: --epochs 1000 restricts to runs with that many epochs (equal training budget);
         --ckpt best uses the best-validation checkpoint (recommended for RNA, see paper).
The world map is downloaded once from Natural Earth (as in the RMF code base) and cached in
data/ne_110m_countries.zip; without internet the Earth panels are drawn without coastlines.
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
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

from eval_synth import load_model  # noqa: E402

EARTH = {"volcano", "earthquake", "flood", "fire"}
TORUS = {"General", "Glycine", "Proline", "Pre-Pro", "RNA"}
WORLD_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
MIN_EPOCHS = 20
WORLD_CACHE = os.path.join("data", "ne_110m_countries.zip")


# ------------------------------------------------------------------ runs
def is_main_setting(cfg, meta):
    """Runs of the main tables only: MeanFlow objective and, for the statistical prior, s = 2 and at most 32 components."""
    if str(cfg.model.get("my_method", "mf")) != "mf":
        return False
    if meta.get("prior_type") == "gmm":
        if float(cfg.model.get("prior_cov_scale", 2.0)) != 2.0 or int(cfg.model.get("prior_k", 32)) > 32:
            return False
        if str(cfg.model.get("prior_select", "bic")) != "bic":     # exclude the sensitivity runs with fixed K
            return False
    return True


# training budget of the main tables
MAIN_EPOCHS = {"fire": 100, "flood": 100, "earthquake": 100, "volcano": 100, "cone": 200, "fisher24": 200,
               "line": 200, "swiss": 200, "General": 1000, "Glycine": 1000, "Proline": 1000, "Pre-Pro": 1000,
               "RNA": 1000}


def pick_runs(root, name, epochs, coupling):
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
        cfg = OmegaConf.load(cfg_file)
        ep = int(cfg.trainer.max_epochs)
        if (epochs and ep != epochs) or ep < MIN_EPOCHS:
            continue
        if not epochs and name in MAIN_EPOCHS and ep != MAIN_EPOCHS[name]:
            continue
        if not is_main_setting(cfg, meta):
            continue
        key = (ep, os.path.basename(run))
        if meta["prior_type"] not in best or key > best[meta["prior_type"]][0]:
            best[meta["prior_type"]] = (key, run)
    return {k: v[1] for k, v in best.items()}


def test_split(cfg, name):
    if name in EARTH:
        from eval_real import split_tensors
    else:
        from eval_manifold import split_tensors
    return split_tensors(cfg)[2]


def generate(run, device, n, ckpt):
    model, ck, cfg = load_model(run, device, ckpt)
    with torch.no_grad():
        out = torch.cat([model.sample_batch(steps=1, sz=min(4096, n - j)) for j in range(0, n, 4096)])
    print(f"   {os.path.basename(run)} ({ck})")
    return out.detach().cpu().double().numpy(), cfg


# ------------------------------------------------------------------ drawing
def lonlat(x):
    return np.degrees(np.arctan2(x[:, 1], x[:, 0])), np.degrees(np.arcsin(np.clip(x[:, 2], -1, 1)))


def density_lonlat(x, bins=(360, 180), sigma=1.5):
    lon, lat = lonlat(x)
    H, _, _ = np.histogram2d(lon, lat, bins=bins, range=[[-180, 180], [-90, 90]])
    # divide by cell area (proportional to cos(lat)) so the map shows density on the sphere
    lat_c = np.linspace(-90, 90, bins[1] + 1)[:-1] + 90 / bins[1]
    H = H / np.clip(np.cos(np.radians(lat_c)), 1e-3, None)[None, :]
    H = gaussian_filter(H, sigma=sigma, mode=("wrap", "nearest"))
    return H / H.sum()


def density_torus(x, bins=180, sigma=1.5):
    ang = (np.degrees(x[:, :2]) + 180) % 360 - 180
    H, _, _ = np.histogram2d(ang[:, 0], ang[:, 1], bins=bins, range=[[-180, 180], [-180, 180]])
    H = gaussian_filter(H, sigma=sigma, mode="wrap")
    return H / H.sum()


def load_world():
    try:
        import geopandas
        if not os.path.exists(WORLD_CACHE):
            import urllib.request
            os.makedirs(os.path.dirname(WORLD_CACHE), exist_ok=True)
            urllib.request.urlretrieve(WORLD_URL, WORLD_CACHE)
        return geopandas.read_file(WORLD_CACHE)
    except Exception as e:  # no internet / no geopandas: plot without coastlines
        print("[warn] world map unavailable:", e)
        return None


NICE = {"fire": "Fire", "flood": "Flood", "earthquake": "Earthquake", "volcano": "Volcano", "cone": "Cone",
        "fisher24": "Fisher", "line": "Line", "swiss": "Swiss roll"}


def _cmap():
    """Shared density colour map; values below the lower limit are transparent (no outlines at the support edge)."""
    cm = style.DENSITY.copy()
    cm.set_under((1, 1, 1, 0))
    return cm


def draw_earth(ax, D, test, world, vmax, show_test):
    if world is not None:
        world.plot(ax=ax, color=style.LAND, edgecolor="white", linewidth=0.2)
    if D is not None:
        ax.imshow(np.maximum(D.T, vmax * 1e-6), origin="lower", extent=[-180, 180, -90, 90], cmap=_cmap(),
                  norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), alpha=0.9, interpolation="bilinear", zorder=2)
    if show_test:
        lon, lat = lonlat(test)
        ax.scatter(lon, lat, s=0.6, c=style.DATA, linewidths=0, zorder=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal"); style.frame(ax)


def draw_torus(ax, D, vmax):
    ax.imshow(np.maximum(D.T, vmax * 1e-6), origin="lower", extent=[-180, 180, -180, 180], cmap=_cmap(),
              norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), interpolation="bilinear")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks([-180, 0, 180]); ax.set_yticks([-180, 0, 180]); ax.tick_params(labelsize=6)
    ax.set_aspect("equal"); style.frame(ax)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--coupling", default="soft", help="coupling of the statistical-prior run; '' = any")
    ap.add_argument("--ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--out", default="fig_density.pdf")
    ap.add_argument("--min_epochs", type=int, default=20, help="skip smoke-test runs")
    a = ap.parse_args()
    global MIN_EPOCHS
    MIN_EPOCHS = a.min_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    style.apply()
    world = load_world() if any(d in EARTH for d in a.datasets) else None

    rows = []
    for name in a.datasets:
        runs = pick_runs(a.root, name, a.epochs, a.coupling)
        if "uniform" not in runs or "gmm" not in runs:
            print(f"[skip] {name}: need both priors, found {sorted(runs)}"); continue
        print(f"[{name}]")
        gen_u, cfg = generate(runs["uniform"], device, a.n, a.ckpt)
        gen_s, _ = generate(runs["gmm"], device, a.n, a.ckpt)
        test = test_split(cfg, name).double().numpy()
        rows.append((name, test, gen_u, gen_s))

    if not rows:
        raise SystemExit("no dataset with both a uniform-prior and a statistical-prior run")
    earth = [r for r in rows if r[0] in EARTH]
    ncol = 3
    if earth:
        fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 2.35, 1.35 * len(rows) + 0.4), squeeze=False)
        titles = [style.TITLE_DATA, style.TITLE_UNIFORM, style.TITLE_STAT]
    else:
        fig, axes = plt.subplots(len(rows), ncol, figsize=(ncol * 2.3, 2.15 * len(rows) + 0.5), squeeze=False)
        titles = [style.TITLE_DATA, style.TITLE_UNIFORM, style.TITLE_STAT]
    for i, (name, test, gu, gs) in enumerate(rows):
        if name in EARTH:
            Dt, Du, Ds = density_lonlat(test, sigma=2.5), density_lonlat(gu), density_lonlat(gs)
            vmax = max(Du.max(), Ds.max())
            draw_earth(axes[i, 0], None, test, world, vmax, True)
            draw_earth(axes[i, 1], Du, test, world, vmax, False)
            draw_earth(axes[i, 2], Ds, test, world, vmax, False)
        else:
            Dt, Du, Ds = density_torus(test), density_torus(gu), density_torus(gs)
            vmax = max(Dt.max(), Du.max(), Ds.max())
            for j, D in enumerate([Dt, Du, Ds]):
                draw_torus(axes[i, j], D, vmax)
            axes[i, 0].set_ylabel(r"$\psi$", fontsize=7)
            for j in range(3):
                axes[i, j].set_xlabel(r"$\phi$", fontsize=7, labelpad=1)
        axes[i, 0].text(-0.08 if name in EARTH else -0.35, 0.5, NICE.get(name, name), rotation=90, va="center", ha="right",
                        transform=axes[i, 0].transAxes, fontsize=9)
    for j in range(ncol):
        axes[0, j].set_title(titles[j])
    fig.tight_layout()
    if not earth:
        fig.subplots_adjust(wspace=0.45, hspace=0.35)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=160, bbox_inches="tight")
    print("written", a.out)


if __name__ == "__main__":
    main()
