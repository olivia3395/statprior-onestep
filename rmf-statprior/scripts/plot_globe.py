"""
Earth data and one-step samples on a shaded globe (orthographic projection): held-out data, the uniform prior and the
statistical prior, one row per dataset with a view centred on the region of interest.

    python scripts/plot_globe.py --datasets fire earthquake --out figs_new/fig_globe.pdf
"""
import argparse, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style
import plot_density as PD
from eval_synth import load_model

VIEWS = {"fire": (15, 5), "earthquake": (150, 5), "volcano": (140, 10), "flood": (80, 15)}   # (lon0, lat0)
NICE = {"fire": "Fire", "earthquake": "Earthquake", "volcano": "Volcano", "flood": "Flood"}


def ortho(lon, lat, lon0, lat0):
    lon, lat, lon0, lat0 = map(np.radians, (lon, lat, lon0, lat0))
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    vis = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0) > 0
    return x, y, vis


def to_limb(x, y, vis):
    """Points on the far side are moved to the limb, which clips polygons to the visible disc."""
    r = np.sqrt(x ** 2 + y ** 2) + 1e-12
    return np.where(vis, x, x / r), np.where(vis, y, y / r)


def draw_globe(ax, world, lon, lat, lon0, lat0, color, title=None):
    g = np.linspace(-1, 1, 500); X, Y = np.meshgrid(g, g)
    shade = np.clip(1 - 0.33 * ((X + 0.35) ** 2 + (Y - 0.4) ** 2), 0.55, 1.0)
    ax.imshow(np.where(X ** 2 + Y ** 2 <= 1, shade, np.nan), extent=[-1, 1, -1, 1], origin="lower",
              cmap=LinearSegmentedColormap.from_list("s", ["#e6ecf3", "#ffffff"]), vmin=0.55, vmax=1.0, zorder=0)
    for geom in ([] if world is None else world.geometry):
        for p in getattr(geom, "geoms", [geom]):
            lo, la = np.asarray(p.exterior.coords).T
            x, y, v = ortho(lo, la, lon0, lat0)
            if not v.any():
                continue
            x, y = to_limb(x, y, v)
            ax.fill(x, y, color="#d9d9d9", lw=0, zorder=1)
    for L in range(-180, 180, 30):
        la = np.linspace(-90, 90, 300); x, y, v = ortho(np.full_like(la, L), la, lon0, lat0)
        ax.plot(np.where(v, x, np.nan), np.where(v, y, np.nan), color="white", lw=0.35, zorder=1.5)
    for La in range(-60, 90, 30):
        lo = np.linspace(-180, 180, 600); x, y, v = ortho(lo, np.full_like(lo, La), lon0, lat0)
        ax.plot(np.where(v, x, np.nan), np.where(v, y, np.nan), color="white", lw=0.35, zorder=1.5)
    x, y, v = ortho(lon, lat, lon0, lat0)
    ax.scatter(x[v], y[v], s=7, color=color, alpha=0.05, lw=0, zorder=2)      # soft glow
    ax.scatter(x[v], y[v], s=0.7, color=color, alpha=0.55, lw=0, zorder=3)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#9aa7b5", lw=0.8, zorder=4))
    ax.set_xlim(-1.04, 1.04); ax.set_ylim(-1.04, 1.04); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--datasets", nargs="+", default=["fire", "earthquake"])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--coupling", default="soft")
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--out", default="figs_new/fig_globe.pdf")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    PD.MIN_EPOCHS = a.min_epochs
    world = PD.load_world()
    style.apply()
    fig, axes = plt.subplots(len(a.datasets), 3, figsize=(6.4, 2.25 * len(a.datasets)), squeeze=False)
    titles = [style.TITLE_DATA, style.TITLE_UNIFORM, style.TITLE_STAT]
    for i, name in enumerate(a.datasets):
        runs = PD.pick_runs(a.root, name, a.epochs, a.coupling)
        assert "uniform" in runs and "gmm" in runs, f"{name}: need both priors, found {list(runs)}"
        _, _, cfg = load_model(runs["uniform"], dev, a.ckpt)
        test = PD.test_split(cfg, name).double().numpy()
        panels = [(test, style.DATA)]
        for prior, col in [("uniform", style.UNIFORM), ("gmm", style.STAT)]:
            x, _ = PD.generate(runs[prior], dev, a.n, a.ckpt)
            panels.append((x, col))
        lon0, lat0 = VIEWS.get(name, (0, 10))
        for j, (pts, col) in enumerate(panels):
            lon, lat = PD.lonlat(pts)
            draw_globe(axes[i, j], world, lon, lat, lon0, lat0, col, titles[j] if i == 0 else None)
        axes[i, 0].text(-0.08, 0.5, NICE.get(name, name), rotation=90, va="center", ha="center",
                        transform=axes[i, 0].transAxes)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.02, wspace=0.03, hspace=0.06)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=170, bbox_inches="tight")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
