"""
Ramachandran densities on a 3D torus: held-out data and one-step samples from the uniform and the statistical prior.
phi runs around the large circle and psi around the tube; colour is the density on the same logarithmic scale as the
flat Ramachandran figures, with directional shading from the surface normals.

    python scripts/plot_torus3d.py --dataset Pre-Pro --out figs_new/fig_torus3d.pdf
"""
import argparse, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_rgba, LinearSegmentedColormap
import style
import plot_density as PD
from eval_synth import load_model

R, r = 2.0, 0.85                     # torus radii
BASE = np.array(to_rgba("#e6ebf1"))  # colour of regions without density
SURF = LinearSegmentedColormap.from_list("surf", ["#e6ebf1", "#b9cde6", "#6f9bd0", "#1f5fa8", "#0b2f63"])
LIGHT = np.array([-0.45, -0.55, 0.70]); LIGHT = LIGHT / np.linalg.norm(LIGHT)


def torus_xyz(u, v):
    return (R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)


def draw(ax, D, vmax, title=None, elev=38, azim=-62):
    n = D.shape[0]
    g = np.linspace(-np.pi, np.pi, n + 1)
    U, V = np.meshgrid(g, g, indexing="ij")
    X, Y, Z = torus_xyz(U, V)
    Dw = np.pad(D, ((0, 1), (0, 1)), mode="wrap")                 # vertex grid wraps around
    norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax, clip=True)
    C = SURF(norm(np.clip(Dw, vmax * 1e-3, None)))          # base colour at and below the lowest level
    Nx, Ny, Nz = np.cos(V) * np.cos(U), np.cos(V) * np.sin(U), np.sin(V)
    shade = 0.62 + 0.38 * np.clip(Nx * LIGHT[0] + Ny * LIGHT[1] + Nz * LIGHT[2], 0, 1)
    C[..., :3] = C[..., :3] * shade[..., None]
    ax.plot_surface(X, Y, Z, facecolors=C, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False,
                    rasterized=True)
    t = np.linspace(-np.pi, np.pi, 300)                            # reference circles phi = 0 and psi = 0
    for xs, ys, zs in [torus_xyz(np.zeros_like(t), t), torus_xyz(t, np.zeros_like(t))]:
        ax.plot(xs * 1.003, ys * 1.003, zs * 1.003, color="#9aa7b5", lw=0.5, alpha=0.8)
    ax.view_init(elev=elev, azim=azim)
    lim = R + r
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim * 0.45, lim * 0.45)
    try:
        ax.set_box_aspect((1, 1, 0.42), zoom=1.45)
    except TypeError:                                        # older matplotlib without zoom
        ax.set_box_aspect((1, 1, 0.42))
    ax.set_axis_off()
    if title:
        ax.set_title(title, pad=-14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--dataset", default="Pre-Pro")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--res", type=int, default=360, help="grid bins per angle (360: 1 degree)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--coupling", default="soft")
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--elev", type=float, default=38)
    ap.add_argument("--azim", type=float, default=-62)
    ap.add_argument("--data", choices=["all", "test"], default="all",
                    help="data panel from all residues of the type (smoother) or from the test split only")
    ap.add_argument("--out", default="figs_new/fig_torus3d.pdf")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    PD.MIN_EPOCHS = a.min_epochs
    runs = PD.pick_runs(a.root, a.dataset, a.epochs, a.coupling)
    assert "uniform" in runs and "gmm" in runs, f"need both priors, found {list(runs)}"
    _, _, cfg = load_model(runs["uniform"], dev, a.ckpt)
    if a.data == "all":
        from rmf.data.amino_datamodule import Top500
        panels = [Top500(amino=a.dataset).data.double().numpy()]
    else:
        panels = [PD.test_split(cfg, a.dataset).double().numpy()]
    for prior in ["uniform", "gmm"]:
        x, _ = PD.generate(runs[prior], dev, a.n, a.ckpt)
        print(f"[torus3d] {prior}: {os.path.basename(runs[prior])}")
        panels.append(np.asarray(x))
    sigma = 1.5 * a.res / 180                                   # same smoothing in degrees at any resolution
    Ds = [PD.density_torus(p, bins=a.res, sigma=sigma) for p in panels]
    vmax = max(D.max() for D in Ds)
    style.apply()
    titles = ["Data" if a.data == "all" else style.TITLE_DATA, style.TITLE_UNIFORM, style.TITLE_STAT]
    fig = plt.figure(figsize=(7.2, 2.1))
    for j, D in enumerate(Ds):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        draw(ax, D, vmax, titles[j], a.elev, a.azim)
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=-0.08)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", bbox_inches="tight", dpi=200)
    base = os.path.splitext(a.out)[0]
    for name, D in zip(["data", "uniform", "statistical"], Ds):             # single panels, e.g. for the teaser
        f1 = plt.figure(figsize=(2.6, 2.0)); ax = f1.add_subplot(111, projection="3d")
        draw(ax, D, vmax, None, a.elev, a.azim)
        f1.subplots_adjust(left=0, right=1, top=1, bottom=0)
        f1.savefig(f"{base}_{name}.pdf", bbox_inches="tight", dpi=300); plt.close(f1)
    print(f"written {a.out} and single panels {base}_{{data,uniform,statistical}}.pdf")


if __name__ == "__main__":
    main()
