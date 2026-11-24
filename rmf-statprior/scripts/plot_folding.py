"""
How the learned one-step map deforms space (Proposition 1 and Table tab:jacobian, visualised).

A regular grid on the torus is pushed through the trained one-step map T(z) = Exp_z(-u(z, 0, 1)) of the uniform-prior
and statistical-prior models. Left of each pair: the grid in the space of the prior, coloured by the local volume change
log|det J_T| (blue: contraction, red: folding, i.e. det J_T < 0). Right: the image of the grid lines, drawn over the
density of held-out data; segments where the map folds are red. For the statistical prior, the opacity of every grid
point follows the prior density, so that regions the prior never visits (where the map is not trained) fade out.

    python scripts/plot_folding.py --dataset Pre-Pro --out figs_new/fig_folding.pdf
"""
import argparse, math, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import style
import plot_density as PD
from eval_synth import load_model

TWO_PI = 2 * math.pi
BAD = "#c44e52"


def one_step(model, z):
    r = torch.zeros(z.shape[0], 1, device=z.device); t = torch.ones_like(r)
    with torch.no_grad():
        return model.manifold.project(model.manifold.exp(z, -model.forward(z, r, t)))


def wrap(d):
    return (d + math.pi) % TWO_PI - math.pi


def jacobian_det(model, z, h=1e-3):
    cols = []
    for i in range(2):
        e = torch.zeros_like(z); e[:, i] = h
        cols.append(wrap(one_step(model, z + e) - one_step(model, z - e)) / (2 * h))
    J = torch.stack(cols, dim=-1)
    return (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]).cpu().numpy()


def prior_weight(model, z, prior):
    if prior == "uniform":
        return np.ones(len(z))
    with torch.no_grad():
        lp = model.log_prob_prior(z).cpu().numpy()
    p = np.exp(lp - lp.max())
    return np.clip(p / np.quantile(p, 0.98), 0.0, 1.0)


def deg(a):
    return (np.degrees(np.asarray(a)) + 180) % 360 - 180


def segments(xy):
    """Split a polyline on the torus (degrees) where it wraps around."""
    seg = np.stack([xy[:-1], xy[1:]], axis=1)
    jump = np.abs(seg[:, 1] - seg[:, 0]).max(-1) > 90
    return seg, ~jump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--dataset", default="Pre-Pro")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--lines", type=int, default=28)
    ap.add_argument("--res", type=int, default=600)
    ap.add_argument("--out", default="figs_new/fig_folding.pdf")
    ap.add_argument("--epochs", type=int, default=None, help="training budget (default: that of the main tables)")
    ap.add_argument("--coupling", default="soft")
    ap.add_argument("--min_epochs", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    PD.MIN_EPOCHS = a.min_epochs
    runs = PD.pick_runs(a.root, a.dataset, a.epochs, a.coupling)
    assert "uniform" in runs and "gmm" in runs, f"need both priors, found {list(runs)}"
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.1))
    cmap = LinearSegmentedColormap.from_list("vol", ["#1f5fa8", "#9dbfe2", "#f4f4f4", "#f2b8b9", BAD])
    for r, prior in enumerate(["uniform", "gmm"]):
        model, ck, cfg = load_model(runs[prior], dev, a.ckpt)
        print(f"[folding] {prior}: {os.path.basename(runs[prior])} ({ck})")
        test = PD.test_split(cfg, a.dataset).double().numpy()[:, :2]
        Dt = PD.density_torus(test)
        # ---------------- left: grid in prior space, coloured by log|det J| (folds in red)
        g = np.linspace(0, TWO_PI, 181)[:-1] + TWO_PI / 360
        G = torch.tensor(np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2), dtype=torch.float32, device=dev)
        det = jacobian_det(model, G)
        w = prior_weight(model, G, prior).reshape(180, 180)
        val = np.where(det < 0, 4.0, -np.clip(-np.log(np.abs(det) + 1e-12), -4, 4))  # contraction negative, folds +4
        img = cmap(TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)(val.reshape(180, 180)))
        img[..., 3] = 0.15 + 0.85 * w
        ax = axes[r, 0]
        img = np.roll(np.roll(img, 90, axis=0), 90, axis=1)   # [0, 360) -> [-180, 180)
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=[-180, 180, -180, 180], interpolation="nearest")
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        fold = float((det < 0).mean()) if prior == "uniform" else float(((det < 0) * w.ravel()).sum() / w.sum())
        ax.set_title(f"{style.TITLE_UNIFORM if prior == 'uniform' else style.TITLE_STAT}\n"
                     f"prior space: folded {100 * fold:.0f}%", fontsize=8)
        # ---------------- right: image of grid lines over the data density
        ax = axes[r, 1]
        PD.draw_torus(ax, Dt, Dt.max())
        tt = np.linspace(0, TWO_PI, a.res)
        for k in np.linspace(0, TWO_PI, a.lines, endpoint=False):
            for line in (np.stack([np.full_like(tt, k), tt], 1), np.stack([tt, np.full_like(tt, k)], 1)):
                z = torch.tensor(line, dtype=torch.float32, device=dev)
                img_pts = deg(one_step(model, z).cpu().numpy())
                dl = jacobian_det(model, z); wl = prior_weight(model, z, prior)
                seg, keep = segments(img_pts)
                folded = (dl[:-1] < 0) & (dl[1:] < 0)
                alpha = 0.08 + 0.7 * np.minimum(wl[:-1], wl[1:])
                cols = np.array([matplotlib.colors.to_rgba(BAD if f else style.DATA) for f in folded])
                cols[:, 3] = alpha
                ax.add_collection(LineCollection(seg[keep], colors=cols[keep], linewidths=0.45))
        ax.set_title("image of the grid under one step", fontsize=8)
        ax.set_yticklabels([])
        for ax in axes[r]:
            ax.set_aspect("equal"); style.frame(ax)
    for ax in axes[:, 0]:
        ax.set_xticks([-180, 0, 180]); ax.set_yticks([-180, 0, 180]); ax.tick_params(labelsize=6)
        ax.set_ylabel(r"$\psi$ (prior space)", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel(r"$\phi$", fontsize=7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(color="#1f5fa8", label="strong contraction"), Patch(color="#f4f4f4", label="volume preserved"),
          Patch(color=BAD, label="folded ($\\det J_T<0$)"), Line2D([], [], color=style.DATA, lw=0.8, label="grid image"),
          Line2D([], [], color=BAD, lw=0.8, label="folded segment")]
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=6.5, handlelength=1.4)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.08, wspace=0.08, hspace=0.25)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight"); fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=170, bbox_inches="tight")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
