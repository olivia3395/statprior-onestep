"""Bandwidth sensitivity: one-step MMD^2 in noise-floor s.d. vs kernel bandwidth gamma (data: gamma_ll.csv)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.unicode_minus": False})
# mean one-step MMD^2 in noise-floor s.d. per gamma (v2_eval_gamma_ll.log, part A)
G = {
 "cone": ([0.3, 1, 3, 5, 10, 20, 50, 100],
          {"uniform": [8.90, 14.41, 30.08, 45.40, 80.97, 118.77, 113.10, 78.36],
           "independent": [-0.13, -0.17, -0.27, -0.26, 0.07, 0.66, 1.15, 1.23],
           "soft": [-0.17, -0.13, -0.06, -0.01, 0.13, 0.34, 0.60, 0.64]}),
 "Pre-Pro": ([0.1, 0.3, 1, 3, 10, 30],
          {"uniform": [57.33, 55.52, 53.30, 58.23, 45.07, 26.42],
           "independent": [4.32, 6.07, 13.03, 32.25, 39.79, 27.38],
           "soft": [-0.09, 0.03, 0.74, 3.30, 5.33, 4.31]}),
 "fire": ([0.3, 1, 3, 10, 30, 50, 100, 200],
          {"uniform": [4.44, 7.43, 19.44, 53.70, 112.25, 120.48, 100.36, 71.81],
           "independent": [2.71, 5.10, 14.85, 48.85, 116.76, 131.04, 113.96, 82.90],
           "soft": [-0.15, -0.15, 0.22, 2.43, 12.50, 21.70, 34.33, 41.03]}),
}
PANELS = [("cone", "Cone, $\\mathrm{SO}(3)$ (chordal Laplace)", None),
          ("Pre-Pro", "Pre-Pro, $\\mathbb{T}^2$ (product Laplace)", None),
          ("fire", "Fire, $S^2$ (geodesic Laplace)", None)]
SERIES = [("uniform", "uniform prior", "#8c8c8c", "-", "o"),
          ("independent", "statistical prior only", "#7fa7d4", "--", "s"),
          ("soft", "statistical prior + posterior coupling", "#1f5fa8", "-", "o")]
LO, FLOORY = 0.25, 0.5
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (ds, title, gmax) in zip(axes, PANELS):
    gam, vals = G[ds]
    ax.axhspan(LO, 1.0, color="#eef4ee", lw=0, zorder=0)
    for key, lab, col, ls, mk in SERIES:
        y = np.array(vals[key])
        yp = np.where(y > 1, y, FLOORY)
        ax.plot(gam, yp, ls=ls, color=col, lw=1.1, zorder=2)
        ax.scatter(np.array(gam)[y > 1], y[y > 1], s=14, color=col, marker=mk, zorder=3, lw=0)
        ax.scatter(np.array(gam)[y <= 1], [FLOORY] * int((y <= 1).sum()), s=14, marker=mk, facecolor="white",
                   edgecolor=col, lw=0.9, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(LO, 400)
    ax.set_title(title, fontsize=8.5, pad=4); ax.set_xlabel(r"kernel bandwidth $\gamma$", fontsize=8)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    if ax is not axes[0]: ax.tick_params(axis="y", left=False)
axes[0].set_ylabel("distance from data\n(noise-floor s.d.)", fontsize=8)
h = [plt.Line2D([], [], color=c, ls=ls, marker=mk, ms=3.5, lw=1.1, label=lab) for _, lab, c, ls, mk in SERIES]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.07))
fig.subplots_adjust(left=0.1, right=0.995, top=0.88, bottom=0.27)
fig.savefig("fig_gamma.pdf", bbox_inches="tight"); fig.savefig("fig_gamma.png", dpi=200, bbox_inches="tight")
