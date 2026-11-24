"""Prior vs objective vs number of steps, under the RMF kernel (top) and at the finest kernel scale (bottom).
Values: one-step and four-step MMD^2 x 1e3 from the v2 evaluations (v2_eval_so3, v2_eval_torus_last, v2_eval_earth)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 0,
                     "ytick.major.size": 2.5, "axes.unicode_minus": False})
# V[row][dataset][objective][prior] = (1 step, 4 steps)
V = {
 "rmf": {
  "cone":   {"mf": {"u": (2.167, 0.838), "s": (-0.043, -0.065)}, "imf": {"u": (3.683, 0.474), "s": (-0.109, -0.088)},
             "alpha-f": {"u": (240.071, 10.820), "s": (-0.064, -0.071)}, "pcgrad": {"u": (2.055, 0.743), "s": (-0.032, -0.039)}},
  "prepro": {"mf": {"u": (12.978, 0.507), "s": (-0.079, -0.085)}, "imf": {"u": (318.184, 316.138), "s": (-0.031, -0.085)},
             "alpha-f": {"u": (26.598, 7.746), "s": (11.604, 0.799)}},
  "fire":   {"mf": {"u": (0.659, -0.177), "s": (-0.221, -0.212)}, "imf": {"u": (0.492, -0.145), "s": (-0.279, -0.277)},
             "alpha-f": {"u": (130.689, 3.278), "s": (-0.177, -0.226)}}},
 "fine": {
  "cone":   {"mf": {"u": (4.489, 1.454), "s": (0.003, 0.002)}, "imf": {"u": (4.703, 2.008), "s": (0.005, 0.000)},
             "alpha-f": {"u": (14.153, 6.846), "s": (0.002, -0.004)}, "pcgrad": {"u": (4.404, 1.340), "s": (0.051, 0.057)}},
  "prepro": {"mf": {"u": (6.119, 3.510), "s": (0.683, 0.164)}, "imf": {"u": (13.455, 13.461), "s": (0.400, 0.115)},
             "alpha-f": {"u": (26.026, 10.768), "s": (37.683, 1.200)}},
  "fire":   {"mf": {"u": (5.373, 5.187), "s": (0.983, 0.538)}, "imf": {"u": (4.933, 3.289), "s": (0.816, 0.503)},
             "alpha-f": {"u": (8.603, 5.907), "s": (13.677, 1.596)}}},
}
FLOOR = {"rmf": {"cone": 0.156, "prepro": 0.157, "fire": 0.403}, "fine": {"cone": 0.0285, "prepro": 0.105, "fire": 0.049}}
KER = {"rmf": {"cone": "RMF kernel", "prepro": "RMF kernel", "fire": "RMF kernel"},
       "fine": {"cone": r"chordal Laplace, $\gamma=20$", "prepro": r"product Laplace, $\gamma=10$", "fire": r"Laplace, $\gamma=50$"}}
PANELS = [("cone", r"Cone, $\mathrm{SO}(3)$", ["mf", "imf", "alpha-f", "pcgrad"]),
          ("prepro", r"Pre-Pro, $\mathbb{T}^2$", ["mf", "imf", "alpha-f"]),
          ("fire", r"Fire, $S^2$", ["mf", "imf", "alpha-f"])]
NAMES = {"mf": "MeanFlow", "imf": "iMF", "alpha-f": r"$\alpha$-Flow", "pcgrad": "RMF-MT"}
GREY, BLUE, LO, HI, FLOORY = "#8c8c8c", "#1f5fa8", 0.25, 5000, 0.5
y = lambda v: v if v > 1 else FLOORY

def arrow(ax, x, a, b, c):
    if y(a) / y(b) > 1.6:
        ax.annotate("", xy=(x, y(b)), xytext=(x, y(a)),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, mutation_scale=6, shrinkA=2.5, shrinkB=2.5))

fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.1), sharey=True,
                         gridspec_kw=dict(width_ratios=[4, 3, 3], wspace=0.08, hspace=0.42))
for r, row in enumerate(["rmf", "fine"]):
    for c, (ds, title, methods) in enumerate(PANELS):
        ax = axes[r, c]; f = FLOOR[row][ds]
        ax.axhspan(LO, 1.0, color="#eef4ee", zorder=0, lw=0)
        for i, m in enumerate(methods):
            u1, u4 = (v / f for v in V[row][ds][m]["u"]); s1, s4 = (v / f for v in V[row][ds][m]["s"])
            xu, xs = i - 0.13, i + 0.13
            arrow(ax, xu, u1, u4, GREY)
            ax.scatter([xu], [y(u1)], s=16, color=GREY, zorder=3, lw=0)
            ax.scatter([xu], [y(u4)], s=16, facecolor="white", edgecolor=GREY, lw=0.9, zorder=3)
            ax.scatter([xs], [y(s1)], s=18, color=BLUE, zorder=4, lw=0)
            if s1 > 1:
                arrow(ax, xs, s1, s4, BLUE)
                ax.scatter([xs], [y(s4)], s=18, facecolor="white", edgecolor=BLUE, lw=0.9, zorder=4)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([NAMES[m] for m in methods], fontsize=7)
        ax.set_xlim(-0.55, len(methods) - 0.45); ax.set_yscale("log"); ax.set_ylim(LO, HI)
        ax.set_title(f"{title} ({KER[row][ds]})", fontsize=7.5, pad=3)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        if c: ax.tick_params(axis="y", left=False)
    axes[r, 0].set_ylabel(("RMF kernel" if row == "rmf" else "finest scale") + "\ndistance from data\n(noise-floor s.d.)",
                          fontsize=7.5)
h = [plt.Line2D([], [], color=GREY, marker="o", ms=4, ls="", label="uniform prior, 1 step"),
     plt.Line2D([], [], color=GREY, marker="o", ms=4, mfc="white", ls="", label="uniform prior, 4 steps"),
     plt.Line2D([], [], color=BLUE, marker="o", ms=4, ls="", label="statistical prior, 1 step"),
     plt.Line2D([], [], color=BLUE, marker="o", ms=4, mfc="white", ls="", label="statistical prior, 4 steps")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.2, columnspacing=1.4)
fig.subplots_adjust(left=0.12, right=0.995, top=0.95, bottom=0.12)
fig.savefig("fig_axes.pdf", bbox_inches="tight"); fig.savefig("fig_axes.png", dpi=170, bbox_inches="tight")
