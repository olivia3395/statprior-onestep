"""
Result figures for the paper, drawn from the numbers in the result tables.
Edit the dictionaries below when new results arrive and rerun:
    python make_figs.py            # writes figs/*.pdf (and *.png previews)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
import style
style.apply()
C_UNI, C_VMF, C_STAT, C_FLOOR = style.UNIFORM, style.KDE, style.STAT, style.FLOOR


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2
# One-step MMD^2 (x1e-3) vs concentration kappa (Table 2)
KAPPA = [5, 25, 100, 250]
CONC = {
    "lap10": {"uniform": [0.155, 1.335, 5.232, 13.133], "vmf": [0.419, 0.254, 0.132, 1.005],
              "floor": [0.007, 0.021, 0.017, 0.024]},
    "lap50": {"uniform": [0.011, 0.326, 2.853, 7.470], "vmf": [0.076, 0.115, 0.076, 0.357],
              "floor": [0.002, 0.003, 0.004, 0.012]},
}


def fig_concentration():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for ax, (key, title) in zip(axes, [("lap10", r"Laplace $\gamma=10$"), ("lap50", r"Laplace $\gamma=50$")]):
        d = CONC[key]
        ax.fill_between(KAPPA, 1e-4, 2 * np.array(d["floor"]), color=C_FLOOR, alpha=0.5, lw=0, label="noise floor (2 s.d.)")
        ax.plot(KAPPA, d["uniform"], "o-", color=C_UNI, label="uniform prior")
        ax.plot(KAPPA, d["vmf"], "s-", color=C_VMF, label="statistical prior")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 40)
        ax.set_xticks(KAPPA); ax.set_xticklabels([str(k) for k in KAPPA])
        ax.set_xlabel(r"data concentration $\kappa$"); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(r"one-step MMD$^2$ ($\times10^{-3}$)")
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_concentration")


# ------------------------------------------------------------------ Fig. 3
# Audit: floor vs published numbers (published protocol), Table 1
PUB = {  # RFM, RCT, G-LSD, RMF, RMF-MT (Zhong et al., 2026)
    "Volcano": [0.351, 0.155, 0.115, 0.092, 0.102], "Earthquake": [0.309, 0.053, 0.032, 0.042, 0.035],
    "Flood": [0.272, 0.086, 0.065, 0.068, 0.048], "Fire": [0.377, 0.080, 0.027, 0.042, 0.032],
    "General": [0.45, 0.01, 0.02, 0.11, 0.04], "Glycine": [0.27, 0.04, 0.03, 0.04, 0.03],
    "Proline": [0.52, 0.05, 0.04, 0.09, 0.04], "Pre-Pro": [0.47, 0.06, 0.05, 0.09, 0.05],
    "RNA": [0.68, 0.11, 0.08, 0.20, 0.07],
    "Cone": [0.311, 0.096, 0.044, 0.057, 0.049], "Fisher": [0.154, 0.071, 0.073, 0.076, 0.039],
    "Line": [0.095, 0.051, 0.037, 0.038, 0.035],
}
FLOOR = {"Volcano": (0.121, 0.035), "Earthquake": (0.044, 0.015), "Flood": (0.047, 0.011),
         "Fire": (0.029, 0.010), "General": (0.0064, 0.0016), "Glycine": (0.0245, 0.0073),
         "Proline": (0.0232, 0.0097), "Pre-Pro": (0.0274, 0.0092), "RNA": (0.0193, 0.0031),
         "Cone": (0.0224, 0.0060), "Fisher": (0.0217, 0.0031), "Line": (0.0208, 0.0027)}
SHUFFLED = {"General": 0.0074, "Glycine": 0.0294, "Proline": 0.0232, "Pre-Pro": 0.0315, "RNA": 0.0204}
METHODS = ["RFM", "RCT", "G-LSD", "RMF", "RMF-MT"]
MARK = ["v", "^", "D", "o", "s"]


def fig_audit():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = list(PUB)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for i, n in enumerate(names):
        m, s_ = FLOOR[n]
        ax.add_patch(plt.Rectangle((i - 0.38, 1e-3), 0.76, m + 2 * s_ - 1e-3, color=style.FLOOR, lw=0, zorder=1))
        ax.plot([i - 0.38, i + 0.38], [m, m], color=style.FLOOR_TEXT, lw=0.9, zorder=2)
    for j, (meth, mk) in enumerate(zip(METHODS, MARK)):
        ax.scatter(x + (j - 2) * 0.12, [PUB[n][j] for n in names], marker=mk, s=16, color=style.BASELINES[j],
                   edgecolor="white", linewidth=0.4, zorder=3)
    xs = [names.index(n) for n in SHUFFLED]
    ax.scatter(xs, list(SHUFFLED.values()), marker="x", s=36, color=style.DATA, lw=1.3, zorder=4)
    ax.set_yscale("log"); ax.set_ylim(4e-3, 1.0); ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    for b in [3.5, 8.5]:
        ax.axvline(b, color=style.EDGE, lw=0.6, ls=":")
    for xc, lab in [(1.5, r"$S^2$ (Earth)"), (6, r"$\mathbb{T}^d$"), (10, r"$\mathrm{SO}(3)$")]:
        ax.text(xc, 0.75, lab, ha="center", fontsize=8)
    ax.set_ylabel("one-step MMD (RMF protocol)")
    style.clean(ax)
    hs = [Line2D([], [], marker=mk, ls="", ms=4.5, color=style.BASELINES[j], label=meth)
          for j, (meth, mk) in enumerate(zip(METHODS, MARK))]
    hs += [Line2D([], [], marker="x", ls="", ms=5, color=style.DATA, mew=1.3, label=r"trivial model ($\phi\perp\psi$)"),
           Patch(color=style.FLOOR, label="noise floor (mean $+2$ s.d.)")]
    ax.legend(handles=hs, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.36), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig3_audit")


# ------------------------------------------------------------------ Fig. 4
# Prior inflation s (Table 7). Values <= 0 are within the noise floor and drawn at the floor.
WIDTH_S = [1, 2, 4, 8]
WIDTH = {
    "arc30": {"prior_lap50": [-0.002, 0.401, 1.171, 1.995], "net_lap50": [0.193, 0.273, 0.421, 0.933],
              "prior_ll": [0.145, -0.707, -2.127, -4.239], "net_ll": [0.097, 0.042, -0.429, -1.364],
              "uniform_lap50": 1.720, "uniform_ll": -3.830, "floor": 0.009, "true_ll": 0.161,
              "title": "thin arcs (30 mrad)"},
    "ring30": {"prior_lap50": [0.059, 0.133, 0.377, 0.680], "net_lap50": [0.147, 0.183, 0.179, 0.273],
               "prior_ll": [-0.440, -1.106, -2.721, -6.315], "net_ll": [-0.476, -0.588, -0.628, -0.806],
               "uniform_lap50": 0.664, "uniform_ll": -3.312, "floor": 0.009, "true_ll": -0.384,
               "title": "thin rings (30 mrad)"},
}


def fig_width():
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    for r, key in enumerate(["arc30", "ring30"]):
        d = WIDTH[key]
        ax = axes[r, 0]
        ax.axhspan(1e-4, 2 * d["floor"], color=C_FLOOR, alpha=0.5, lw=0, label="noise floor")
        clip = lambda v: [max(x, 2 * d["floor"]) for x in v]
        ax.plot(WIDTH_S, clip(d["prior_lap50"]), "o--", color=C_STAT, mfc="white", label="prior alone")
        ax.plot(WIDTH_S, clip(d["net_lap50"]), "o-", color=C_STAT, label="prior + network (1 step)")
        ax.axhline(d["uniform_lap50"], color=C_UNI, ls=":", label="uniform prior + network")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_ylim(3e-3, 5)
        ax.set_xticks(WIDTH_S); ax.set_xticklabels([str(x) for x in WIDTH_S])
        ax.set_ylabel(r"MMD$^2$ ($\gamma=50$, $\times10^{-3}$)")
        ax.set_title(d["title"], fontsize=9, loc="left")
        ax = axes[r, 1]
        ax.axhline(d["true_ll"], color="k", lw=0.8, label="truth")
        ax.plot(WIDTH_S, d["prior_ll"], "o--", color=C_STAT, mfc="white", label="prior alone")
        ax.plot(WIDTH_S, d["net_ll"], "o-", color=C_STAT, label="prior + network")
        ax.axhline(d["uniform_ll"], color=C_UNI, ls=":", label="uniform prior + network")
        ax.set_xscale("log", base=2); ax.set_xticks(WIDTH_S); ax.set_xticklabels([str(x) for x in WIDTH_S])
        ax.set_ylabel("mean true log-likelihood")
        ax.set_ylim(min(d["prior_ll"] + [d["uniform_ll"]]) - 0.5, max(d["true_ll"], max(d["net_ll"])) + 0.5)
    for ax in axes[1]:
        ax.set_xlabel(r"prior inflation $s$")
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="lower right")
    for r in range(2):
        axes[r, 1].text(8, WIDTH[["arc30", "ring30"][r]]["true_ll"] + 0.15, "truth", ha="right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_width")


# ------------------------------------------------------------------ Fig. 5
# Fire (Table 6): MMD^2 at gamma=10; None = pending
FIRE = [  # (label, value, color, hatch); values <= floor are drawn as "within floor"
    ("uniform\n1 step", 9.450, C_UNI, None), ("uniform\n4 steps", 4.252, C_UNI, "//"),
    ("vMF\n1 step", 0.196, C_VMF, None), ("stat.\n1 step", 0.171, C_STAT, None),
    ("uniform\n1 step", 6.026, C_UNI, None), ("uniform\n4 steps", 0.675, C_UNI, "//"),
    ("vMF\n1 step", -0.103, C_VMF, None), ("stat.\n1 step", -0.019, C_STAT, None),
]
FIRE_FLOOR = 0.147


def fig_fire():
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i, (lab, v, c, h) in enumerate(FIRE):
        if v is None:
            ax.text(i, 0.02, "pending", ha="center", rotation=90, fontsize=7, color="red")
            continue
        if v <= 2 * FIRE_FLOOR:
            ax.bar(i, 0.02, color=c, hatch=h, edgecolor="white", lw=0.5, alpha=0.9)
            ax.text(i, 0.024, "within\nfloor", ha="center", va="bottom", fontsize=6.5)
            continue
        ax.bar(i, v, color=c, hatch=h, edgecolor="white" if h is None else "k", lw=0.5)
    ax.axhline(2 * FIRE_FLOOR, color="k", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_ylim(0.01, 20)
    ax.set_xticks(range(len(FIRE))); ax.set_xticklabels([f[0] for f in FIRE], fontsize=7)
    for xc, lab in [(1.5, "100 epochs"), (5.5, "400 epochs")]:
        ax.text(xc, 12, lab, ha="center", fontsize=8)
    ax.set_ylabel(r"MMD$^2$ ($\gamma=10$, $\times10^{-3}$)")
    ax.axvline(3.5, color="k", lw=0.5, ls=":")
    ax.text(4.0, 2 * FIRE_FLOOR * 1.15, "noise floor (2 s.d.)", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_fire")


# ------------------------------------------------------------------ Fig. 7
# MMD^2 vs number of sampling steps (1, 2, 4), uniform prior vs statistical prior.
# Values from the repeated evaluations (means); values <= 0 are within the noise floor.
STEPS = [1, 2, 4]
STEPCURVES = [  # (title, kernel, uniform, statistical, floor s.d.); finest scale, v2 evaluation
    ("Fire (real)", r"$\gamma=50$", [5.373, 5.479, 5.187], [0.983, 0.66, 0.538], 0.049),
    ("Earthquake (real)", r"$\gamma=50$", [2.365, 1.921, 1.694], [0.567, 0.438, 0.296], 0.1),
    ("Glycine (real)", r"joint $\gamma=10$", [2.176, 1.933, 1.443], [0.189, 0.15, 0.113], 0.062),
    ("Proline (real)", r"joint $\gamma=10$", [15.851, 23.081, 16.115], [0.532, 0.141, 0.184], 0.158),
    ("RNA, 7-D (real)", r"joint $\gamma=10$", [2.798, 2.283, 1.927], [0.17, 0.146, 0.112], 0.01),
    ("Cone", r"$\gamma=20$", [4.489, 2.565, 1.454], [0.003, -0.001, 0.002], 0.0285),
    ("Fisher", r"$\gamma=20$", [3.548, 3.471, 3.021], [0.571, 0.567, 0.566], 0.009),
    ("Swiss roll", r"$\gamma=20$", [27.539, 25.976, 23.681], [0.173, 0.186, 0.192], 0.0495),
]


def fig_steps():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.5), sharex=True)
    for ax, (title, kern, uni, stat, floor) in zip(axes.ravel(), STEPCURVES):
        lo = 2 * floor
        ax.axhspan(1e-4, lo, color=style.FLOOR, lw=0, zorder=0)
        ax.plot(STEPS, uni, "o-", color=style.UNIFORM, ms=3.5, zorder=3)
        stat_c = [max(v, lo) for v in stat]
        ax.plot(STEPS, stat_c, "o-", color=style.STAT, ms=3.5, zorder=4)
        for x, v in zip(STEPS, stat):
            if v <= lo:
                ax.plot(x, lo, "o", color=style.STAT, ms=3.5, mfc="white", zorder=5)
        ax.set_yscale("log"); ax.set_xscale("log", base=2)
        ax.set_xticks(STEPS); ax.set_xticklabels([str(k) for k in STEPS]); ax.minorticks_off()
        ymin = min(lo, min(uni)) / 3
        ax.set_ylim(max(ymin, 1e-3), max(uni) * 3)
        ax.set_title(title)
        ax.text(0.96, 0.94, kern, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#555555")
        style.clean(ax)
    for ax in axes[1]:
        ax.set_xlabel("sampling steps")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"MMD$^2$ ($\times10^{-3}$)")
    hs = [Line2D([], [], color=style.UNIFORM, marker="o", ms=3.5, label="uniform prior"),
          Line2D([], [], color=style.STAT, marker="o", ms=3.5, label="statistical prior (ours)"),
          Line2D([], [], color=style.STAT, marker="o", ms=3.5, mfc="white", ls="", label="within twice the noise floor"),
          Patch(color=style.FLOOR, label="twice the noise-floor s.d.")]
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig7_steps")


# ------------------------------------------------------------------ Fig. KL (why it works)
# Synthetic data with known densities: x = exact KL(p1 || uniform) = E[log p1] + log(4 pi);
# y = log-likelihood lost by one-step samples = E_data[log p1] - E_samples[log p1] (0 = perfect).
KL_POINTS = [  # name, KL (nats), gap uniform, gap ours (structured prior), marker
    ("vMF $\\kappa$=5",   0.35, 0.010, 0.002, "o"),
    ("vMF $\\kappa$=25",  1.54, 0.211, -0.017, "o"),
    ("vMF $\\kappa$=100", 2.84, 1.019, -0.001, "o"),
    ("vMF $\\kappa$=250", 3.74, 4.640, 0.231, "o"),
    ("ring 100",          1.21, 0.232, 0.030, "^"),
    ("ring 30",           2.15, 2.928, 0.204, "^"),
    ("arc 100",           1.71, 0.404, 0.003, "s"),
    ("arc 30",            2.69, 3.991, 0.119, "s"),
]


def fig_kl():
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    ax.axhspan(-0.25, 0.3, color=style.FLOOR, lw=0, zorder=0)
    for name, k, u, o, mk in KL_POINTS:
        if u - max(o, 0) > 0.25:
            ax.add_patch(FancyArrowPatch((k, u - 0.08), (k, max(o, 0) + 0.12), arrowstyle="-|>", mutation_scale=7,
                                         lw=0.8, color=style.PRIOR, zorder=2))
    for name, k, u, o, mk in KL_POINTS:
        ax.scatter(k, u, marker=mk, s=40, color=style.UNIFORM, edgecolor="white", linewidth=0.7, zorder=4)
        ax.scatter(k, max(o, 0), marker=mk, s=40, color=style.STAT, edgecolor="white", linewidth=0.7, zorder=5)
    for name, k, u, o, mk in KL_POINTS:
        if u > 1.5:
            ax.text(k + 0.07, u, name, fontsize=6.5, va="center", color="#555555")
    ax.axhline(0, color=style.FLOOR_TEXT, lw=0.6, zorder=1)
    mh = [Line2D([], [], marker="o", ls="", ms=5.5, mfc=style.UNIFORM, mec="white", label="uniform prior"),
          Line2D([], [], marker="o", ls="", ms=5.5, mfc=style.STAT, mec="white", label="statistical prior (ours)")]
    fh = [Line2D([], [], marker=m, ls="", ms=4.5, mfc="#cfcfcf", mec="#777777", label=l)
          for m, l in [("o", "vMF mixtures"), ("^", "rings"), ("s", "arcs")]]
    l1 = ax.legend(handles=mh, loc="upper left", handletextpad=0.2, borderaxespad=0.3)
    ax.add_artist(l1)
    ax.legend(handles=fh, loc="upper left", bbox_to_anchor=(0.0, 0.79), handletextpad=0.1, labelspacing=0.25)
    ax.set_xlabel(r"divergence from uniformity, $\mathrm{KL}(p_1\,\|\,\mathcal{U})$ (nats)")
    ax.set_ylabel("log-likelihood lost by\none-step samples (nats)")
    ax.set_xlim(0, 4.0); ax.set_ylim(-0.25, 5.1)
    style.clean(ax)
    fig.tight_layout()
    save(fig, "fig_kl")

# ------------------------------------------------------------------ Fig. dimension scaling
DIMS = [2, 5, 10, 20, 50]
DIM_UNI = [0.349, 2.529, 3.090, 2.506, 2.034]        # lower bounds where most samples are off the data
DIM_UNI_BELOW = [0.0, 0.173, 0.721, 0.940, 1.000]    # fraction of samples below the density floor
DIM_KDE = [0.0835, 0.2467, 0.4497, 0.9232, 1.4587]
DIM_PRIOR = [0.389, 0.487, 0.546, 0.611, 0.504]
DIM_OURS = [0.077, 0.115, 0.145, 0.186, 0.231]
DIM_OURS_SD = [0.012, 0.031, 0.003, 0.017, 0.008]


def fig_dim():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.plot(DIMS, DIM_UNI, color=style.UNIFORM, lw=1.3, zorder=2)
    for d, v, b in zip(DIMS, DIM_UNI, DIM_UNI_BELOW):
        ax.scatter(d, v, s=28, zorder=3, color=style.UNIFORM if b < 0.5 else "white", edgecolor=style.UNIFORM,
                   linewidth=1.1)
        if b >= 0.5:
            ax.annotate("", xy=(d, v + 0.42), xytext=(d, v + 0.08),
                        arrowprops=dict(arrowstyle="-|>", color=style.UNIFORM, lw=0.9, mutation_scale=7))
    ax.plot(DIMS, DIM_KDE, "D-", color=style.KDE, ms=4, lw=1.3, mec="white", zorder=3)
    ax.plot(DIMS, DIM_PRIOR, "s--", color=style.PRIOR, ms=4, lw=1.2, mec="white", zorder=3)
    ax.errorbar(DIMS, DIM_OURS, yerr=DIM_OURS_SD, fmt="o-", color=style.STAT, ms=4.5, lw=1.6, mec="white",
                capsize=2, zorder=4)
    ax.set_xscale("log"); ax.set_xticks(DIMS); ax.set_xticklabels([str(d) for d in DIMS]); ax.minorticks_off()
    ax.set_ylim(-0.05, 3.7); ax.set_xlim(1.7, 60)
    ax.set_xlabel("dimension $d$ of the torus")
    ax.set_ylabel("log-likelihood lost\nper dimension (nats)")
    style.clean(ax)
    hs = [Line2D([], [], color=style.UNIFORM, marker="o", ms=4, label="uniform prior"),
          Line2D([], [], color=style.UNIFORM, marker="o", ms=4, mfc="white", ls="", label="  lower bound (samples off the data)"),
          Line2D([], [], color=style.KDE, marker="D", ms=3.5, label="kernel density estimate"),
          Line2D([], [], color=style.PRIOR, marker="s", ms=3.5, ls="--", label="statistical prior alone"),
          Line2D([], [], color=style.STAT, marker="o", ms=4, lw=1.6, label="statistical prior (ours)")]
    ax.legend(handles=hs, loc="center left", handlelength=1.8, labelspacing=0.55, bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save(fig, "fig_dim")


# ------------------------------------------------------------------ Fig. motivation (1-D, exact transport maps)
def _wrapped_mix(x, mus, sds, ws):
    p = np.zeros_like(x)
    for m, sd, w in zip(mus, sds, ws):
        for k in (-1, 0, 1):
            p += w * np.exp(-0.5 * ((x - m + 2 * np.pi * k) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return p


def fig_motivation():
    C_R, C_B, C_D = "#6b6b6b", "#1f6fb4", "#b2182b"
    x = np.linspace(-np.pi, np.pi, 6001)[:-1]; dx = x[1] - x[0]
    mus, sds, ws = [-1.9, 0.35, 1.7], [0.13, 0.22, 0.09], [0.35, 0.45, 0.20]
    p1 = _wrapped_mix(x, mus, sds, ws); p1 /= p1.sum() * dx
    ps = _wrapped_mix(x, mus, [np.sqrt(2) * s for s in sds], ws)      # fitted mixture, inflated (s = 2)
    priors = {"uniform": np.full_like(x, 1 / (2 * np.pi)), "stat": ps / (ps.sum() * dx)}
    F1 = np.cumsum(p1) * dx
    ent1 = -(p1 * np.log(p1 + 1e-300)).sum() * dx

    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.75, 1.25, 0.55, 1.0], hspace=0.05, wspace=0.14)
    titles = {"uniform": "start from the uniform prior (all prior work)",
              "stat": "start from a statistical prior (ours)"}
    for col, key in enumerate(["uniform", "stat"]):
        p0 = priors[key]; F0 = np.cumsum(p0) * dx
        c = C_R if key == "uniform" else C_B
        T = np.interp(F0, F1, x)
        logslope = np.log(p0) - np.log(np.interp(T, x, p1))
        ent0 = -(p0 * np.log(p0)).sum() * dx
        mean_ls = ent1 - ent0
        u = (np.arange(36) + 0.5) / 36
        xs = np.interp(u, F0, x); ys = np.interp(u, F1, x)

        ax0 = fig.add_subplot(gs[0, col])                      # prior
        ax0.fill_between(x, 0, p0, color=c, alpha=0.25, lw=0); ax0.plot(x, p0, color=c, lw=1.2)
        ax0.set_ylim(0, 1.9); ax0.set_xlim(-np.pi, np.pi); ax0.axis("off")
        ax0.set_title(titles[key], fontsize=8.5, color=c, pad=3, weight="bold")
        ax0.text(-np.pi, 1.55, "prior $p_0$", fontsize=7, color=c)
        ax0.text(np.pi, 1.55, f"mean distance travelled: {np.abs(ys - xs).mean():.2f} rad", fontsize=7,
                 color=c, ha="right")

        ax1 = fig.add_subplot(gs[1, col])                      # particles
        for a, b in zip(xs, ys):
            ax1.plot([a, b], [1, 0], color=c, lw=0.8, alpha=0.75)
        ax1.scatter(xs, np.ones_like(xs), s=9, color=c, zorder=3)
        ax1.scatter(ys, np.zeros_like(ys), s=9, color=C_D, zorder=3)
        ax1.set_xlim(-np.pi, np.pi); ax1.set_ylim(-0.06, 1.06); ax1.axis("off")

        ax2 = fig.add_subplot(gs[2, col])                      # data, upside down
        ax2.fill_between(x, 0, -p1, color=C_D, alpha=0.25, lw=0); ax2.plot(x, -p1, color=C_D, lw=1.0)
        ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-1.35, 0); ax2.axis("off")
        ax2.text(-np.pi, -1.1, "data $p_{\\rm data}$", fontsize=7, color=C_D)

        ax3 = fig.add_subplot(gs[3, col])                      # log local volume change
        ax3.axhline(0, color="k", lw=0.5)
        ax3.fill_between(u, 0, np.interp(u, F0, logslope), color=c, alpha=0.15, lw=0)
        ax3.plot(u, np.interp(u, F0, logslope), color=c, lw=1.4)
        ax3.axhline(mean_ls, color=c, lw=0.9, ls="--")
        ax3.set_ylim(-2.6, 1.4); ax3.set_xlim(0, 1); ax3.set_xticks([]); ax3.tick_params(labelsize=6.5)
        if key == "uniform":
            lab = f"average $= -\\mathrm{{KL}}(p_{{\\rm data}}\\,\\|\\,\\mathrm{{uniform}}) = {mean_ls:.2f}$"
        else:
            lab = f"average $= H(p_{{\\rm data}}) - H(p_0) = {mean_ls:.2f}$"
        ax3.text(0.5, 0.97, lab, transform=ax3.transAxes, fontsize=6.8, color=c, va="top", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=5)
        if col == 0:
            ax3.set_ylabel("$\\log T'$ (local\nvolume change)", fontsize=7)
        else:
            ax3.set_yticklabels([])
        ax3.set_xlabel("position along the prior (quantile)", fontsize=7, labelpad=1)
        for sp in ["top", "right"]:
            ax3.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.07)
    save(fig, "fig_motivation")

if __name__ == "__main__":
    fig_concentration(); fig_audit(); fig_width(); fig_fire(); fig_steps(); fig_kl(); fig_dim()
    print("figures written to", OUT)


# ---------------------------------------------------------------------------------------------------------------
# Prior complexity versus transport: number of mixture components K (Table tab:sens), one step, finest scale.
TRADEOFF = {
    # name: (K values, prior alone, prior + network, BIC choice, uniform prior one step, noise-floor s.d., kernel)
    "Pre-Pro": ([2, 4, 8, 16, 32], [6.89, 3.16, 1.60, 0.938, 0.877], [2.73, 1.82, 0.788, 0.586, 0.313], 8, 5.71, 0.105,
                r"joint $\gamma=10$"),
    "Fire": ([8, 16, 32, 64], [5.53, 2.68, 1.71, 1.20], [1.86, 1.43, 0.978, 0.653], 32, 5.37, 0.049, r"$\gamma=50$"),
}


def fig_tradeoff():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for ax, (name, (K, prior, net, kb, uni, floor, kern)) in zip(axes, TRADEOFF.items()):
        ax.axhspan(1e-3, floor, color=style.FLOOR, lw=0, zorder=0)
        ax.axhline(uni, color=style.UNIFORM, lw=1.0, ls=":", zorder=1)
        ax.fill_between(K, net, prior, color=style.PRIOR, alpha=0.18, lw=0, zorder=1)
        ax.plot(K, prior, "s--", color=style.PRIOR, ms=4, lw=1.2, mec="white", zorder=3)
        ax.plot(K, net, "o-", color=style.STAT, ms=4.5, lw=1.6, mec="white", zorder=4)
        i = K.index(kb)
        ax.plot(K[i], net[i], "o", ms=8, mfc="none", mec=style.DATA, mew=0.9, zorder=5)
        ax.annotate("BIC", (K[i], net[i]), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=6.5,
                    color=style.DATA)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K]); ax.minorticks_off()
        ax.set_ylim(floor * 0.6, 12)
        ax.set_title(name)
        ax.text(0.97, 0.94, kern, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#555555")
        ax.set_xlabel("number of components $K$")
        style.clean(ax)
    axes[0].set_ylabel(r"MMD$^2$ ($\times10^{-3}$)")
    hs = [Line2D([], [], color=style.PRIOR, marker="s", ms=3.5, ls="--", label="statistical prior alone"),
          Line2D([], [], color=style.STAT, marker="o", ms=4, lw=1.6, label="prior + one-step network"),
          Line2D([], [], color=style.UNIFORM, ls=":", label="uniform prior + one-step network"),
          Patch(color=style.FLOOR, label="within noise floor")]
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), handlelength=1.8,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig_tradeoff")


# ---------------------------------------------------------------------------------------------------------------
# Residue-type-conditioned generation (Table tab:cond): multiples of the noise-floor s.d., one step.
COND = {  # residue: {(model, prior): (RMF kernel, finest scale)}
    "General": {("per-type", "uniform"): (74.0, 245.6), ("conditional", "uniform"): (168.8, 880.4),
                ("per-type", "statistical"): (1.59, 10.25), ("conditional", "statistical"): (1.01, 34.7)},
    "Glycine": {("per-type", "uniform"): (2.55, 22.0), ("conditional", "uniform"): (15.7, 72.8),
                ("per-type", "statistical"): (-0.83, 1.72), ("conditional", "statistical"): (-0.41, 2.49)},
    "Proline": {("per-type", "uniform"): (18.5, 117.5), ("conditional", "uniform"): (103.2, 112.8),
                ("per-type", "statistical"): (-0.82, 4.13), ("conditional", "statistical"): (-0.60, 4.59)},
    "Pre-Pro": {("per-type", "uniform"): (22.5, 55.0), ("conditional", "uniform"): (23.4, 97.3),
                ("per-type", "statistical"): (-0.68, 5.55), ("conditional", "statistical"): (0.41, 7.99)},
}


def fig_cond_summary():
    from matplotlib.patches import Patch
    keys = [("per-type", "uniform"), ("conditional", "uniform"), ("per-type", "statistical"),
            ("conditional", "statistical")]
    colors = {("per-type", "uniform"): "#c9c9c9", ("conditional", "uniform"): style.UNIFORM,
              ("per-type", "statistical"): style.PRIOR, ("conditional", "statistical"): style.STAT}
    hatch = {"per-type": "", "conditional": ""}
    names = list(COND)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, (col, title) in zip(axes, [(0, "RMF kernel"), (1, r"finest scale (joint $\gamma=10$)")]):
        ax.axhspan(0.3, 1.0, color=style.FLOOR, lw=0, zorder=0)
        x = np.arange(len(names)); w = 0.2
        for j, k in enumerate(keys):
            vals = [max(COND[n][k][col], 0.5) for n in names]
            ax.bar(x + (j - 1.5) * w, vals, w * 0.92, color=colors[k], edgecolor="white", lw=0.4, zorder=2)
        ax.set_yscale("log"); ax.set_ylim(0.3, 2000)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_title(title)
        style.clean(ax)
    axes[0].set_ylabel("noise-floor s.d. (one step)")
    hs = [Patch(color=colors[k], label=f"{k[1]} prior, {k[0]}") for k in keys] + \
         [Patch(color=style.FLOOR, label="within noise floor")]
    fig.legend(handles=hs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05), handlelength=1.2,
               columnspacing=0.8, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig_cond_summary")


# ---------------------------------------------------------------------------------------------------------------
# Real data at a glance (Tables tab:fine-summary and tab:division): multiples of the noise-floor s.d., one step.
# None = within the noise floor.
GLANCE = [  # name, RMF uniform, RMF statistical, finest uniform, finest prior alone, finest statistical
    ("Fire", 1.6, None, 110, 35, 20), ("Earthquake", None, None, 24, 8.6, 5.7), ("Flood", 2.9, 2.7, 15, 8.1, 9.0),
    ("Volcano", None, None, 4.6, None, 1.0), ("General", 176, None, 442, 149, 13), ("Glycine", 1.3, None, 35, 11, 3.0),
    ("Proline", 31, None, 100, 18, 3.4), ("Pre-Pro", 49, None, 54, 15, 7.3), ("RNA (7-D)", 146, None, 280, 95, 17),
]


def fig_glance():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    names = [g[0] for g in GLANCE][::-1]
    rows = GLANCE[::-1]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    def pt(ax, x, yy, color, marker, size, z):
        if x is None:
            xin = 0.66 if color == style.UNIFORM else 0.84
            ax.scatter(xin, yy, s=size, marker=marker, facecolor="white", edgecolor=color, lw=1.0, zorder=z)
        else:
            ax.scatter(x, yy, s=size, marker=marker, color=color, edgecolor="white", lw=0.5, zorder=z)
    for ax, title in zip(axes, ["RMF kernel", "finest scale"]):
        ax.axvspan(0.5, 1.0, color=style.FLOOR, lw=0, zorder=0)
        ax.set_xscale("log"); ax.set_xlim(0.5, 800)
        ax.set_title(title); ax.set_xlabel("noise-floor standard deviations (one step)")
        for yy in y:
            ax.axhline(yy, color=style.GRID, lw=0.5, zorder=0)
        style.clean(ax, grid=False)
    for yy, r in zip(y, rows):
        _, ru, rs, fu, fp, fs = r
        a = axes[0]
        a.plot([max(ru or 0.8, 0.8), max(rs or 0.8, 0.8)], [yy, yy], color="#cfcfcf", lw=1.2, zorder=1)
        pt(a, ru, yy, style.UNIFORM, "o", 30, 3); pt(a, rs, yy, style.STAT, "o", 30, 4)
        b = axes[1]
        b.plot([fs, fu], [yy, yy], color="#cfcfcf", lw=1.2, zorder=1)
        if fp is not None:
            b.annotate("", xy=(fs, yy), xytext=(fp, yy), arrowprops=dict(arrowstyle="-|>", color=style.PRIOR, lw=0.9,
                                                                           mutation_scale=7), zorder=2)
        pt(b, fu, yy, style.UNIFORM, "o", 30, 3); pt(b, fp, yy, style.PRIOR, "s", 22, 3); pt(b, fs, yy, style.STAT, "o", 30, 4)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names)
    axes[0].axhline(4.5, color=style.EDGE, lw=0.6, ls=":"); axes[1].axhline(4.5, color=style.EDGE, lw=0.6, ls=":")
    hs = [Line2D([], [], marker="o", ls="", color=style.UNIFORM, ms=5, label="uniform prior"),
          Line2D([], [], marker="s", ls="", color=style.PRIOR, ms=4.5, label="statistical prior alone"),
          Line2D([], [], marker="o", ls="", color=style.STAT, ms=5, label="statistical prior + network"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=style.DATA, ms=5, label="within noise floor"),
          Patch(color=style.FLOOR, label="noise floor")]
    fig.legend(handles=hs, loc="upper center", ncol=5, bbox_to_anchor=(0.55, 1.03), fontsize=6.5, handletextpad=0.3,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig_glance")


# Downstream application (Table tab:downstream)
DOWN = [("uniform\n1 step", 34.1, 12.1, style.UNIFORM), ("uniform\n4 steps", 68.7, 4.24, "#b5b5b5"),
        ("statistical\n1 step", 92.8, 0.55, style.STAT), ("statistical\n4 steps", 94.0, 0.43, style.PRIOR),
        ("independent\nheld-out draws", 95.0, 0.24, "#e0e0e0"), ("real\nsegments", 99.3, 0.03, "#e0e0e0")]


def fig_downstream():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2))
    x = np.arange(len(DOWN))
    for ax, col, title, lim in [(axes[0], 1, "usable fragments (\\%)", 105), (axes[1], 2, "Ramachandran outliers (\\%)", 13.5)]:
        vals = [d[col] for d in DOWN]
        bars = ax.bar(x, vals, 0.7, color=[d[3] for d in DOWN], edgecolor=[style.EDGE if i >= 4 else "white" for i in x],
                      lw=0.6, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + lim * 0.02, f"{v:g}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels([d[0] for d in DOWN], fontsize=6)
        ax.set_ylim(0, lim); ax.set_title(title.replace("\\%", "%"))
        style.clean(ax)
    fig.tight_layout()
    save(fig, "fig_downstream")


# ---------------------------------------------------------------------------------------------------------------
# Side-chain rotamer outliers for all 14 residue types (one step, mean over two seeds)
SIDECHAIN = {  # residue: (uniform, statistical, held-out) rotamer outliers in %
    "PRO": (84.33, 1.16, 0.27), "ILE": (56.58, 2.27, 0.90), "TRP": (54.18, 5.98, 3.06), "LEU": (52.12, 3.22, 1.56),
    "MET": (44.71, 4.41, 2.52), "PHE": (40.78, 4.20, 0.62), "TYR": (39.29, 4.46, 1.25), "ASN": (38.29, 4.34, 2.28),
    "HIS": (37.84, 6.19, 2.46), "ASP": (36.23, 4.21, 1.62), "GLN": (35.57, 4.17, 3.80), "GLU": (30.50, 3.65, 1.76),
    "ARG": (28.65, 3.37, 2.01), "LYS": (22.70, 3.01, 2.39),
}
FULL = dict(PRO="Proline", ILE="Isoleucine", TRP="Tryptophan", LEU="Leucine", MET="Methionine", PHE="Phenylalanine",
            TYR="Tyrosine", ASN="Asparagine", HIS="Histidine", ASP="Aspartate", GLN="Glutamine", GLU="Glutamate",
            ARG="Arginine", LYS="Lysine")


def fig_sidechain_bars():
    from matplotlib.patches import Patch
    names = sorted(SIDECHAIN, key=lambda r: SIDECHAIN[r][0])
    y = np.arange(len(names)); h = 0.26
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for off, col, idx in [(h, style.UNIFORM, 0), (0, style.STAT, 1), (-h, "#cfcfcf", 2)]:
        ax.barh(y + off, [SIDECHAIN[r][idx] for r in names], h * 0.92, color=col, zorder=2, lw=0)
    for yy, r in zip(y, names):
        ax.text(SIDECHAIN[r][0] + 1.0, yy + h, f"{SIDECHAIN[r][0]:.0f}", va="center", fontsize=5.8, color="#555555")
        ax.text(SIDECHAIN[r][1] + 1.0, yy, f"{SIDECHAIN[r][1]:.1f}", va="center", fontsize=5.8, color=style.STAT)
    ax.set_yticks(y); ax.set_yticklabels([FULL[r] for r in names], fontsize=7)
    ax.set_xlabel("rotamer outliers (% of one-step samples)"); ax.set_xlim(0, 92)
    style.clean(ax)
    ax.legend(handles=[Patch(color=style.UNIFORM, label="uniform prior"), Patch(color=style.STAT, label="statistical prior"),
                       Patch(color="#cfcfcf", label="held-out data")], loc="lower right", fontsize=6.5, frameon=False)
    fig.tight_layout(); save(fig, "fig_sidechain_bars")


# Geometry of the learned maps (Table tab:jacobian): required vs achieved contraction, and folding
JAC = {  # dataset: (uniform: contraction, required, folded %), (statistical: contraction, required, folded %)
    "Volcano": ((1.74, 4.25, 33.6), (0.47, 0.04, 28.9)), "Earthquake": ((1.86, 2.10, 18.4), (0.75, 0.33, 14.2)),
    "Flood": ((1.26, 1.55, 15.5), (0.50, 0.33, 8.6)), "Fire": ((2.12, 3.10, 22.3), (0.82, 0.46, 8.2)),
    "General": ((1.88, 2.54, 28.9), (0.51, 0.32, 0.2)), "Glycine": ((1.68, 1.66, 18.1), (0.53, 0.34, 0.5)),
    "Proline": ((2.59, 3.35, 44.9), (0.54, 0.41, 0.3)), "Pre-Pro": ((1.34, 2.34, 37.8), (0.61, 0.35, 1.2)),
    "RNA": ((6.95, 16.84, 51.0), (3.26, 1.63, 13.0)),
}


def fig_geometry():
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    earth = {"Volcano", "Earthquake", "Flood", "Fire"}
    ax = axes[0]
    t = np.logspace(-1.6, 1.4, 50); ax.plot(t, t, color=style.EDGE, lw=0.8, ls="--", zorder=1)
    ax.fill_between(t, 1e-3, t, color="#fbeaea", lw=0, zorder=0)
    ax.text(9.0, 0.42, "below the injective bound:\nthe map must fold", fontsize=6, color="#b04a4c", ha="center")
    for name, (u, s_) in JAC.items():
        mk = "o" if name not in earth else "s"
        ax.scatter(u[1], u[0], s=24, marker=mk, color=style.UNIFORM, edgecolor="white", lw=0.5, zorder=3)
        ax.scatter(s_[1], s_[0], s=24, marker=mk, color=style.STAT, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.025, 25); ax.set_ylim(0.3, 10)
    ax.set_xlabel("required contraction (nats), Proposition 1"); ax.set_ylabel("achieved contraction (nats)")
    ax.set_title("Trained maps versus the injective bound")
    style.clean(ax)
    ax = axes[1]
    for name, (u, s_) in JAC.items():
        mk = "o" if name not in earth else "s"
        ax.annotate("", xy=(s_[1], s_[2] + 0.3), xytext=(u[1], u[2]), arrowprops=dict(arrowstyle="-|>", color="#c9d6e6",
                    lw=0.8, mutation_scale=7, shrinkA=3, shrinkB=3), zorder=1)
        ax.scatter(u[1], u[2], s=24, marker=mk, color=style.UNIFORM, edgecolor="white", lw=0.5, zorder=3)
        ax.scatter(s_[1], s_[2] + 0.3, s=24, marker=mk, color=style.STAT, edgecolor="white", lw=0.5, zorder=3)
        if name in ("RNA", "Proline", "Pre-Pro", "Volcano", "Glycine"):
            ax.text(u[1] * 1.12, u[2] + 1.2, name, fontsize=5.8, color="#555555")
    ax.set_xscale("log"); ax.set_xlim(0.025, 25); ax.set_ylim(0, 56)
    ax.set_xlabel("required contraction (nats), Proposition 1"); ax.set_ylabel("prior samples folded (%)")
    ax.set_title("Folding grows with the required contraction")
    style.clean(ax)
    hs = [Line2D([], [], marker="o", ls="", color=style.UNIFORM, ms=4.5, label="uniform prior"),
          Line2D([], [], marker="o", ls="", color=style.STAT, ms=4.5, label="statistical prior"),
          Line2D([], [], marker="o", ls="", color="#9a9a9a", ms=4.5, label="torus"),
          Line2D([], [], marker="s", ls="", color="#9a9a9a", ms=4.2, label="sphere")]
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "fig_geometry")


def fig_coupling_hist():
    d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coupling_dist.npz"))
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    bins = np.linspace(0, 3.2, 65)
    for key, col, lab in [("independent", style.UNIFORM, "independent"), ("posterior", style.PRIOR, "posterior"),
                          ("hot", style.STAT, "hierarchical OT")]:
        v = d[key]
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=col, alpha=0.35, lw=0, zorder=2)
        ax.hist(v, bins=bins, density=True, histtype="step", color=col, lw=1.1, zorder=3)
        ax.axvline(v.mean(), color=col, lw=0.9, ls="--", zorder=4)
        ax.text(v.mean() + 0.04, ax.get_ylim()[1] * 0.0 + {"independent": 0.55, "posterior": 1.3, "hot": 3.0}[key],
                f"{lab}\nmean {v.mean():.2f}", fontsize=6, color=col if key != "posterior" else "#4f7fb8")
    ax.set_xlim(0, 3.2); ax.set_xlabel("geodesic distance between paired noise and data (rad)"); ax.set_ylabel("density")
    style.clean(ax)
    fig.tight_layout(); save(fig, "fig_coupling_hist")


def fig_cost():
    from matplotlib.patches import Patch
    rows = [("Fire\n($S^2$)", 202, 200, 264), ("Cone\n($SO(3)$)", 14.7, 15.8, 26.6), ("Pre-Pro\n($\\mathbb{T}^2$)", 11.2, 12.6, 34.3),
            ("Fragments\n($\\mathbb{T}^{20}$, HMM)", 11.4, 14.0, None)]
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    x = np.arange(len(rows)); w = 0.26
    for off, idx, col in [(-w, 1, style.UNIFORM), (0, 2, style.STAT), (w, 3, style.PRIOR)]:
        vals = [r[idx] / r[1] if r[idx] else np.nan for r in rows]
        ax.bar(x + off, vals, w * 0.92, color=col, zorder=2, lw=0)
        for xx, v in zip(x + off, vals):
            if np.isfinite(v):
                ax.text(xx, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(1, color=style.EDGE, lw=0.6, ls=":", zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_ylabel("time per training step\n(relative to uniform prior)"); ax.set_ylim(0, 3.5)
    style.clean(ax)
    ax.legend(handles=[Patch(color=style.UNIFORM, label="uniform prior"), Patch(color=style.STAT, label="posterior coupling"),
                       Patch(color=style.PRIOR, label="hierarchical OT")], loc="upper left", fontsize=6.5, frameon=False)
    fig.tight_layout(); save(fig, "fig_cost")
