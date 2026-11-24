"""
Ramachandran densities of residue-type-conditioned generation: held-out data, and one-step samples of the single
conditional model with the uniform prior and with the statistical prior, for General, Glycine, Proline and Pre-Pro.

    python scripts/plot_cond.py --out figs_new/fig_cond.pdf
"""
import argparse, glob, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style (scripts/style.py)
from omegaconf import OmegaConf
from eval_synth import load_model
from eval_cond import class_split
import plot_density as PD
from rmf.data.amino_datamodule import COND_CLASSES


def pick(root, seed):
    """Conditional runs (data_type cond4) of the requested seed, one per prior type."""
    out = {}
    for cf in sorted(glob.glob(os.path.join(root, "*", ".hydra", "config.yaml"))):
        run = os.path.dirname(os.path.dirname(cf)); cfg = OmegaConf.load(cf)
        if str(cfg.data.get("data_type", "")) != "cond4" or int(cfg.seed) != seed:
            continue
        if not os.path.exists(os.path.join(run, "checkpoints", "last.ckpt")):
            continue
        out[str(cfg.model.get("prior_type", "uniform"))] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--data_dir", default="data/top500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="figs_new/fig_cond.pdf")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    runs = pick(a.root, a.seed)
    assert "uniform" in runs and "gmm" in runs, f"need both priors for seed {a.seed}, found {list(runs)}"
    for k, v in runs.items():
        print(f"[plot_cond] {k}: {v}")
    models = {k: load_model(v, dev, "last")[0] for k, v in runs.items()}

    style.apply()
    fig, axes = plt.subplots(len(COND_CLASSES), 3, figsize=(6.4, 2.05 * len(COND_CLASSES)))
    titles = [style.TITLE_DATA, style.TITLE_UNIFORM, style.TITLE_STAT]
    for i, amino in enumerate(COND_CLASSES):
        test = class_split(a.data_dir, amino)[1]
        panels = [test]
        for prior in ["uniform", "gmm"]:
            m = models[prior]
            with torch.no_grad():
                m.set_condition(torch.full((a.n,), i, dtype=torch.long, device=dev))
                panels.append(m.sample_batch(1, a.n).float().cpu())
        Ds = [PD.density_torus(p) for p in panels]
        vmax = max(D.max() for D in Ds)
        for j, D in enumerate(Ds):
            ax = axes[i, j]
            PD.draw_torus(ax, D, vmax)
            if i == 0:
                ax.set_title(titles[j])
            if j == 0:
                ax.set_ylabel(r"$\psi$", fontsize=7)
                ax.text(-0.42, 0.5, amino, rotation=90, va="center", ha="center", transform=ax.transAxes)
            else:
                ax.set_yticklabels([])
            if i == len(COND_CLASSES) - 1:
                ax.set_xlabel(r"$\phi$", fontsize=7)
            else:
                ax.set_xticklabels([])
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.06, wspace=0.08, hspace=0.12)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=160, bbox_inches="tight")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
