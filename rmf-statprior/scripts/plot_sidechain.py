"""
Side-chain (chi1, chi2) densities of a few residue types: held-out data and one-step samples of the conditional model
(14 residue types) with the uniform prior and with the statistical prior.

    python scripts/plot_sidechain.py --residues PRO LEU PHE --out figs_new/fig_sidechain.pdf
"""
import argparse, glob, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import style
import plot_density as PD
from eval_synth import load_model
from eval_sidechain import split
from rmf.data.amino_datamodule import SC_CLASSES

NAMES = {"PRO": "Proline", "LEU": "Leucine", "PHE": "Phenylalanine", "ILE": "Isoleucine", "TRP": "Tryptophan",
         "ARG": "Arginine", "LYS": "Lysine", "MET": "Methionine", "TYR": "Tyrosine", "HIS": "Histidine",
         "ASN": "Asparagine", "ASP": "Aspartate", "GLN": "Glutamine", "GLU": "Glutamate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--data_dir", default="data/sidechain")
    ap.add_argument("--residues", nargs="+", default=["PRO", "LEU", "PHE"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="figs_new/fig_sidechain.pdf")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    runs = {}
    for cf in sorted(glob.glob(os.path.join(a.root, "*", ".hydra", "config.yaml"))):
        run = os.path.dirname(os.path.dirname(cf)); cfg = OmegaConf.load(cf)
        if str(cfg.data.get("data_type", "")) == "chi14" and int(cfg.seed) == a.seed \
                and os.path.exists(os.path.join(run, "checkpoints", "last.ckpt")):
            runs[str(cfg.model.get("prior_type", "uniform"))] = run
    assert "uniform" in runs and "gmm" in runs, f"need both priors, found {list(runs)}"
    for k, v in runs.items():
        print(f"[plot_sidechain] {k}: {v}")
    models = {k: load_model(v, dev, "last")[0] for k, v in runs.items()}
    style.apply()
    fig, axes = plt.subplots(len(a.residues), 3, figsize=(6.4, 2.05 * len(a.residues)), squeeze=False)
    titles = [style.TITLE_DATA, style.TITLE_UNIFORM, style.TITLE_STAT]
    for i, rn in enumerate(a.residues):
        k = SC_CLASSES.index(rn)
        panels = [split(a.data_dir, rn)[2]]
        for prior in ["uniform", "gmm"]:
            with torch.no_grad():
                models[prior].set_condition(torch.full((a.n,), k, dtype=torch.long, device=dev))
                panels.append(models[prior].sample_batch(1, a.n).float().cpu())
        Ds = [PD.density_torus(p) for p in panels]
        vmax = max(D.max() for D in Ds)
        for j, D in enumerate(Ds):
            ax = axes[i, j]
            PD.draw_torus(ax, D, vmax)
            if i == 0:
                ax.set_title(titles[j])
            ax.set_xlabel(r"$\chi_1$" if i == len(a.residues) - 1 else "", fontsize=7)
            if i != len(a.residues) - 1:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(r"$\chi_2$", fontsize=7)
                ax.text(-0.42, 0.5, NAMES.get(rn, rn), rotation=90, va="center", ha="center", transform=ax.transAxes)
            else:
                ax.set_ylabel(""); ax.set_yticklabels([])
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.07, wspace=0.08, hspace=0.12)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=160, bbox_inches="tight")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
