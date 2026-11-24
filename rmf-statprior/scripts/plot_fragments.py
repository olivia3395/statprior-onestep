"""
3D ensembles of backbone fragments sampled for given sequences (downstream application, Section sec:downstream).

For a few real segments of L residues, (phi, psi) is sampled at every position from the residue-type-conditioned model
(one step, uniform and statistical prior), and the backbone is rebuilt with ideal geometry. All fragments share the
frame of their first residue, so each ensemble fans out from a common origin. Fragments that pass validation (no
Ramachandran outlier, no clash) are drawn in blue, the others in red, with outlier residues marked; the real fragment
is drawn in black. The ensembles are also written as multi-model PDB files (N, CA, C atoms) for molecular viewers.

    python scripts/plot_fragments.py --out figs_new/fig_fragments.pdf --pdb_dir figs_new/fragments_pdb
"""
import argparse, glob, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import style
import backbone as BB
from eval_synth import load_model
from make_peptide_windows import load as load_residues
from downstream_fragments import rama_regions, classify
from rmf.data.amino_datamodule import COND_CLASSES

THREE2ONE = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L",
                 LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
BAD = "#c44e52"


def real_segments(df, L):
    cmap = {a: k for k, a in enumerate(COND_CLASSES)}
    segs = []
    for (prot, chain), g in df.groupby(["prot", "chain"], sort=False):
        num = g["num"].to_numpy(); res = g["res"].to_numpy()
        ang = np.radians(g[["phi", "psi"]].to_numpy()) % (2 * np.pi)
        cl = g["amino"].map(cmap).to_numpy()
        for run in np.split(np.arange(len(num)), np.where(np.diff(num) != 1)[0] + 1):
            for s in range(0, len(run) - L + 1, L):
                idx = run[s:s + L]
                seq = "".join(THREE2ONE.get(str(r).upper(), "X") for r in res[idx])
                segs.append(dict(name=f"{prot}:{chain}:{num[idx[0]]}", seq=seq, ang=ang[idx],
                                 cls=cl[idx].astype(int), res=[str(r).upper() for r in res[idx]]))
    return segs


def pick_examples(segs, k, rng):
    """One segment containing glycine, one containing proline, one without either (when available)."""
    out = []
    for cond in [lambda s: "G" in s["seq"] and "P" not in s["seq"], lambda s: "P" in s["seq"],
                 lambda s: "G" not in s["seq"] and "P" not in s["seq"]]:
        cand = [s for s in segs if cond(s) and "X" not in s["seq"]]
        out.append(cand[rng.integers(len(cand))])
    return out[:k]


def write_pdb(path, coords_list, resnames):
    lines = []
    for m, X in enumerate(coords_list, 1):
        lines.append(f"MODEL     {m:4d}")
        serial = 1
        for i, rn in enumerate(resnames):
            for a, atom in enumerate(["N", "CA", "C"]):
                x, y, z = X[i, a]
                lines.append(f"ATOM  {serial:5d}  {atom:<3s} {rn:>3s} A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                             f"           {atom[0]}")
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="figs_new/fig_fragments.pdf")
    ap.add_argument("--pdb_dir", default="figs_new/fragments_pdb")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(3); torch.manual_seed(0)
    runs = {}
    for cf in sorted(glob.glob(os.path.join(a.root, "*", ".hydra", "config.yaml"))):
        run = os.path.dirname(os.path.dirname(cf)); cfg = OmegaConf.load(cf)
        if str(cfg.data.get("data_type", "")) == "cond4" and int(cfg.seed) == a.seed \
                and os.path.exists(os.path.join(run, "checkpoints", "last.ckpt")):
            runs[str(cfg.model.get("prior_type", "uniform"))] = run
    assert "uniform" in runs and "gmm" in runs, f"need both priors, found {list(runs)}"
    models = {k: load_model(v, dev, "last")[0] for k, v in runs.items()}

    df = load_residues()
    regions = rama_regions(df)
    examples = pick_examples(real_segments(df, a.L), 3, rng)
    os.makedirs(a.pdb_dir, exist_ok=True)

    style.apply()
    fig = plt.figure(figsize=(6.6, 4.6))
    labels = {"uniform": style.TITLE_UNIFORM, "gmm": style.TITLE_STAT}
    for r, prior in enumerate(["uniform", "gmm"]):
        for c, ex in enumerate(examples):
            cls = np.tile(ex["cls"], (a.n, 1))
            with torch.no_grad():
                models[prior].set_condition(torch.as_tensor(cls.reshape(-1), device=dev))
                ang = models[prior].sample_batch(1, cls.size).float().cpu().numpy().reshape(a.n, a.L, 2)
            out, _ = classify(ang, cls, regions)
            fm = BB.fragment_metrics(ang.reshape(a.n, -1))
            ok = (~out.any(1)) & (fm["clash"] < 0.5)
            X = BB.build_backbone(ang.reshape(a.n, -1)); Xr = BB.build_backbone(ex["ang"].reshape(1, -1))[0]
            write_pdb(os.path.join(a.pdb_dir, f"example{c + 1}_{prior}.pdb"), list(X), ex["res"])
            if r == 0:
                write_pdb(os.path.join(a.pdb_dir, f"example{c + 1}_real.pdb"), [Xr], ex["res"])
            ax = fig.add_subplot(2, 3, 3 * r + c + 1, projection="3d")
            for f in range(a.n):
                ca = X[f, :, 1]
                ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], color=style.STAT if ok[f] else BAD, lw=0.7,
                        alpha=0.55 if ok[f] else 0.75)
                bad = out[f]
                if bad.any():
                    ax.scatter(ca[bad, 0], ca[bad, 1], ca[bad, 2], color=BAD, s=6, depthshade=False)
            ca = Xr[:, 1]
            ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], color=style.DATA, lw=1.8)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-60)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_facecolor("white"); pane.set_edgecolor("#dddddd")
            ax.set_title(f"{ex['seq']}\nusable {int(ok.sum())}/{a.n}", fontsize=7.5, pad=0)
            if c == 0:
                ax.text2D(-0.12, 0.5, labels[prior], rotation=90, va="center", ha="center", transform=ax.transAxes,
                          fontsize=8.5)
            print(f"[fragments] {prior:7s} {ex['name']} {ex['seq']}: usable {int(ok.sum())}/{a.n}, "
                  f"outlier residues {int(out.sum())}, clashes {int((fm['clash'] > 0.5).sum())}")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=style.STAT, lw=1.2, label="passes validation"),
          Line2D([], [], color=BAD, lw=1.2, label="Ramachandran outlier or clash"),
          Line2D([], [], color=BAD, marker="o", ls="", ms=3, label="outlier residue"),
          Line2D([], [], color=style.DATA, lw=1.8, label="real fragment")]
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.08, wspace=0.0, hspace=0.12)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=170, bbox_inches="tight")
    print(f"written {a.out} and PDB files in {a.pdb_dir}")


if __name__ == "__main__":
    main()
