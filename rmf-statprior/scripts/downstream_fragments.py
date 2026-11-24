"""
Downstream application: sequence-conditioned backbone fragment sampling.

Fragment libraries (as used, for example, in loop modelling) are built by sampling backbone conformations for a given
sequence segment. Here, real segments of L consecutive residues are taken from the Top500 chains, and only their
sequence (residue type per position: General, Glycine, Proline or Pre-Pro) is used. For every position, (phi, psi) is
sampled from the residue-type-conditioned model, and the backbone is rebuilt with ideal geometry (scripts/backbone.py).

Metrics, following standard structure validation:
  outlier    fraction of residues outside the allowed Ramachandran region of their type (99.95% density contour of
             the training split of that type, as in MolProbity-style validation)
  favoured   fraction of residues inside the favoured region (98% contour)
  clash      fraction of fragments with a severe steric clash (CA atoms >= 3 residues apart closer than 3.0 A)
  usable     fraction of fragments with neither a Ramachandran outlier nor a clash
References: independent draws of held-out (phi, psi) of the same residue type (the best any per-residue sampler can
do), and the real segments themselves.

    python scripts/downstream_fragments.py --csv downstream.csv > downstream.log 2>&1
"""
import argparse, glob, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter
import backbone as BB
from eval_synth import load_model
from make_peptide_windows import load as load_residues
from rmf.data.amino_datamodule import COND_CLASSES

BINS = 180  # 2-degree bins on the torus


def type_splits(root="data/top500"):
    """Training and test angles (radians) of each residue type, with the split used for training."""
    import torch
    from torch.utils.data import random_split
    from rmf.data.amino_datamodule import Top500
    out = {}
    for amino in COND_CLASSES:
        ds = Top500(root=root, amino=amino)
        n_val = int(0.1 * len(ds)); n_test = int(0.1 * len(ds)); n_train = len(ds) - n_val - n_test
        tr, _, te = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
        out[amino] = (ds.data[torch.as_tensor(tr.indices)].numpy(), ds.data[torch.as_tensor(te.indices)].numpy())
    return out


def rama_regions(splits, sigma=1.5):
    """Per residue type, from the training split only: smoothed density on a 2-degree grid and the 99.95% (allowed)
    and 98% (favoured) levels."""
    out = {}
    for amino in COND_CLASSES:
        ang = splits[amino][0] % (2 * np.pi)
        H, _, _ = np.histogram2d(ang[:, 0], ang[:, 1], bins=BINS, range=[[0, 2 * np.pi], [0, 2 * np.pi]])
        H = gaussian_filter(H, sigma=sigma, mode="wrap"); H = H / H.sum()
        v = np.sort(H.ravel())[::-1]; c = np.cumsum(v)
        allowed = v[np.searchsorted(c, 0.9995)]; favoured = v[np.searchsorted(c, 0.98)]
        out[amino] = (H, allowed, favoured)
    return out


def classify(ang, cls, regions):
    """ang: (n, L, 2) radians; cls: (n, L) class indices -> boolean arrays outlier, favoured."""
    i = np.clip((ang[..., 0] % (2 * np.pi)) / (2 * np.pi) * BINS, 0, BINS - 1).astype(int)
    j = np.clip((ang[..., 1] % (2 * np.pi)) / (2 * np.pi) * BINS, 0, BINS - 1).astype(int)
    out = np.zeros(cls.shape, bool); fav = np.zeros(cls.shape, bool)
    for k, amino in enumerate(COND_CLASSES):
        H, allowed, favoured = regions[amino]
        m = cls == k
        d = H[i[m], j[m]]
        out[m] = d < allowed; fav[m] = d >= favoured
    return out, fav


def segments(df, L, n, rng):
    """Random real segments of L consecutive residues: torsions (n, L, 2) and residue classes (n, L)."""
    cmap = {a: k for k, a in enumerate(COND_CLASSES)}
    ang_all, cls_all = [], []
    for (_, _), g in df.groupby(["prot", "chain"], sort=False):
        num = g["num"].to_numpy()
        ang = np.radians(g[["phi", "psi"]].to_numpy()) % (2 * np.pi)
        cl = g["amino"].map(cmap).to_numpy()
        for run in np.split(np.arange(len(num)), np.where(np.diff(num) != 1)[0] + 1):
            for s in range(0, len(run) - L + 1, L):
                idx = run[s:s + L]
                if not np.isnan(cl[idx].astype(float)).any():
                    ang_all.append(ang[idx]); cls_all.append(cl[idx].astype(int))
    ang_all, cls_all = np.stack(ang_all), np.stack(cls_all)
    pick = rng.choice(len(ang_all), size=min(n, len(ang_all)), replace=False)
    return ang_all[pick], cls_all[pick]


def metrics(ang, cls, regions):
    out, fav = classify(ang, cls, regions)
    fm = BB.fragment_metrics(ang.reshape(len(ang), -1))
    clash = fm["clash"].astype(bool)
    return dict(outlier=100 * out.mean(), favoured=100 * fav.mean(), clash=100 * clash.mean(),
                usable=100 * ((~out.any(1)) & (~clash)).mean(), helix=100 * fm["helix"].mean(),
                sheet=100 * fm["sheet"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--csv", default="downstream.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0); torch.manual_seed(0)
    df = load_residues()
    splits = type_splits()
    regions = rama_regions(splits)
    ang, cls = segments(df, a.L, a.n, rng)
    print(f"{len(ang)} segments of {a.L} residues; class frequencies:",
          {c: round(float((cls == k).mean()), 3) for k, c in enumerate(COND_CLASSES)})
    rows = [dict(method="real segments", prior="-", steps=0, seed=-1, **metrics(ang, cls, regions))]
    # independent draws of held-out (test-split) angles of the same residue type
    ind = np.empty_like(ang)
    for k, amino in enumerate(COND_CLASSES):
        pool = splits[amino][1] % (2 * np.pi)
        m = cls == k
        ind[m] = pool[rng.integers(0, len(pool), size=int(m.sum()))]
    rows.append(dict(method="independent real draws", prior="-", steps=0, seed=-1, **metrics(ind, cls, regions)))
    # conditional models
    for cf in sorted(glob.glob(os.path.join(a.root, "*", ".hydra", "config.yaml"))):
        run = os.path.dirname(os.path.dirname(cf)); cfg = OmegaConf.load(cf)
        if str(cfg.data.get("data_type", "")) != "cond4" or not os.path.exists(os.path.join(run, "checkpoints", "last.ckpt")):
            continue
        model = load_model(run, dev, "last")[0]
        prior = str(cfg.model.get("prior_type", "uniform"))
        flat = torch.as_tensor(cls.reshape(-1), device=dev)
        for steps in a.steps:
            with torch.no_grad():
                model.set_condition(flat)
                gen = model.sample_batch(steps, len(flat)).float().cpu().numpy().reshape(ang.shape)
            rows.append(dict(method="conditional model", prior=prior, steps=steps, seed=int(cfg.seed),
                             **metrics(gen, cls, regions)))
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    res = pd.DataFrame(rows); res.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    cols = ["outlier", "favoured", "clash", "usable", "helix", "sheet"]
    print("\nPercentages; mean (s.d.) over seeds for the conditional models:")
    print(res.groupby(["method", "prior", "steps"])[cols].agg(["mean", "std"]).round(2).to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
