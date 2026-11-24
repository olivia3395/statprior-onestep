"""
Residue-type-conditioned generation of side-chain dihedral angles (chi1, chi2), 14 residue types.

For every conditional model (data_type chi14) and residue type: MMD^2 to the test split of that type (RMF kernel and
joint Laplace gamma = 10, multiples of the noise floor), and rotamer validation: the fraction of samples outside the
99.7% density contour of the training data of that type (rotamer outliers, as in MolProbity-style validation) and inside
the 98% contour (favoured). The held-out test split, evaluated against the same contours, is the reference.

    python scripts/eval_sidechain.py --csv sidechain_eval.csv > eval_sidechain.log 2>&1
"""
import argparse, glob, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter
from torch.utils.data import random_split
import eval_manifold as EM
from eval_synth import load_model
from rmf.data.amino_datamodule import SideChain, SC_CLASSES

KERNELS = [("rmf", "paper", 1.0), ("joint10", "lap", 10.0)]
BINS = 180


def split(root, rn):
    """full data, training split and test split of one residue type (same split as in training)."""
    ds = SideChain(root=root, resname=rn)
    n_val = int(0.1 * len(ds)); n_test = int(0.1 * len(ds)); n_train = len(ds) - n_val - n_test
    tr, _, te = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    return ds.data, ds.data[torch.as_tensor(tr.indices)], ds.data[torch.as_tensor(te.indices)]


def regions(full, sigma=1.5):
    a = full.numpy() % (2 * np.pi)
    H, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=BINS, range=[[0, 2 * np.pi], [0, 2 * np.pi]])
    H = gaussian_filter(H, sigma=sigma, mode="wrap"); H = H / H.sum()
    v = np.sort(H.ravel())[::-1]; c = np.cumsum(v)
    return H, v[np.searchsorted(c, 0.997)], v[np.searchsorted(c, 0.98)]


def rotamer(x, reg):
    H, allowed, favoured = reg
    a = x % (2 * np.pi)
    i = np.clip((a[:, 0] / (2 * np.pi) * BINS).astype(int), 0, BINS - 1)
    j = np.clip((a[:, 1] / (2 * np.pi) * BINS).astype(int), 0, BINS - 1)
    d = H[i, j]
    return 100 * float((d < allowed).mean()), 100 * float((d >= favoured).mean())


def floors(full, n_test, reps, dev, gen):
    out = {k: [] for k, _, _ in KERNELS}
    for _ in range(reps):
        p = torch.randperm(len(full), generator=gen)
        a, b = full[p[:n_test]].to(dev), full[p[n_test:]][:10000].to(dev)
        for name, kind, g in KERNELS:
            out[name].append(EM.mmd2(a, b, "torus", kind, g))
    return {k: float(np.std(v)) * 1e3 for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--data_dir", default="data/sidechain")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--floor_reps", type=int, default=10)
    ap.add_argument("--csv", default="sidechain_eval.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0); torch.manual_seed(0)
    data = {rn: split(a.data_dir, rn) for rn in SC_CLASSES}
    reg = {rn: regions(train) for rn, (_, train, _) in data.items()}          # contours from training data only
    fl = {rn: floors(full, len(te), a.floor_reps, dev, gen) for rn, (full, _, te) in data.items()}
    rows = []
    # reference: the test split itself (independent real draws)
    for rn, (_, _, te) in data.items():
        o, f = rotamer(te.numpy(), reg[rn])
        rows.append(dict(model="held-out data", prior="-", residue=rn, steps=0, seed=-1, rmf_x=np.nan,
                         joint10_x=np.nan, outlier=o, favoured=f))
    for cf in sorted(glob.glob(os.path.join(a.root, "*", ".hydra", "config.yaml"))):
        run = os.path.dirname(os.path.dirname(cf)); cfg = OmegaConf.load(cf)
        if str(cfg.data.get("data_type", "")) != "chi14" or not os.path.exists(os.path.join(run, "checkpoints", "last.ckpt")):
            continue
        model = load_model(run, dev, "last")[0]
        prior = str(cfg.model.get("prior_type", "uniform"))
        for k, rn in enumerate(SC_CLASSES):
            te = data[rn][2].to(dev)
            for steps in a.steps:
                vals = {name: [] for name, _, _ in KERNELS}; xs = []
                for _ in range(a.reps):
                    with torch.no_grad():
                        model.set_condition(torch.full((a.n,), k, dtype=torch.long, device=dev))
                        x = model.sample_batch(steps, a.n).float()
                    xs.append(x.cpu().numpy())
                    for name, kk, g in KERNELS:
                        vals[name].append(EM.mmd2(x, te, "torus", kk, g))
                o, f = rotamer(np.concatenate(xs), reg[rn])
                row = dict(model="conditional", prior=prior, residue=rn, steps=steps, seed=int(cfg.seed),
                           rmf_x=1e3 * float(np.mean(vals["rmf"])) / fl[rn]["rmf"],
                           joint10_x=1e3 * float(np.mean(vals["joint10"])) / fl[rn]["joint10"], outlier=o, favoured=f)
                rows.append(row)
                print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 220)
    cols = ["rmf_x", "joint10_x", "outlier", "favoured"]
    print("\nAveraged over the 14 residue types (multiples of the noise floor; percentages):")
    print(df.groupby(["model", "prior", "steps"])[cols].mean().round(2).to_string())
    print("\nPer residue type, one step (mean over seeds):")
    one = df[(df.steps <= 1)].groupby(["residue", "model", "prior"])[cols].mean().round(2)
    print(one.to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
