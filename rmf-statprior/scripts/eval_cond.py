"""
Residue-type-conditioned generation of backbone dihedral angles.

For every conditional model (data_type cond4) and every residue type, generate samples conditioned on that type and
compare them with the test split of that type (the same split as in the per-type experiments). For reference, the
per-type unconditional models of the main experiments (MeanFlow, uniform or statistical prior) are evaluated on the
same test splits. Reported: unbiased MMD^2 (x1e-3) under the RMF kernel and the joint Laplace kernel (gamma = 10), and
multiples of the noise-floor standard deviation of each type.

    python scripts/eval_cond.py --csv cond_eval.csv > eval_cond.log 2>&1
"""
import argparse, glob, json, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from torch.utils.data import random_split
import eval_manifold as EM
from eval_synth import load_model
from plot_density import MAIN_EPOCHS, is_main_setting
from rmf.data.amino_datamodule import Top500, COND_CLASSES

KERNELS = [("rmf", "paper", 1.0), ("joint10", "lap", 10.0)]
BEST = {"Proline", "Pre-Pro"}   # checkpoint type used for these types in the main tables


def class_split(root, amino):
    ds = Top500(root=root, amino=amino)
    n_val = int(0.1 * len(ds)); n_test = int(0.1 * len(ds)); n_train = len(ds) - n_val - n_test
    _, _, te = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    return ds.data.float(), ds.data[torch.as_tensor(te.indices)].float()


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
    ap.add_argument("--data_dir", default="data/top500")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--floor_reps", type=int, default=10)
    ap.add_argument("--csv", default="cond_eval.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    data = {k: class_split(a.data_dir, k) for k in COND_CLASSES}
    fl = {k: floors(full, len(test), a.floor_reps, dev, gen) for k, (full, test) in data.items()}

    # runs: conditional models, and the per-type unconditional models of the main experiments
    runs = []
    for cf in sorted(glob.glob(os.path.join(a.root, "*", ".hydra", "config.yaml"))):
        run = os.path.dirname(os.path.dirname(cf)); cfg = OmegaConf.load(cf)
        dt = str(cfg.data.get("data_type", ""))
        if dt == "cond4" and os.path.exists(os.path.join(run, "checkpoints", "last.ckpt")):
            runs.append(("conditional", None, run))
        elif dt in COND_CLASSES and os.path.exists(os.path.join(run, "test_metrics.json")):
            meta = json.load(open(os.path.join(run, "test_metrics.json")))
            if str(cfg.model.get("my_method", "mf")) != "mf" or int(cfg.trainer.max_epochs) != MAIN_EPOCHS[dt]:
                continue
            if meta.get("prior_type") not in ("uniform", "gmm") or not is_main_setting(cfg, meta):
                continue
            if meta["prior_type"] == "gmm" and str(cfg.model.get("prior_coupling", "soft")) != "soft":
                continue
            runs.append(("separate", dt, run))

    rows = []
    for kind, only, run in runs:
        ck = "last" if kind == "conditional" else ("best" if only in BEST else "last")
        model, ckname, cfg = load_model(run, dev, ck)
        if model is None:
            continue
        prior = str(cfg.model.get("prior_type", "uniform"))
        for k, amino in enumerate(COND_CLASSES):
            if only is not None and amino != only:
                continue
            test = data[amino][1].to(dev)
            for steps in a.steps:
                vals = {name: [] for name, _, _ in KERNELS}
                for _ in range(a.reps):
                    with torch.no_grad():
                        if kind == "conditional":
                            model.set_condition(torch.full((a.n,), k, dtype=torch.long, device=dev))
                        x = model.sample_batch(steps, a.n).float()
                    for name, kk, g in KERNELS:
                        vals[name].append(EM.mmd2(x, test, "torus", kk, g))
                row = dict(model=kind, prior=prior, residue=amino, steps=steps, run=os.path.basename(run), ckpt=ckname,
                           seed=int(cfg.seed))
                for name, _, _ in KERNELS:
                    m = 1e3 * float(np.mean(vals[name]))
                    row[name] = m; row[name + "_x"] = m / fl[amino][name]
                rows.append(row)
                print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)

    df = pd.DataFrame(rows); df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 220)
    agg = df.groupby(["residue", "steps", "model", "prior"])[["rmf", "rmf_x", "joint10", "joint10_x"]].agg(["mean", "std", "count"])
    print("\nMMD^2 (x1e-3) and multiples of the noise-floor s.d. (_x); mean / s.d. / number of runs over seeds:")
    print(agg.round(3).to_string())
    print("\nnoise-floor s.d. (x1e-3):", {k: {kk: round(v, 4) for kk, v in d.items()} for k, d in fl.items()})
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
