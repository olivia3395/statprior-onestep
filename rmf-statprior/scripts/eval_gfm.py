"""
Statistical prior with Generalised Flow Maps: one- and four-step samples of GFM models (run directories ending in
"_gfm") trained with the uniform or the statistical prior, on Cone (SO(3)), the four Earth datasets (sphere) and the
torus data (Pre-Pro, Glycine, Proline, General, RNA).
Distances to the test split are unbiased MMD^2 under the RMF kernel and at the finest scale, in multiples of the
noise-floor standard deviation, as in the comparison of objectives of the paper.

    python scripts/eval_gfm.py --csv gfm_eval.csv > eval_gfm.log 2>&1
"""
import argparse, zlib, glob, json, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
import eval_manifold as EM
import eval_real as ER
from eval_synth import load_model

SPACE = {"fire": "sphere", "earthquake": "sphere", "flood": "sphere", "volcano": "sphere",
         "Pre-Pro": "torus", "Glycine": "torus", "Proline": "torus", "General": "torus", "RNA": "torus",
         "cone": "so3"}
KERNELS = {"sphere": [("rmf", "gauss", 1.0), ("finest", "lap", 50.0)],
           "torus": [("rmf", "paper", 1.0), ("finest", "lap", 10.0)],
           "so3": [("rmf", "paper", 1.0), ("finest", "lap", 20.0)]}



def stable_seed(*parts):
    """Seed that depends only on its arguments (dataset, run, setting), so that every noise floor and every set of
    samples is the same whichever other datasets or runs are evaluated in the same call, and in whatever order."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF

def mmd2(x, y, space, kind, gamma):
    if space == "sphere":
        return ER.mmd2_unbiased(x, y, kind, gamma)
    return EM.mmd2(x, y, space, kind, gamma)


def splits(cfg, space):
    return ER.split_tensors(cfg) if space == "sphere" else EM.split_tensors(cfg)


def floors(pool, n_test, space, dev, reps, gen):
    out = {name: [] for name, _, _ in KERNELS[space]}
    for _ in range(reps):
        p = torch.randperm(len(pool), generator=gen)
        a, b = pool[p[:n_test]].to(dev), pool[p[n_test:]][:10000].to(dev)
        for name, kind, g in KERNELS[space]:
            out[name].append(mmd2(a, b, space, kind, g))
    return {k: float(np.std(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--floor_reps", type=int, default=20)
    ap.add_argument("--csv", default="gfm_eval.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0); torch.manual_seed(0)
    rows, floor_cache = [], {}
    for run in sorted(glob.glob(os.path.join(a.root, "*_gfm"))):
        if not os.path.exists(os.path.join(run, "checkpoints", "last.ckpt")) \
                or not os.path.exists(os.path.join(run, "test_metrics.json")):     # skip interrupted runs
            continue
        model, ck, cfg = load_model(run, dev, "last")
        if model is None:
            continue
        ds = str(cfg.data.get("dataset_file", None) or cfg.data.get("data_type", ""))
        space = SPACE.get(ds)
        if space is None:
            continue
        train, val, test = splits(cfg, space)
        if ds not in floor_cache:
            pool = torch.cat([train, val, test]).float()
            floor_cache[ds] = floors(pool, len(test), space, dev, a.floor_reps,
                                     torch.Generator().manual_seed(stable_seed("floor", ds)))
        fl = floor_cache[ds]
        test = test.to(dev)
        prior = str(cfg.model.get("prior_type", "uniform"))
        loss = str(cfg.model.get("underlying_loss", "lsd"))
        for steps in a.steps:
            vals = {name: [] for name, _, _ in KERNELS[space]}
            torch.manual_seed(stable_seed(os.path.basename(run), steps))
            for _ in range(a.reps):
                with torch.no_grad():
                    x = model.sample_batch(steps, a.n).float()
                for name, kind, g in KERNELS[space]:
                    vals[name].append(mmd2(x, test, space, kind, g))
            row = dict(dataset=ds, loss=f"G-{loss.upper()}", prior=prior, steps=steps, seed=int(cfg.seed),
                       run=os.path.basename(run))
            for name in vals:
                m = float(np.mean(vals[name]))
                row[name] = 1e3 * m; row[name + "_x"] = m / fl[name]
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    print("\nMultiples of the noise-floor s.d. (mean over seeds; <=1: within the floor):")
    print(df.groupby(["dataset", "loss", "steps", "prior"])[["rmf_x", "finest_x"]].mean().round(2).unstack("prior").to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
