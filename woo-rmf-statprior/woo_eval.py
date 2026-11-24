"""
Evaluate the samples exported by run_woo.sh and run_woo_more.sh (runs/*/samples.npz) with the kernels of the RMF
repository: unbiased MMD^2 under the RMF kernel and at the finest scale, in multiples of the noise-floor standard
deviation. Handles all nine real datasets (Fire, Earthquake, Flood, Volcano on S^2; Pre-Pro, Glycine, Proline, General
on T^2; RNA on T^7). Only the datasets that have runs are loaded. Run in the environment of the RMF repository:

    python woo_eval.py --rmf_root ../rmf-statprior --csv woo_eval.csv > eval_woo.log 2>&1
"""
import argparse, zlib, glob, importlib.util, os, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = {"sphere": [("rmf", "gauss", 1.0), ("finest", "lap", 50.0)],
           "torus": [("rmf", "paper", 1.0), ("finest", "lap", 10.0)]}



def stable_seed(*parts):
    """Seed that depends only on its arguments (dataset, run, setting), so that every noise floor and every set of
    samples is the same whichever other datasets or runs are evaluated in the same call, and in whatever order."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def points(ds):
    obj = getattr(ds, "data", ds)
    while hasattr(obj, "dset"):
        obj = obj.dset
    return torch.stack([torch.as_tensor(obj[i]) for i in range(len(obj))]).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--rmf_root", required=True)
    ap.add_argument("--floor_reps", type=int, default=20)
    ap.add_argument("--csv", default="woo_eval.csv")
    a = ap.parse_args()
    sys.path.insert(0, a.rmf_root); sys.path.insert(0, os.path.join(a.rmf_root, "scripts"))
    import eval_manifold as EM
    import eval_real as ER

    def mmd2(x, y, space, kind, g):
        return ER.mmd2_unbiased(x, y, kind, g) if space == "sphere" else EM.mmd2(x, y, space, kind, g)

    earth = load_module(os.path.join(HERE, "src/data/earth/earth_dataset.py"), "woo_earth")
    amino = load_module(os.path.join(HERE, "src/data/amino/amino_dataset.py"), "woo_amino")
    data_dir = os.path.join(HERE, "data")
    EARTH = {"fire": "fire", "earthquake": "earthquake", "flood": "flood", "volcano": "volcano"}
    AMINO = {"prepro": "Pre-Pro", "glycine": "Glycine", "proline": "Proline", "general": "General"}
    present = sorted({os.path.basename(os.path.dirname(f)).split("_")[1]
                      for f in glob.glob(os.path.join(a.runs, "*", "samples.npz"))})

    def load_splits(ds):
        if ds in EARTH:
            return {sp: points(earth.EarthDataset(os.path.join(data_dir, "earth_data"), EARTH[ds], sp)) for sp in ("train", "val", "test")}
        if ds in AMINO:
            return {sp: points(amino.Top500(os.path.join(data_dir, "top500"), AMINO[ds], sp)) for sp in ("train", "val", "test")}
        if ds == "rna":
            return {sp: points(amino.RNA(os.path.join(data_dir, "rna"), sp)) for sp in ("train", "val", "test")}
        raise ValueError(ds)
    splits = {ds: load_splits(ds) for ds in present}
    space_of = {ds: ("sphere" if ds in EARTH else "torus") for ds in present}
    display = {**{k: k.capitalize() for k in EARTH}, "prepro": "Pre-Pro", "glycine": "Glycine", "proline": "Proline",
               "general": "General", "rna": "RNA"}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    floors = {}
    for ds, sp in splits.items():
        gen = torch.Generator().manual_seed(stable_seed("floor", ds))
        space = space_of[ds]; pool = torch.cat([sp["train"], sp["val"], sp["test"]]); n = len(sp["test"])
        out = {nm: [] for nm, _, _ in KERNELS[space]}
        for _ in range(a.floor_reps):
            p = torch.randperm(len(pool), generator=gen)
            for nm, kind, g in KERNELS[space]:
                out[nm].append(mmd2(pool[p[:n]].to(dev), pool[p[n:]][:10000].to(dev), space, kind, g))
        floors[ds] = {k: float(np.std(v)) for k, v in out.items()}
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "samples.npz"))):
        run = os.path.basename(os.path.dirname(f))
        variant, ds, prior, seed = run.split("_")
        space = space_of[ds]; test = splits[ds]["test"].to(dev); smp = np.load(f)
        for steps in (1, 2, 4):
            x = torch.from_numpy(smp[f"steps{steps}"]).double().reshape(len(smp[f"steps{steps}"]), -1).to(dev)
            if space == "torus":
                x = torch.remainder(x, 2 * np.pi)
            row = dict(variant={"rmf": "Eulerian", "lmf": "Lagrangian", "smf": "semigroup"}[variant],
                       dataset=display[ds],
                       prior="statistical" if prior == "gmm" else "uniform", seed=int(seed[1:]), steps=steps)
            for nm, kind, g in KERNELS[space]:
                m = mmd2(x, test, space, kind, g)
                row[nm] = 1e3 * m; row[nm + "_x"] = m / floors[ds][nm]
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    print("\nMultiples of the noise-floor s.d. (mean over seeds; <=1: within the floor):")
    print(df.groupby(["dataset", "variant", "steps", "prior"])[["rmf_x", "finest_x"]].mean().round(2).unstack("prior").to_string())
    print(f"\nper-run results written to {a.csv}")


if __name__ == "__main__":
    main()
