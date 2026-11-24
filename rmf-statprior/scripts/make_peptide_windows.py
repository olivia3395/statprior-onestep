"""
Real higher-dimensional torus data: backbone torsions (phi_1, psi_1, ..., phi_k, psi_k) of k consecutive
residues of the same chain in the Top500 set, i.e. points on T^{2k}.

Windows are NON-overlapping (stride k) so that no residue appears in two windows, which would otherwise
leak information between the random train/test splits. Only runs of consecutive residue numbers within a
chain are used.

    python scripts/make_peptide_windows.py --ks 2 3 5      # writes data/torus_windows/pepwin_k{k}.npy
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd


def load(path="data/top500/aggregated_angles.tsv"):
    df = pd.read_csv(path, sep="\t", header=None, names=["source", "phi", "psi", "amino"])
    parts = df["source"].str.extract(r"^(?P<prot>[^:]+):(?P<chain>[^:]+):(?P<res>[A-Za-z]+)(?P<num>-?\d+)$")
    df = pd.concat([df, parts], axis=1).dropna(subset=["num"])
    df["num"] = df["num"].astype(int)
    df = df.drop_duplicates(subset=["prot", "chain", "num"])
    return df.sort_values(["prot", "chain", "num"]).reset_index(drop=True)


def windows(df, k):
    out = []
    for (_, _), g in df.groupby(["prot", "chain"], sort=False):
        num = g["num"].to_numpy()
        ang = np.radians(g[["phi", "psi"]].to_numpy()) % (2 * math.pi)
        # split the chain into runs of consecutive residue numbers
        breaks = np.where(np.diff(num) != 1)[0] + 1
        for run in np.split(np.arange(len(num)), breaks):
            for s in range(0, len(run) - k + 1, k):          # non-overlapping windows
                out.append(ang[run[s:s + k]].reshape(-1))
    return np.asarray(out, dtype=np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--out", default="data/torus_windows")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load()
    print(f"{len(df)} residues in {df.groupby(['prot', 'chain']).ngroups} chains")
    for k in a.ks:
        X = windows(df, k)
        np.save(os.path.join(a.out, f"pepwin_k{k}.npy"), X)
        print(f"pepwin_k{k}: {len(X)} windows on T^{2 * k}")
