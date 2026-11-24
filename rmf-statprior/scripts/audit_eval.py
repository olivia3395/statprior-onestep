"""
Audit of the MMD evaluation protocol used in Riemannian one-step generation papers
(RMF, GFM, RCM ...), reproducing the exact estimator of the RMF code base:
    V-statistic MMD, geodesic Gaussian kernel exp(-gamma d^2) with gamma = 1, then sqrt;
    on the flat torus the distance is returned PER ANGLE, so the kernel is the average of
    1-D kernels (an additive kernel).

For every benchmark in the paper (Earth x4, Torus x5, SO(3) x3) it reports:
  1. Noise floor: the MMD between two DISJOINT subsets of REAL data of the paper's test-set
     size. A perfect generative model scores about this; differences between methods smaller
     than its spread cannot be distinguished.
  2. Which reported methods are within the noise floor.
  3. Kernel validity: negative eigenvalues of the Gram matrix (a valid MMD needs a PSD kernel).
  4. Torus only: a "shuffled" control that keeps every 1-D marginal but destroys the joint
     (phi, psi) structure. If the paper's metric cannot tell it apart from real data, the metric
     is blind to the Ramachandran structure. A joint geodesic kernel is shown for comparison.

CPU only. Run from the rmf/ directory:
    python scripts/audit_eval.py                  # everything (~10-30 min on CPU)
    python scripts/audit_eval.py --quick          # fewer repetitions, subsampled large sets
Writes audit_eval.csv and prints a summary table.
"""
import argparse
import csv
import math
import os

import numpy as np
import pandas as pd

# Reported one-step MMD values (Tables 1-3 of Zhong et al., ICML 2026)
PAPER = {
    "volcano": dict(RFM=0.351, RCT=0.155, G_LSD=0.115, RMF=0.092, RMF_MT=0.102),
    "earthquake": dict(RFM=0.309, RCT=0.053, G_LSD=0.032, RMF=0.042, RMF_MT=0.035),
    "flood": dict(RFM=0.272, RCT=0.086, G_LSD=0.065, RMF=0.068, RMF_MT=0.048),
    "fire": dict(RFM=0.377, RCT=0.080, G_LSD=0.027, RMF=0.042, RMF_MT=0.032),
    "General": dict(RFM=0.45, RCT=0.01, G_LSD=0.02, RMF=0.11, RMF_MT=0.04),
    "Glycine": dict(RFM=0.27, RCT=0.04, G_LSD=0.03, RMF=0.04, RMF_MT=0.03),
    "Proline": dict(RFM=0.52, RCT=0.05, G_LSD=0.04, RMF=0.09, RMF_MT=0.04),
    "Pre-Pro": dict(RFM=0.47, RCT=0.06, G_LSD=0.05, RMF=0.09, RMF_MT=0.05),
    "RNA": dict(RFM=0.68, RCT=0.11, G_LSD=0.08, RMF=0.20, RMF_MT=0.07),
    "cone": dict(RFM=0.311, RCT=0.096, G_LSD=0.044, RMF=0.057, RMF_MT=0.049),
    "fisher24": dict(RFM=0.154, RCT=0.071, G_LSD=0.073, RMF=0.076, RMF_MT=0.039),
    "line": dict(RFM=0.095, RCT=0.051, G_LSD=0.037, RMF=0.038, RMF_MT=0.035),
}
SO3_SIZE = {"cone": 20000, "fisher24": 40000, "line": 40000}  # dataset sizes used in the paper


# ------------------------------------------------------------------- loaders
def load_earth(name, root):
    rows = np.array(list(csv.reader(open(os.path.join(root, "earth_data", f"{name}.csv"))))[1:], dtype=float)
    lat, lon = np.radians(rows[:, 0]), np.radians(rows[:, 1])
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)


def load_protein(label, root):
    df = pd.read_csv(os.path.join(root, "top500", "aggregated_angles.tsv"), sep="\t", header=None)
    ang = df[df[3] == label][[1, 2]].to_numpy(float)
    return (ang % 360) * math.pi / 180


def load_rna(root):
    df = pd.read_csv(os.path.join(root, "rna", "aggregated_angles.tsv"), sep="\t", header=None)
    return (df.iloc[:, 2:9].to_numpy(float) % 360) * math.pi / 180


def load_so3(name, root):
    return np.load(os.path.join(root, "raw", f"{name}_train.npy"))[: SO3_SIZE[name]].astype(np.float64)


# ----------------------------------------------------------------- distances
def dist_sphere(a, b):
    return np.arccos(np.clip(a @ b.T, -1, 1))


def dist_so3(a, b):
    tr = np.einsum("iab,jab->ij", a, b)  # trace(A_i^T B_j)
    return np.arccos(np.clip((tr - 1) / 2, -1, 1))


def dist_torus_per_angle(a, b):
    d = np.abs(a[:, None, :] - b[None, :, :])
    return np.minimum(d, 2 * math.pi - d)  # (n, m, dim) -- as in the RMF code


# ----------------------------------------------------------------------- MMD
def kmean(a, b, space, kernel, gamma=1.0, block=1500):
    """Mean of k(a_i, b_j) over all pairs (V-statistic), blockwise."""
    tot = 0.0
    for i in range(0, len(a), block):
        ai = a[i : i + block]
        if space == "sphere":
            d = dist_sphere(ai, b)
        elif space == "so3":
            d = dist_so3(ai, b)
        else:
            d = dist_torus_per_angle(ai, b)
            if kernel == "joint":  # proper geodesic distance on the flat torus
                d = np.sqrt((d**2).sum(-1))
        k = np.exp(-gamma * d**2)
        tot += k.mean(axis=-1).sum() if k.ndim == 3 else k.sum()  # additive kernel: average over angles
    return tot / (len(a) * len(b))


def mmd_paper(x, y, space, kernel="paper"):
    v = kmean(x, x, space, kernel) + kmean(y, y, space, kernel) - 2 * kmean(x, y, space, kernel)
    return math.sqrt(max(v, 0.0))


def min_eig(x, space, n=1500, rng=None):
    x = x[rng.choice(len(x), min(n, len(x)), replace=False)]
    if space == "sphere":
        K = np.exp(-dist_sphere(x, x) ** 2)
    elif space == "so3":
        K = np.exp(-dist_so3(x, x) ** 2)
    else:
        K = np.exp(-dist_torus_per_angle(x, x) ** 2).mean(-1)
    ev = np.linalg.eigvalsh(K)
    return ev.min(), int((ev < -1e-10).sum())


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="5 reps, test sets capped at 3000 points")
    ap.add_argument("--csv", default="audit_eval.csv")
    a = ap.parse_args()
    reps = 5 if a.quick else a.reps
    cap = 3000 if a.quick else None
    rng = np.random.default_rng(0)

    datasets = [(n, "sphere", lambda n=n: load_earth(n, a.root)) for n in ["volcano", "earthquake", "flood", "fire"]]
    datasets += [(l, "torus", lambda l=l: load_protein(l, a.root)) for l in ["General", "Glycine", "Proline", "Pre-Pro"]]
    datasets += [("RNA", "torus", lambda: load_rna(a.root))]
    datasets += [(n, "so3", lambda n=n: load_so3(n, a.root)) for n in ["cone", "fisher24", "line"]]

    rows = []
    for name, space, loader in datasets:
        x = loader()
        n_test = int(0.1 * len(x))
        n_eval = min(n_test, cap) if cap else n_test
        floor = []
        for _ in range(reps):
            idx = rng.permutation(len(x))
            floor.append(mmd_paper(x[idx[:n_eval]], x[idx[n_eval : 2 * n_eval]], space))
        f_mean, f_std = float(np.mean(floor)), float(np.std(floor))
        me, nneg = min_eig(x, space, rng=rng)
        paper = PAPER[name]
        within = [m for m, v in paper.items() if v <= f_mean + 2 * f_std]
        row = dict(dataset=name, space=space, n_data=len(x), n_test=n_test, n_eval=n_eval,
                   floor_mean=round(f_mean, 4), floor_std=round(f_std, 4),
                   best_reported=min(paper.values()),
                   within_floor=", ".join(within) if within else "-",
                   gram_min_eig=f"{me:.1e}", gram_n_neg=nneg)

        if space == "torus":  # shuffle control: keep 1-D marginals, destroy joint structure
            shuf_paper, shuf_joint, real_joint = [], [], []
            for _ in range(reps):
                idx = rng.permutation(len(x))
                A, B = x[idx[:n_eval]], x[idx[n_eval : 2 * n_eval]].copy()
                for j in range(1, B.shape[1]):
                    B[:, j] = B[rng.permutation(len(B)), j]
                shuf_paper.append(mmd_paper(A, B, space))
                shuf_joint.append(mmd_paper(A, B, space, kernel="joint"))
                real_joint.append(mmd_paper(A, x[idx[n_eval : 2 * n_eval]], space, kernel="joint"))
            row["shuffled_paper_metric"] = round(float(np.mean(shuf_paper)), 4)
            row["shuffled_joint_metric"] = round(float(np.mean(shuf_joint)), 4)
            row["real_joint_metric"] = round(float(np.mean(real_joint)), 4)
        rows.append(row)
        print(f"[audit] {name:10s} n_test={n_test:6d} floor={f_mean:.3f}+-{f_std:.3f} "
              f"within floor: {row['within_floor']}"
              + (f" | shuffled: paper metric {row['shuffled_paper_metric']:.3f}, joint metric "
                 f"{row['shuffled_joint_metric']:.3f} (real {row['real_joint_metric']:.3f})" if space == "torus" else ""))

    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + df.to_string(index=False))
    print(f"\nwritten to {a.csv}")


if __name__ == "__main__":
    main()
