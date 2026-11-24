"""
Synthetic sphere datasets with controlled concentration, for the key pilot
experiment: does the benefit of the structured prior grow as data gets more
concentrated?

Each dataset is a mixture of M vMF components with shared concentration kappa
(same mean directions for all kappa), written as LATITUDE,LONGITUDE csv files
into data/earth_data/synth_k{kappa}.csv so the existing EarthDataModule loads them
with `data.dataset_file=synth_k{kappa}`.
"""
import argparse, math, os
import numpy as np


def sample_vmf_s2(mu, kappa, n, rng):
    u = rng.uniform(1e-12, 1.0, n)
    w = 1 + np.log(u + (1 - u) * np.exp(-2 * kappa)) / kappa
    v = rng.normal(size=(n, 3))
    v -= (v @ mu)[:, None] * mu
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return w[:, None] * mu + np.sqrt(np.clip(1 - w**2, 0, None))[:, None] * v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", type=float, nargs="+", default=[2, 5, 10, 25, 50, 100, 250])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--components", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/earth_data")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    mus = rng.normal(size=(a.components, 3))
    mus /= np.linalg.norm(mus, axis=1, keepdims=True)
    weights = rng.dirichlet(np.ones(a.components) * 2)
    for kappa in a.kappas:
        comp = rng.choice(a.components, size=a.n, p=weights)
        x = np.concatenate([sample_vmf_s2(mus[c], kappa, int((comp == c).sum()), rng) for c in range(a.components)])
        lat = np.degrees(np.arcsin(np.clip(x[:, 2], -1, 1)))
        lon = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
        path = os.path.join(a.out, f"synth_k{int(kappa)}.csv")
        np.savetxt(path, np.stack([lat, lon], 1), delimiter=",", header="LATITUDE,LONGITUDE", comments="", fmt="%.6f")
        print(f"wrote {path}: n={a.n}, kappa={kappa}")
