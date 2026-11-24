"""
Synthetic mixtures on the flat torus T^d with exact densities, for the dimension-scaling study.

Each of K components is a correlated Gaussian in normal coordinates v ~ N(0, Sigma_k), bent into a
"banana" by the triangular map
    y_1 = v_1,   y_i = v_i + a * sin(v_{i-1})   (i >= 2),
whose Jacobian determinant is exactly 1, and placed at a centre mu_k: x = (mu_k + y) mod 2 pi. Samples
with any |y_i| >= pi are rejected so that the map to the torus is one-to-one. The components are therefore
NOT Gaussian (the statistical prior is misspecified), but the density is still known in closed form:
    p(x) = sum_k w_k N(v(y); 0, Sigma_k) / Z_k,   y = wrap(x - mu_k),  v = inverse of the bend.

Dataset names: tsynth_d{d}  (e.g. tsynth_d10).

    python scripts/torus_synth.py --dims 2 5 10 20 50      # writes data/torus_synth/tsynth_d*.npy
"""
import argparse
import math
import os
import re

import numpy as np

K = 8
BEND = 0.6
SIG_RANGE = (0.15, 0.45)   # per-axis standard deviations (rad) before rotation


def _wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def params(d, seed=1234):
    rng = np.random.default_rng(seed + d)
    mus = rng.uniform(0, 2 * math.pi, size=(K, d))
    covs = []
    for _ in range(K):
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        sig = rng.uniform(*SIG_RANGE, size=d)
        covs.append(Q @ np.diag(sig**2) @ Q.T)
    w = rng.dirichlet(np.ones(K) * 2)
    return mus, np.stack(covs), w


def _bend(v):
    y = v.copy()
    y[:, 1:] = v[:, 1:] + BEND * np.sin(v[:, :-1])
    return y


def _unbend(y):
    v = np.empty_like(y)
    v[:, 0] = y[:, 0]
    for i in range(1, y.shape[1]):
        v[:, i] = y[:, i] - BEND * np.sin(v[:, i - 1])
    return v


def _sample_comp(cov, n, rng):
    L = np.linalg.cholesky(cov)
    out, total = [], 0
    while total < n:
        v = rng.normal(size=(2 * n, cov.shape[0])) @ L.T
        y = _bend(v)
        y = y[(np.abs(y) < math.pi).all(1)]
        out.append(y); total += len(y)
    return np.concatenate(out)[:n]


def _log_Z(cov, rng, n_mc=200000):
    L = np.linalg.cholesky(cov)
    y = _bend(rng.normal(size=(n_mc, cov.shape[0])) @ L.T)
    return math.log(max((np.abs(y) < math.pi).all(1).mean(), 1e-12))


_CACHE = {}


def _model(d):
    if d not in _CACHE:
        mus, covs, w = params(d)
        rng = np.random.default_rng(99)
        logZ = np.array([_log_Z(c, rng) for c in covs])
        Linv = np.stack([np.linalg.inv(np.linalg.cholesky(c)) for c in covs])
        logdet = np.array([np.linalg.slogdet(c)[1] for c in covs])
        _CACHE[d] = (mus, covs, w, logZ, Linv, logdet)
    return _CACHE[d]


def dim_of(name):
    m = re.fullmatch(r"tsynth_d(\d+)", name)
    return int(m.group(1)) if m else None


def sample(name, n, rng):
    d = dim_of(name)
    mus, covs, w, *_ = _model(d)
    comp = rng.choice(K, size=n, p=w)
    x = np.empty((n, d))
    for k in range(K):
        idx = np.where(comp == k)[0]
        if len(idx):
            x[idx] = (mus[k] + _sample_comp(covs[k], len(idx), rng)) % (2 * math.pi)
    return x


def logpdf(name, x):
    d = dim_of(name)
    mus, covs, w, logZ, Linv, logdet = _model(d)
    comps = []
    for k in range(K):
        y = _wrap(x - mus[k])
        v = _unbend(y)
        z = v @ Linv[k].T
        lp = -0.5 * (z**2).sum(1) - 0.5 * d * math.log(2 * math.pi) - 0.5 * logdet[k] - logZ[k] + math.log(w[k])
        lp[~(np.abs(y) < math.pi).all(1)] = -np.inf
        comps.append(lp)
    comps = np.stack(comps, 1)
    m = comps.max(1, keepdims=True)
    return (m + np.log(np.exp(comps - m).sum(1, keepdims=True))).squeeze(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 50])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="data/torus_synth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for d in a.dims:
        name = f"tsynth_d{d}"
        x = sample(name, a.n, np.random.default_rng(d))
        np.save(os.path.join(a.out, name + ".npy"), x.astype(np.float32))
        ll = logpdf(name, x).mean()
        print(f"{name}: n={a.n}  E[log p]={ll:.3f}  KL(p||uniform)={ll + d * math.log(2 * math.pi):.2f} nats "
              f"({(ll + d * math.log(2 * math.pi)) / d:.2f} per dim)")
