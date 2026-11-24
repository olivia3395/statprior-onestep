"""
Synthetic distributions on the sphere S^2 with known densities, used both to create
datasets (make_synth_shapes.py) and to evaluate trained models (eval_synth.py).

Dataset names (all start with "synth" so EarthDataModule loads them):
  synth_k{kappa}     mixture of 5 vMF blobs, concentration kappa          (make_synth_sphere.py)
  synth_ring{s}      mixture of 3 thin rings (small circles), width s mrad
  synth_arc{s}       mixture of 4 elongated bands along great circles, width s mrad

Rings and arcs are deliberately NOT vMF-shaped, so a vMF-mixture prior is misspecified.
"""
import math
import re

import numpy as np


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _frame(a):
    """Orthonormal e1, e2 perpendicular to unit vector a."""
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _normalize(t - (t @ a) * a)
    return e1, np.cross(a, e1)


def _log_normal(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - math.log(s * math.sqrt(2 * math.pi))


def _logsumexp(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


# ----------------------------------------------------------------------- vMF
def vmf_params(components=5, seed=0):
    """Mirrors scripts/make_synth_sphere.py (defaults)."""
    rng = np.random.default_rng(seed)
    mus = _normalize(rng.normal(size=(components, 3)))
    weights = rng.dirichlet(np.ones(components) * 2)
    return mus, weights


def vmf_sample(kappa, n, rng):
    mus, weights = vmf_params()
    comp = rng.choice(len(weights), size=n, p=weights)
    u = rng.uniform(1e-12, 1.0, n)
    w = 1 + np.log(u + (1 - u) * np.exp(-2 * kappa)) / kappa
    mu = mus[comp]
    v = rng.normal(size=(n, 3))
    v = _normalize(v - (v * mu).sum(1, keepdims=True) * mu)
    return w[:, None] * mu + np.sqrt(np.clip(1 - w**2, 0, None))[:, None] * v


def vmf_logpdf(kappa, x):
    mus, weights = vmf_params()
    logc = math.log(kappa) - math.log(2 * math.pi) - kappa - math.log1p(-math.exp(-2 * kappa))
    return _logsumexp(np.log(weights)[None] + logc + kappa * (x @ mus.T))


# ---------------------------------------------------------------------- rings
def ring_params(seed=1):
    rng = np.random.default_rng(seed)
    axes = _normalize(rng.normal(size=(3, 3)))
    theta0 = np.array([0.5, 0.9, 1.3])  # opening angles of the small circles (rad)
    weights = rng.dirichlet(np.ones(3) * 3)
    return axes, theta0, weights


def ring_sample(sigma, n, rng):
    axes, theta0, weights = ring_params()
    comp = rng.choice(3, size=n, p=weights)
    out = np.empty((n, 3))
    for c in range(3):
        idx = np.where(comp == c)[0]
        m = len(idx)
        theta = np.empty(0)
        while len(theta) < m:  # truncate to (0, pi); negligible for the widths used
            t = rng.normal(theta0[c], sigma, size=2 * m)
            theta = np.concatenate([theta, t[(t > 0) & (t < math.pi)]])
        theta = theta[:m]
        phi = rng.uniform(0, 2 * math.pi, m)
        e1, e2 = _frame(axes[c])
        out[idx] = (np.cos(theta)[:, None] * axes[c]
                    + np.sin(theta)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
    return out


def ring_logpdf(sigma, x):
    axes, theta0, weights = ring_params()
    theta = np.arccos(np.clip(x @ axes.T, -1, 1))  # (n, 3)
    sin_t = np.clip(np.sin(theta), 1e-12, None)
    comp = np.log(weights)[None] + _log_normal(theta, theta0[None], sigma) - np.log(2 * math.pi * sin_t)
    return _logsumexp(comp)


# ----------------------------------------------------------------------- arcs
def arc_params(seed=2):
    """Each arc: a great circle (pole n), a centre angle phi_c, and a spread along the circle."""
    rng = np.random.default_rng(seed)
    poles = _normalize(rng.normal(size=(4, 3)))
    phi_c = rng.uniform(0, 2 * math.pi, 4)
    along = np.array([0.35, 0.45, 0.55, 0.6])  # std of position along the circle (rad): long, thin bands
    weights = rng.dirichlet(np.ones(4) * 3)
    return poles, phi_c, along, weights


def arc_sample(sigma, n, rng):
    poles, phi_c, along, weights = arc_params()
    comp = rng.choice(4, size=n, p=weights)
    out = np.empty((n, 3))
    for c in range(4):
        idx = np.where(comp == c)[0]
        m = len(idx)
        e1, e2 = _frame(poles[c])
        phi = phi_c[c] + rng.normal(0, along[c], m)
        delta = rng.normal(0, sigma, m)
        circ = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
        out[idx] = np.cos(delta)[:, None] * circ + np.sin(delta)[:, None] * poles[c]
    return out


def arc_logpdf(sigma, x):
    poles, phi_c, along, weights = arc_params()
    comps = []
    for c in range(4):
        e1, e2 = _frame(poles[c])
        delta = np.arcsin(np.clip(x @ poles[c], -1, 1))
        phi = np.arctan2(x @ e2, x @ e1)
        d = (phi - phi_c[c] + math.pi) % (2 * math.pi) - math.pi  # wrapped offset along the circle
        lp_phi = _logsumexp(np.stack([_log_normal(d + 2 * math.pi * k, 0, along[c]) for k in (-1, 0, 1)], 1))
        lp = lp_phi + _log_normal(delta, 0, sigma) - np.log(np.clip(np.cos(delta), 1e-12, None))
        comps.append(math.log(weights[c]) + lp)
    return _logsumexp(np.stack(comps, 1))


# ------------------------------------------------------------------ dispatch
def parse(name):
    """'synth_k100' -> ('vmf', 100.0); 'synth_ring50' -> ('ring', 0.05); 'synth_arc25' -> ('arc', 0.025)."""
    m = re.fullmatch(r"synth_k(\d+)", name)
    if m:
        return "vmf", float(m.group(1))
    m = re.fullmatch(r"synth_(ring|arc)(\d+)", name)
    if m:
        return m.group(1), float(m.group(2)) / 1000.0
    return None, None


def sample(name, n, rng):
    kind, p = parse(name)
    return {"vmf": vmf_sample, "ring": ring_sample, "arc": arc_sample}[kind](p, n, rng)


def logpdf(name, x):
    kind, p = parse(name)
    return {"vmf": vmf_logpdf, "ring": ring_logpdf, "arc": arc_logpdf}[kind](p, x)
