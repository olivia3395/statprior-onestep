"""
Statistical prior for the JAX code of Woo et al. (Riemannian MeanFlow, ICML 2026).

The tangent-space Gaussian mixture of stat_prior.py is fitted with PyTorch on the CPU, once, on the training set.
Noise is drawn outside the jitted training step, so the prior and the posterior coupling run in PyTorch and are
returned as JAX arrays. StatPrior has the interface of UniformPrior (sample, log_prob) and in addition couple(x_1),
which draws the noise for a batch of data with the posterior coupling.
"""
import numpy as np
import torch
import jax.numpy as jnp

from .prior import UniformPrior
from .stat_prior import SphereTangentGMM, TorusTangentGMM


def _base_points(dataset):
    """Training points without the repetition of ExpandDataset (Earth and Top500 datasets of this repository)."""
    obj = getattr(dataset, "data", dataset)
    while hasattr(obj, "dset"):
        obj = obj.dset
    return torch.stack([torch.as_tensor(obj[i]) for i in range(len(obj))]).float()


class StatPrior:
    def __init__(self, kind, dim, points, max_k=32, cov_scale=2.0, coupling="soft", seed=0, max_points=10000):
        cands = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64) if k <= max_k]
        kw = dict(max_k=max_k, k_candidates=cands, select="bic", cov_scale=cov_scale, coupling=coupling)
        self.gmm = SphereTangentGMM(**kw) if kind == "sphere" else TorusTangentGMM(dim=dim, **kw)
        x = points.reshape(len(points), -1)
        if len(x) > max_points:
            g = torch.Generator().manual_seed(seed)
            x = x[torch.randperm(len(x), generator=g)[:max_points]]
        self.gmm.fit(x, seed=seed)
        torch.manual_seed(seed)
        sel = ", ".join(f"K={k}:{c:.0f}" for k, c in self.gmm.selection_log)
        print(f"[statistical prior, Woo RMF] {kind}: fitted on {len(x)} points, selected K={int(self.gmm.n_active)} ({sel})",
              flush=True)

    def sample(self, batch_size, *xshape):
        with torch.no_grad():
            y = self.gmm.sample(batch_size).float().numpy()
        return jnp.asarray(y.reshape(batch_size, *xshape))

    def couple(self, x_1):
        x = torch.from_numpy(np.asarray(x_1, dtype=np.float32))
        with torch.no_grad():
            y = self.gmm.sample_matched(x.reshape(len(x), -1)).float().numpy()
        return jnp.asarray(y.reshape(np.shape(x_1)))

    def log_prob(self, x, *xshape):
        with torch.no_grad():
            lp = self.gmm.log_prob(torch.from_numpy(np.asarray(x, dtype=np.float64)).reshape(len(x), -1))
        return jnp.asarray(lp.numpy())


def make_prior(cfg, manifold, seed, train_dataset):
    """UniformPrior (default) or StatPrior when cfg.prior_type == 'gmm'."""
    if str(cfg.get("prior_type", "uniform")) != "gmm":
        return UniformPrior(manifold, seed)
    kind = "sphere" if "sphere" in type(manifold).__name__.lower() else "torus"
    pts = _base_points(train_dataset)
    return StatPrior(kind, int(pts.reshape(len(pts), -1).shape[1]), pts, max_k=int(cfg.get("prior_k", 32)),
                     cov_scale=float(cfg.get("prior_cov_scale", 2.0)), coupling=str(cfg.get("prior_coupling", "soft")),
                     seed=int(cfg.get("seed", 0)))
