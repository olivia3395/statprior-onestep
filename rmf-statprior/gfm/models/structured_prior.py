"""
Structured prior for one-step generation on the sphere S^{d-1}.

Motivation: on a compact manifold the default uniform prior has no tails, so the
hard part of the one-step map cannot be "hidden" far away (as it can be under a
Gaussian prior in R^d). We replace the uniform prior by a mixture of von
Mises-Fisher (vMF) distributions fitted to the training data, and pair each data
point with noise drawn from the vMF component of its own cluster
("cluster-matched coupling"). The coupling is exact at the population level:
if component weights equal the cluster proportions, the noise marginal equals
the mixture prior used at sampling time. No OT problem is solved.

Usage (see MeanFlowModule):
    prior = SphereClusterPrior(dim=3, n_clusters=16, kappa_scale=0.5)
    prior.fit(train_points)                 # (N, dim) unit vectors
    eps = prior.sample_matched(x)           # training: noise paired with x
    z0 = prior.sample(n)                    # generation: fresh noise
    logp = prior.log_prob(z)                # NLL evaluation (dim == 3 only)
"""

import math

import torch
from torch import Tensor, nn


def _log_c3(kappa: Tensor) -> Tensor:
    """log normaliser of vMF on S^2: kappa / (4 pi sinh kappa), numerically stable."""
    kappa = kappa.clamp(min=1e-6)
    return torch.log(kappa) - math.log(2 * math.pi) - kappa - torch.log1p(-torch.exp(-2 * kappa))


class SphereClusterPrior(nn.Module):
    """
    Mixture-of-vMF prior on S^{dim-1} fitted by spherical k-means.

    :param dim: ambient dimension (points are unit vectors in R^dim).
    :param n_clusters: number of mixture components K.
    :param kappa_scale: multiplies the fitted concentrations. < 1 widens the
        prior (safer coverage), 1 matches cluster spread, 0 recovers uniform.
    :param kappa_max: upper bound on concentrations (numerical safety).
    :param n_iter: k-means iterations.
    """

    def __init__(
        self,
        dim: int,
        n_clusters: int = 16,
        kappa_scale: float = 0.5,
        kappa_max: float = 200.0,
        n_iter: int = 50,
    ):
        super().__init__()
        self.dim = dim
        self.n_clusters = n_clusters
        self.kappa_scale = kappa_scale
        self.kappa_max = kappa_max
        self.n_iter = n_iter
        # buffers have fixed shapes so checkpoints load before/after fitting
        self.register_buffer("mu", torch.zeros(n_clusters, dim))
        self.register_buffer("kappa", torch.zeros(n_clusters))
        self.register_buffer("log_weights", torch.full((n_clusters,), -math.log(n_clusters)))
        self.register_buffer("fitted", torch.tensor(False))

    # ------------------------------------------------------------------ fitting
    @torch.no_grad()
    def fit(self, x: Tensor, seed: int = 0) -> "SphereClusterPrior":
        x = torch.nn.functional.normalize(x.double(), dim=-1)
        n, d = x.shape
        assert d == self.dim, f"expected dim {self.dim}, got {d}"
        k = min(self.n_clusters, n)
        g = torch.Generator().manual_seed(seed)

        # k-means++ initialisation with cosine distance
        centers = [x[torch.randint(n, (1,), generator=g)].squeeze(0)]
        for _ in range(1, k):
            c = torch.stack(centers)
            dist = (1 - x @ c.T).min(dim=1).values.clamp(min=0)
            probs = dist / dist.sum() if dist.sum() > 0 else torch.full((n,), 1.0 / n, dtype=x.dtype)
            centers.append(x[torch.multinomial(probs, 1, generator=g)].squeeze(0))
        c = torch.stack(centers)

        for _ in range(self.n_iter):
            labels = (x @ c.T).argmax(dim=1)
            new_c = torch.zeros_like(c).index_add_(0, labels, x)
            empty = new_c.norm(dim=1) < 1e-12
            if empty.any():  # re-seed empty clusters at random points
                new_c[empty] = x[torch.randint(n, (int(empty.sum()),), generator=g)]
            new_c = torch.nn.functional.normalize(new_c, dim=-1)
            if torch.allclose(new_c, c, atol=1e-10):
                c = new_c
                break
            c = new_c
        labels = (x @ c.T).argmax(dim=1)

        counts = torch.bincount(labels, minlength=k).double()
        sums = torch.zeros_like(c).index_add_(0, labels, x)
        rbar = (sums.norm(dim=1) / counts.clamp(min=1)).clamp(max=1 - 1e-8)
        # Banerjee et al. (2005) approximation of the vMF concentration MLE
        kappa = rbar * (d - rbar**2) / (1 - rbar**2)
        kappa = torch.where(counts > 1, kappa, torch.zeros_like(kappa))  # singletons -> uniform
        kappa = (self.kappa_scale * kappa).clamp(min=0.0, max=self.kappa_max)

        mu = torch.zeros(self.n_clusters, d, dtype=x.dtype)
        kap = torch.zeros(self.n_clusters, dtype=x.dtype)
        logw = torch.full((self.n_clusters,), -float("inf"), dtype=x.dtype)
        mu[:k], kap[:k] = c, kappa
        logw[:k] = torch.log(counts / counts.sum())
        logw = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -1e30))

        self.mu.copy_(mu.float())
        self.kappa.copy_(kap.float())
        self.log_weights.copy_(logw.float())
        self.fitted.fill_(True)
        return self

    # ----------------------------------------------------------------- sampling
    def assign(self, x: Tensor) -> Tensor:
        """Cluster index of each point (nearest mean direction)."""
        sim = x.reshape(-1, self.dim) @ self.mu.T
        sim = sim.masked_fill(self.log_weights < -1e29, -float("inf"))
        return sim.argmax(dim=1)

    def _sample_vmf(self, mu: Tensor, kappa: Tensor) -> Tensor:
        """One vMF sample per row of (mu, kappa). Wood (1994); exact inverse CDF for dim 3."""
        n, d = mu.shape
        device, dtype = mu.device, mu.dtype
        kappa = kappa.to(dtype)
        if d == 3:
            u = torch.rand(n, device=device, dtype=dtype).clamp(min=1e-12)
            k = kappa.clamp(min=1e-6)
            w = 1 + torch.log(u + (1 - u) * torch.exp(-2 * k)) / k
            w = torch.where(kappa < 1e-6, 2 * u - 1, w)  # uniform limit
        else:
            w = self._wood_w(kappa, d)
        w = w.clamp(-1, 1)
        # uniform direction orthogonal to mu
        v = torch.randn(n, d, device=device, dtype=dtype)
        v = v - (v * mu).sum(-1, keepdim=True) * mu
        v = torch.nn.functional.normalize(v, dim=-1)
        return w[:, None] * mu + torch.sqrt((1 - w**2).clamp(min=0))[:, None] * v

    @staticmethod
    def _wood_w(kappa: Tensor, d: int) -> Tensor:
        """Rejection sampler for the cosine component of a vMF in general dimension."""
        n = kappa.shape[0]
        device, dtype = kappa.device, kappa.dtype
        kappa = kappa.clamp(min=1e-6)
        b = (d - 1) / (2 * kappa + torch.sqrt(4 * kappa**2 + (d - 1) ** 2))
        x0 = (1 - b) / (1 + b)
        c = kappa * x0 + (d - 1) * torch.log(1 - x0**2)
        beta = torch.distributions.Beta((d - 1) / 2.0, (d - 1) / 2.0)
        w = torch.empty(n, device=device, dtype=dtype)
        todo = torch.ones(n, dtype=torch.bool, device=device)
        for _ in range(1000):
            if not todo.any():
                break
            m = int(todo.sum())
            z = beta.sample((m,)).to(device=device, dtype=dtype)
            bb, xx, cc, kk = b[todo], x0[todo], c[todo], kappa[todo]
            ww = (1 - (1 + bb) * z) / (1 - (1 - bb) * z)
            u = torch.rand(m, device=device, dtype=dtype)
            ok = kk * ww + (d - 1) * torch.log(1 - xx * ww) - cc >= torch.log(u)
            idx = todo.nonzero().squeeze(1)
            w[idx[ok]] = ww[ok]
            todo[idx[ok]] = False
        return w

    @torch.no_grad()
    def sample(self, n: int, device=None) -> Tensor:
        device = device or self.mu.device
        comp = torch.multinomial(self.log_weights.softmax(0), n, replacement=True).to(device)
        return self._sample_vmf(self.mu[comp].to(device), self.kappa[comp].to(device))

    @torch.no_grad()
    def sample_matched(self, x: Tensor) -> Tensor:
        """Noise paired with each data point: drawn from the vMF of the point's own cluster."""
        shape = x.shape
        comp = self.assign(x)
        eps = self._sample_vmf(self.mu[comp], self.kappa[comp])
        return eps.reshape(shape).to(x.dtype)

    # ------------------------------------------------------------------ density
    def log_prob(self, x: Tensor) -> Tensor:
        if self.dim != 3:
            raise NotImplementedError("log_prob implemented for S^2 only")
        flat = x.reshape(-1, 3)
        comp_logp = self.log_weights[None] + _log_c3(self.kappa)[None] + self.kappa[None] * (flat @ self.mu.T)
        return torch.logsumexp(comp_logp, dim=1).reshape(x.shape[:-1])
