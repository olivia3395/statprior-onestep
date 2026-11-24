"""
Statistically estimated priors for one-step generation on manifolds.

A mixture of *tangent-space (Riemannian) Gaussians*: component k has a mean point mu_k
on the manifold and a full covariance Sigma_k in the tangent space at mu_k, and generates
x = Exp_{mu_k}(v), v ~ N(0, Sigma_k) truncated to |v| < pi (so Exp is one-to-one).
Components can be elongated, so thin rings / arcs / bands are described with few components.

Statistical ingredients
  * Estimation: EM on the manifold (E-step: posterior responsibilities; M-step:
    weighted Frechet mean + weighted tangent covariance).
  * Model selection: number of components K by BIC, or by held-out log-likelihood (CV).
  * Coupling: each data point x picks a component k ~ p(k | x) (posterior, "soft"),
    then noise eps ~ component k. At an EM fixed point the average responsibility equals
    the mixture weight, so the noise marginal equals the prior used at sampling time.
    "hot" (hierarchical OT) keeps this outer, statistical assignment and, inside each
    component, pairs the noise and data points of the minibatch by an optimal one-to-one
    matching under the squared geodesic distance. The noise is only permuted, so its
    marginal is unchanged; when the prior is already accurate, each point is paired with
    nearby noise and the flow to learn is close to the identity.
  * cov_scale > 1 inflates the fitted covariances (prior deliberately wider than the data).

Density w.r.t. the Riemannian volume (exact, incl. exp-map Jacobian and truncation):
    p_k(x) = N(v; 0, Sigma_k) / Z_k * J(|v|)^{-1},   v = Log_{mu_k}(x) in normal coordinates
with J(r) = sin r / r on the unit sphere S^2 and J(r) = 2(1 - cos r) / r^2 on SO(3).

Classes: SphereTangentGMM (S^2, tangent dim 2), SO3TangentGMM (SO(3), tangent dim 3),
TorusTangentGMM (flat torus T^d, tangent dim d; wrapped Gaussians truncated to the cube (-pi, pi)^d).
The geometry is isolated in a few hooks (_coords, _exp, _log_jac, _init_centers).
"""

import math

import torch
from torch import Tensor, nn

LOG2PI = math.log(2 * math.pi)


# ============================================================ sphere geometry
def _basis(mu: Tensor) -> Tensor:
    """(K,3) unit vectors -> (K,3,2) orthonormal bases of the tangent planes."""
    t = torch.zeros_like(mu)
    use_x = mu[:, 0].abs() < 0.9
    t[use_x, 0] = 1.0
    t[~use_x, 1] = 1.0
    e1 = torch.nn.functional.normalize(t - (t * mu).sum(-1, keepdim=True) * mu, dim=-1)
    e2 = torch.cross(mu, e1, dim=-1)
    return torch.stack([e1, e2], dim=-1)


def _log_map(mu: Tensor, x: Tensor) -> Tensor:
    """Log_mu(x) in ambient coordinates."""
    c = (mu * x).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.arccos(c)
    u = x - c * mu
    return theta * u / u.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _exp_map(mu: Tensor, v: Tensor) -> Tensor:
    r = v.norm(dim=-1, keepdim=True)
    out = torch.cos(r) * mu + torch.sin(r) * v / r.clamp(min=1e-12)
    return torch.nn.functional.normalize(out, dim=-1)


def _log_jac(r: Tensor) -> Tensor:
    """log(r / sin r): density correction of the exponential map on the unit sphere."""
    r = r.clamp(max=math.pi - 1e-4)
    small = r < 1e-3
    val = torch.log(r.clamp(min=1e-12)) - torch.log(torch.sin(r).clamp(min=1e-12))
    return torch.where(small, r**2 / 6, val)


class SphereTangentGMM(nn.Module):
    """
    Tangent-space Gaussian mixture on the sphere S^2.

    :param max_k: maximal number of components (buffer size).
    :param k_candidates: candidate K values for model selection (default 1..max_k).
    :param select: "bic", "cv" (held-out log-likelihood) or "fixed" (use max_k).
    :param cov_scale: multiplies fitted covariances for the prior (>= 1 widens it).
    :param coupling: "soft" (posterior), "hard" (MAP component), "independent", or
        "hot" (posterior component + optimal matching within each component).
    :param ot_chunk: maximal block size for the within-component matching.
    """

    tdim = 2          # tangent dimension
    pshape = (3,)     # shape of one point

    def __init__(self, max_k: int = 16, k_candidates=None, select: str = "bic",
                 cov_scale: float = 2.0, coupling: str = "soft", n_iter: int = 100,
                 reg: float = 1e-6, ot_chunk: int = 128):
        super().__init__()
        self.ot_chunk = ot_chunk
        assert select in ("bic", "cv", "fixed")
        assert coupling in ("soft", "hard", "independent", "hot")
        self.max_k = max_k
        self.k_candidates = list(k_candidates) if k_candidates else list(range(1, max_k + 1))
        self.select = select
        self.cov_scale = cov_scale
        self.coupling = coupling
        self.n_iter = n_iter
        self.reg = reg
        K, d = max_k, self.tdim
        self.register_buffer("mu", torch.zeros((K,) + self.pshape))
        self.register_buffer("chol_fit", torch.eye(d).repeat(K, 1, 1))  # MLE covariance (responsibilities)
        self.register_buffer("log_weights", torch.full((K,), -1e30))
        self.register_buffer("log_trunc_fit", torch.zeros(K))
        self.register_buffer("log_trunc_prior", torch.zeros(K))
        self.register_buffer("n_active", torch.tensor(0))
        self.register_buffer("fitted", torch.tensor(False))
        self.selection_log = []  # (K, criterion value)

    # ------------------------------------------------------------ geometry hooks
    def _flat(self, x: Tensor) -> Tensor:
        return x.reshape((-1,) + self.pshape)

    def _prepare(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, dim=-1)

    def _coords(self, mu: Tensor, x: Tensor) -> Tensor:
        """(K,...) means, (n,...) points -> (n,K,d) normal coordinates of Log_{mu_k}(x_i)."""
        return torch.einsum("nkd,kdj->nkj", _log_map(mu[None], x[:, None]), _basis(mu))

    def _exp(self, mu_sel: Tensor, v: Tensor) -> Tensor:
        """(n,...) base points, (n,d) coordinates -> (n,...) points."""
        return _exp_map(mu_sel, torch.einsum("nij,nj->ni", _basis(mu_sel), v))

    def _log_jac(self, r: Tensor) -> Tensor:
        return _log_jac(r)

    def _in_domain(self, v: Tensor) -> Tensor:
        """Where Exp_mu is one-to-one: the ball |v| < pi (sphere, SO(3))."""
        return v.norm(dim=-1) < math.pi - 1e-4

    def _init_centers(self, x, k, g):
        n = x.shape[0]
        c = [x[torch.randint(n, (1,), generator=g)].squeeze(0)]
        for _ in range(1, k):
            dd = (1 - x @ torch.stack(c).T).min(1).values.clamp(min=0)
            c.append(x[torch.multinomial(dd / dd.sum() if dd.sum() > 0 else torch.ones(n) / n, 1, generator=g)].squeeze(0))
        c = torch.stack(c)
        for _ in range(20):
            lab = (x @ c.T).argmax(1)
            new = torch.zeros_like(c).index_add_(0, lab, x)
            empty = new.norm(dim=1) < 1e-12
            new[empty] = c[empty]
            c = torch.nn.functional.normalize(new, dim=-1)
        return c

    # --------------------------------------------------------- component densities
    def _log_trunc(self, chol: Tensor, n_mc: int = 200000, seed: int = 0) -> Tensor:
        """log P(|v| < pi) for v ~ N(0, L L^T), per component (Monte Carlo, fixed seed)."""
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n_mc, chol.shape[-1], generator=g, dtype=chol.dtype)
        v = torch.einsum("kij,nj->kni", chol.cpu(), z)
        p = self._in_domain(v).double().mean(1).clamp(min=1e-12)
        return torch.log(p).to(chol)

    def _comp_logpdf(self, x, mu, chol, log_trunc):
        """(n,...) points, K components -> (n,K) log densities w.r.t. the Riemannian volume."""
        v = self._coords(mu, x)  # (n,K,d)
        r = v.norm(dim=-1)
        d = v.shape[-1]
        sol = torch.linalg.solve_triangular(chol[None].expand(v.shape[0], -1, -1, -1), v[..., None], upper=False)
        maha = (sol.squeeze(-1) ** 2).sum(-1)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        lp = -0.5 * maha - 0.5 * d * LOG2PI - 0.5 * logdet[None] - log_trunc[None] + self._log_jac(r)
        return torch.where(self._in_domain(v), lp, torch.full_like(lp, -1e30))

    # --------------------------------------------------------------------- EM
    def _em(self, x, k, seed):
        g = torch.Generator().manual_seed(seed)
        n, d = x.shape[0], self.tdim
        mu = self._init_centers(x, k, g)
        chol = torch.eye(d, dtype=x.dtype).repeat(k, 1, 1) * 0.3
        logw = torch.full((k,), -math.log(k), dtype=x.dtype)
        prev = -float("inf")
        for it in range(self.n_iter):
            log_trunc = self._log_trunc(chol, n_mc=20000, seed=it)
            lp = logw[None] + self._comp_logpdf(x, mu, chol, log_trunc)
            ll = torch.logsumexp(lp, 1)
            resp = torch.exp(lp - ll[:, None])
            nk = resp.sum(0).clamp(min=1e-8)
            logw = torch.log(nk / n)
            for _ in range(3):  # weighted Frechet mean (Gauss-Newton steps)
                t = (resp[..., None] * self._coords(mu, x)).sum(0) / nk[:, None]
                mu = self._exp(mu, t)
            v = self._coords(mu, x)
            cov = torch.einsum("nk,nki,nkj->kij", resp, v, v) / nk[:, None, None]
            cov = cov + self.reg * torch.eye(d, dtype=x.dtype)
            chol = torch.linalg.cholesky(cov)
            cur = ll.mean().item()
            if abs(cur - prev) < 1e-7:
                break
            prev = cur
        return mu, chol, logw, self._log_trunc(chol)

    def _loglik(self, x, mu, chol, logw, log_trunc):
        return torch.logsumexp(logw[None] + self._comp_logpdf(x, mu, chol, log_trunc), 1)

    @torch.no_grad()
    def fit(self, x: Tensor, seed: int = 0):
        x = self._prepare(self._flat(x.double().cpu()))
        n, d = x.shape[0], self.tdim
        cands = [k for k in self.k_candidates if k <= min(self.max_k, n)]
        if self.select == "fixed":
            cands = [min(self.max_k, n)]
        self.selection_log = []
        best = None
        if self.select == "cv":
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
            tr, va = x[perm[: int(0.8 * n)]], x[perm[int(0.8 * n):]]
        for k in cands:
            if self.select == "cv":
                params = self._em(tr, k, seed)
                crit = -self._loglik(va, *params).mean().item()
            else:
                params = self._em(x, k, seed)
                ll = self._loglik(x, *params).sum().item()
                n_params = k * (d + d * (d + 1) // 2) + (k - 1)
                crit = -2 * ll + n_params * math.log(n)
            self.selection_log.append((k, crit))
            if best is None or crit < best[0]:
                best = (crit, k, params)
        _, k, params = best
        if self.select == "cv":
            params = self._em(x, k, seed)
        mu, chol, logw, log_trunc = params
        K = self.max_k
        self.mu.zero_(); self.mu[:k] = mu.float()
        self.chol_fit.copy_(torch.eye(d).repeat(K, 1, 1)); self.chol_fit[:k] = chol.float()
        self.log_weights.fill_(-1e30); self.log_weights[:k] = logw.float()
        self.log_trunc_fit.zero_(); self.log_trunc_fit[:k] = log_trunc.float()
        self.log_trunc_prior.zero_()
        self.log_trunc_prior[:k] = self._log_trunc(chol * math.sqrt(self.cov_scale)).float()
        self.n_active.fill_(k)
        self.fitted.fill_(True)
        return self

    # ------------------------------------------------------------------ accessors
    def _active(self):
        k = int(self.n_active)
        chol_prior = self.chol_fit[:k] * math.sqrt(self.cov_scale)
        return k, self.mu[:k], self.chol_fit[:k], chol_prior, self.log_weights[:k]

    def responsibilities(self, x: Tensor) -> Tensor:
        """Posterior p(k | x) under the fitted (MLE) mixture."""
        k, mu, chol, _, logw = self._active()
        lp = logw[None] + self._comp_logpdf(self._flat(x).to(mu), mu, chol, self.log_trunc_fit[:k])
        return torch.softmax(lp, dim=1)

    def assign(self, x: Tensor) -> Tensor:
        return self.responsibilities(x).argmax(1)

    def _sample_comp(self, comp: Tensor) -> Tensor:
        k, mu, _, chol_prior, _ = self._active()
        n = comp.shape[0]
        v = torch.empty(n, self.tdim, device=mu.device, dtype=mu.dtype)
        todo = torch.ones(n, dtype=torch.bool, device=mu.device)
        for _ in range(100):  # rejection for the truncation |v| < pi
            m = int(todo.sum())
            if m == 0:
                break
            z = torch.randn(m, self.tdim, device=mu.device, dtype=mu.dtype)
            cand = torch.einsum("nij,nj->ni", chol_prior[comp[todo]], z)
            ok = self._in_domain(cand)
            idx = todo.nonzero().squeeze(1)
            v[idx[ok]] = cand[ok]
            todo[idx[ok]] = False
        return self._exp(mu[comp], v)

    @torch.no_grad()
    def sample(self, n: int, device=None) -> Tensor:
        k, *_, logw = self._active()
        comp = torch.multinomial(torch.softmax(logw, 0), n, replacement=True).to(self.mu.device)
        out = self._sample_comp(comp)
        return out.to(device) if device is not None else out

    @torch.no_grad()
    def sample_matched(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = self._flat(x)
        if self.coupling == "independent":
            return self.sample(flat.shape[0], device=x.device).reshape(shape).to(x.dtype)
        resp = self.responsibilities(flat)
        comp = resp.argmax(1) if self.coupling == "hard" else torch.multinomial(resp, 1).squeeze(1)
        eps = self._sample_comp(comp)
        if self.coupling == "hot":
            eps = self._match_within_components(eps, flat.to(eps), comp)
        return eps.reshape(shape).to(x.dtype)

    def _pair_cost(self, noise: Tensor, data: Tensor) -> Tensor:
        """Squared geodesic distances, (n_noise, n_data)."""
        return (self._coords(noise, data) ** 2).sum(-1).transpose(0, 1)

    @torch.no_grad()
    def _match_within_components(self, eps: Tensor, x: Tensor, comp: Tensor) -> Tensor:
        """Permute eps inside each component so that eps[j] is optimally matched to x[j]."""
        from scipy.optimize import linear_sum_assignment

        out = eps.clone()
        for k in torch.unique(comp).tolist():
            idx = (comp == k).nonzero().squeeze(1)
            if idx.numel() < 2:
                continue
            idx = idx[torch.randperm(idx.numel(), device=idx.device)]  # random blocks
            for s0 in range(0, idx.numel(), self.ot_chunk):
                blk = idx[s0 : s0 + self.ot_chunk]
                if blk.numel() < 2:
                    continue
                cost = self._pair_cost(eps[blk], x[blk]).double().cpu().numpy()
                rows, cols = linear_sum_assignment(cost)  # noise rows[i] -> data cols[i]
                perm = torch.empty(blk.numel(), dtype=torch.long)
                perm[torch.as_tensor(cols)] = torch.as_tensor(rows)
                out[blk] = eps[blk][perm.to(eps.device)]
        return out

    def log_prob(self, x: Tensor) -> Tensor:
        k, mu, _, chol_prior, logw = self._active()
        flat = self._flat(x).to(mu)
        lp = logw[None] + self._comp_logpdf(flat, mu, chol_prior, self.log_trunc_prior[:k])
        return torch.logsumexp(lp, 1).reshape(x.shape[: x.dim() - len(self.pshape)])


# ================================================================ SO(3) geometry
def _hat(w: Tensor) -> Tensor:
    z = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([z, -w[..., 2], w[..., 1]], -1),
        torch.stack([w[..., 2], z, -w[..., 0]], -1),
        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def so3_exp(w: Tensor) -> Tensor:
    """Rotation vectors (...,3) -> rotation matrices (...,3,3) (Rodrigues)."""
    th = w.norm(dim=-1, keepdim=True)[..., None]
    K = _hat(w)
    small = th < 1e-4
    a = torch.where(small, 1 - th**2 / 6, torch.sin(th) / th.clamp(min=1e-12))
    b = torch.where(small, 0.5 - th**2 / 24, (1 - torch.cos(th)) / (th**2).clamp(min=1e-24))
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return I + a * K + b * (K @ K)


def so3_log(R: Tensor) -> Tensor:
    """Rotation matrices (...,3,3) -> rotation vectors (...,3), angle in [0, pi].
    Stable near pi: goes through a unit quaternion computed with Shepperd's method."""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    i = torch.stack([tr, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1).argmax(-1)
    s0 = torch.sqrt((1 + tr).clamp(min=1e-12)) * 2
    q0 = torch.stack([(m[..., 2, 1] - m[..., 1, 2]) / s0, (m[..., 0, 2] - m[..., 2, 0]) / s0,
                      (m[..., 1, 0] - m[..., 0, 1]) / s0, 0.25 * s0], -1)
    s1 = torch.sqrt((1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp(min=1e-12)) * 2
    q1 = torch.stack([0.25 * s1, (m[..., 0, 1] + m[..., 1, 0]) / s1, (m[..., 0, 2] + m[..., 2, 0]) / s1,
                      (m[..., 2, 1] - m[..., 1, 2]) / s1], -1)
    s2 = torch.sqrt((1 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2]).clamp(min=1e-12)) * 2
    q2 = torch.stack([(m[..., 0, 1] + m[..., 1, 0]) / s2, 0.25 * s2, (m[..., 1, 2] + m[..., 2, 1]) / s2,
                      (m[..., 0, 2] - m[..., 2, 0]) / s2], -1)
    s3 = torch.sqrt((1 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1]).clamp(min=1e-12)) * 2
    q3 = torch.stack([(m[..., 0, 2] + m[..., 2, 0]) / s3, (m[..., 1, 2] + m[..., 2, 1]) / s3, 0.25 * s3,
                      (m[..., 1, 0] - m[..., 0, 1]) / s3], -1)
    q = torch.where((i == 0)[..., None], q0,
                    torch.where((i == 1)[..., None], q1, torch.where((i == 2)[..., None], q2, q3)))
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[..., 3:4] < 0, -q, q)
    vn = q[..., :3].norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(vn, q[..., 3:4])
    scale = torch.where(vn < 1e-8, 2 / q[..., 3:4].clamp(min=1e-12), ang / vn.clamp(min=1e-12))
    return scale * q[..., :3]


def _so3_log_jac(r: Tensor) -> Tensor:
    """log(r^2 / (2 (1 - cos r))): density correction of exp on SO(3) (distance = rotation angle)."""
    r = r.clamp(max=math.pi - 1e-4)
    small = r < 1e-3
    val = 2 * torch.log(r.clamp(min=1e-12)) - torch.log((2 * (1 - torch.cos(r))).clamp(min=1e-24))
    return torch.where(small, r**2 / 12, val)


class SO3TangentGMM(SphereTangentGMM):
    """
    Tangent-space Gaussian mixture on SO(3): x = mu_k Exp(hat(v)), v ~ N(0, Sigma_k) in R^3,
    truncated to |v| < pi. Density w.r.t. the Riemannian volume of the metric whose geodesic
    distance is the rotation angle (total volume 8 pi^2), matching SO3.distance in the code base.
    """

    tdim = 3
    pshape = (3, 3)

    def _prepare(self, x):
        return x

    def _coords(self, mu, x):
        return so3_log(mu.transpose(-1, -2)[None] @ x[:, None])  # (n,K,3): Log(mu_k^T x_i)

    def _exp(self, mu_sel, v):
        return mu_sel @ so3_exp(v)

    def _log_jac(self, r):
        return _so3_log_jac(r)

    @staticmethod
    def _proj_so3(M):
        U, _, Vt = torch.linalg.svd(M)
        d = torch.sign(torch.linalg.det(U @ Vt))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
        return U @ D @ Vt

    def _init_centers(self, x, k, g):
        n = x.shape[0]

        def ang(a, b):
            return torch.arccos(((torch.einsum("nij,kij->nk", a, b) - 1) / 2).clamp(-1, 1))

        c = [x[torch.randint(n, (1,), generator=g)].squeeze(0)]
        for _ in range(1, k):
            dd = ang(x, torch.stack(c)).min(1).values ** 2
            c.append(x[torch.multinomial(dd / dd.sum() if dd.sum() > 0 else torch.ones(n) / n, 1, generator=g)].squeeze(0))
        c = torch.stack(c)
        for _ in range(20):  # k-means with chordal (SVD-projected) means
            lab = ang(x, c).argmin(1)
            new = torch.zeros_like(c).index_add_(0, lab, x)
            cnt = torch.bincount(lab, minlength=k)
            c = torch.where((cnt > 0)[:, None, None], self._proj_so3(new + 1e-9 * torch.eye(3, dtype=x.dtype)), c)
        return c


# ============================================================ flat torus T^d
def _wrap(d: Tensor) -> Tensor:
    return torch.remainder(d + math.pi, 2 * math.pi) - math.pi


class TorusTangentGMM(SphereTangentGMM):
    """
    Mixture of (truncated) wrapped Gaussians on the flat torus [0, 2 pi)^d:
    x = (mu_k + v) mod 2 pi, v ~ N(0, Sigma_k) truncated to the cube (-pi, pi)^d, where the
    exponential map is one-to-one. The metric is flat, so there is no Jacobian term, and the
    density is w.r.t. Lebesgue measure on [0, 2 pi)^d (as FlatTorus.logp0 = -d log 2 pi).
    A full covariance per component captures the dependence between angles, e.g. the
    (phi, psi) correlation of Ramachandran plots.
    """

    def __init__(self, dim: int, **kwargs):
        object.__setattr__(self, "tdim", dim)
        object.__setattr__(self, "pshape", (dim,))
        super().__init__(**kwargs)

    def _prepare(self, x):
        return torch.remainder(x, 2 * math.pi)

    def _coords(self, mu, x):
        return _wrap(x[:, None] - mu[None])

    def _exp(self, mu_sel, v):
        return torch.remainder(mu_sel + v, 2 * math.pi)

    def _log_jac(self, r):
        return torch.zeros_like(r)

    def _in_domain(self, v):
        return (v.abs() < math.pi).all(dim=-1)

    def _init_centers(self, x, k, g):
        n = x.shape[0]

        def d2(a, b):
            return (_wrap(a[:, None] - b[None]) ** 2).sum(-1)

        c = [x[torch.randint(n, (1,), generator=g)].squeeze(0)]
        for _ in range(1, k):
            dd = d2(x, torch.stack(c)).min(1).values
            c.append(x[torch.multinomial(dd / dd.sum() if dd.sum() > 0 else torch.ones(n) / n, 1, generator=g)].squeeze(0))
        c = torch.stack(c)
        for _ in range(20):  # k-means with circular (per-angle) means
            lab = d2(x, c).argmin(1)
            s_ = torch.zeros_like(c).index_add_(0, lab, torch.sin(x))
            c_ = torch.zeros_like(c).index_add_(0, lab, torch.cos(x))
            cnt = torch.bincount(lab, minlength=k)
            c = torch.where((cnt > 0)[:, None], torch.remainder(torch.atan2(s_, c_), 2 * math.pi), c)
        return c
