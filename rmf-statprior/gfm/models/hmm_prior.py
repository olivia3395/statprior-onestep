"""
Hidden-Markov prior on a product of flat tori (T^p)^L, e.g. the (phi, psi) torsions of L consecutive
residues (p = 2). Each site carries a hidden state z_j in {1..S}; states follow a Markov chain with initial
distribution pi and transition matrix A; given its state, a site emits a truncated wrapped Gaussian on T^p,
    x_j = (mu_{z_j} + v) mod 2 pi,   v ~ N(0, Sigma_{z_j}) truncated to (-pi, pi)^p.
Compared with a flat mixture on T^{pL}, the number of parameters grows linearly in L and the dependence
between neighbouring sites (e.g. runs of helical residues) is modelled explicitly.

  fit(x)              Baum-Welch (EM, several initialisations) for S in the candidate set, S chosen by BIC
  sample(n)           prior samples (emission covariances inflated by cov_scale)
  sample_matched(x)   posterior coupling: a state path z ~ p(z | x) is drawn by forward filtering-backward
                      sampling under the fitted model, then noise is emitted from the inflated emissions
  log_prob(x)         exact log-density of the (inflated) prior by the forward algorithm

Posterior coupling preserves the prior's single-site and pairwise state marginals at an EM fixed point (the
M-step matches expected state and transition counts); for longer paths it holds approximately.
"""
import math

import torch
import torch.nn as nn
from torch import Tensor

TWO_PI = 2 * math.pi


def _wrap(a: Tensor) -> Tensor:
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


class TorusHMMPrior(nn.Module):
    def __init__(self, site_dim: int = 2, n_sites: int = 1, max_states: int = 32, state_candidates=None, select: str = "bic",
                 cov_scale: float = 2.0, n_iter: int = 60, tol: float = 1e-5, coupling: str = "soft",
                 n_init: int = 3):
        super().__init__()
        self.p = site_dim
        self.n_sites = n_sites
        self.max_states = max_states
        self.state_candidates = list(state_candidates) if state_candidates else [2, 4, 8, 12, 16, 24, 32]
        self.select = select
        self.cov_scale = cov_scale
        self.n_iter, self.tol, self.n_init = n_iter, tol, n_init
        self.coupling = coupling
        S, p = max_states, site_dim
        self.register_buffer("log_pi", torch.full((S,), -1e30))
        self.register_buffer("log_A", torch.full((S, S), -1e30))
        self.register_buffer("mu", torch.zeros(S, p))
        self.register_buffer("chol", torch.eye(p).repeat(S, 1, 1))
        self.register_buffer("log_trunc_fit", torch.zeros(S))
        self.register_buffer("log_trunc_prior", torch.zeros(S))
        self.register_buffer("n_active", torch.tensor(0))
        self.register_buffer("fitted", torch.tensor(False))
        self.selection_log = []

    # ------------------------------------------------------------------ helpers
    def _sites(self, x: Tensor) -> Tensor:
        """(n, p*L) -> (n, L, p) angles in [0, 2 pi)."""
        return torch.remainder(x.reshape(x.shape[0], -1, self.p), TWO_PI)

    @staticmethod
    def _log_trunc(chol: Tensor, n_mc: int = 20000, seed: int = 0) -> Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(n_mc, chol.shape[-1], generator=g, dtype=chol.dtype).to(chol.device)
        v = torch.einsum("sij,nj->sni", chol, z)
        frac = (v.abs() < math.pi).all(-1).double().mean(1).clamp_min(1e-12)
        return frac.log().to(chol.dtype)

    def _emission_logpdf(self, xs: Tensor, mu: Tensor, chol: Tensor, log_trunc: Tensor) -> Tensor:
        """xs: (n, L, p) -> (n, L, S) log emission densities."""
        d = _wrap(xs[:, :, None, :] - mu[None, None])                                 # (n, L, S, p)
        n, L, S, p = d.shape
        sol = torch.linalg.solve_triangular(chol[None].expand(n * L, S, p, p),
                                            d.reshape(n * L, S, p, 1), upper=False).reshape(n, L, S, p)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)       # (S,)
        lp = -0.5 * (sol ** 2).sum(-1) - 0.5 * p * math.log(TWO_PI) - 0.5 * logdet - log_trunc
        inside = (d.abs() < math.pi).all(-1)
        return torch.where(inside, lp, torch.full_like(lp, -1e30))

    @staticmethod
    def _forward(log_pi, log_A, le):
        """le: (n, L, S). Returns log alpha (n, L, S) and log-likelihood per sequence (n,)."""
        n, L, S = le.shape
        la = torch.empty_like(le)
        la[:, 0] = log_pi[None] + le[:, 0]
        for t in range(1, L):
            la[:, t] = torch.logsumexp(la[:, t - 1, :, None] + log_A[None], dim=1) + le[:, t]
        return la, torch.logsumexp(la[:, -1], dim=-1)

    @staticmethod
    def _backward(log_A, le):
        n, L, S = le.shape
        lb = torch.zeros_like(le)
        for t in range(L - 2, -1, -1):
            lb[:, t] = torch.logsumexp(log_A[None] + (le[:, t + 1] + lb[:, t + 1])[:, None, :], dim=-1)
        return lb

    # ------------------------------------------------------------------ EM
    def _init(self, xs: Tensor, S: int, g: torch.Generator):
        """k-means with circular means on pooled sites -> emissions; transitions from hard labels."""
        pts = xs.reshape(-1, self.p)
        m = min(len(pts), 20000)
        sub = pts[torch.randperm(len(pts), generator=g)[:m]]
        c = [sub[torch.randint(m, (1,), generator=g)].squeeze(0)]
        for _ in range(1, S):
            dd = (_wrap(sub[:, None] - torch.stack(c)[None]) ** 2).sum(-1).min(1).values
            c.append(sub[torch.multinomial(dd / dd.sum(), 1, generator=g)].squeeze(0))
        c = torch.stack(c)
        for _ in range(25):
            lab = (_wrap(sub[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            s_ = torch.zeros_like(c).index_add_(0, lab, torch.sin(sub))
            c_ = torch.zeros_like(c).index_add_(0, lab, torch.cos(sub))
            cnt = torch.bincount(lab, minlength=S)
            c = torch.where((cnt > 0)[:, None], torch.remainder(torch.atan2(s_, c_), TWO_PI), c)
        lab = (_wrap(pts[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        chol = []
        for k in range(S):
            d = _wrap(pts[lab == k] - c[k])
            cov = d.T @ d / max(len(d), 1) + 1e-3 * torch.eye(self.p, dtype=pts.dtype)
            chol.append(torch.linalg.cholesky(cov))
        chol = torch.stack(chol)
        labs = lab.reshape(xs.shape[0], xs.shape[1])
        A = torch.ones(S, S, dtype=pts.dtype)
        A.index_put_((labs[:, :-1].reshape(-1), labs[:, 1:].reshape(-1)), torch.ones(labs[:, 1:].numel(), dtype=pts.dtype),
                     accumulate=True)
        pi = torch.bincount(labs[:, 0], minlength=S).to(pts.dtype) + 1.0
        return (pi / pi.sum()).log(), (A / A.sum(1, keepdim=True)).log(), c, chol

    def _em(self, xs: Tensor, S: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        log_pi, log_A, mu, chol = self._init(xs, S, g)
        prev = -math.inf
        for it in range(self.n_iter):
            log_tr = self._log_trunc(chol, seed=seed + it)
            le = self._emission_logpdf(xs, mu, chol, log_tr)
            la, ll = self._forward(log_pi, log_A, le)
            lb = self._backward(log_A, le)
            lg = la + lb - ll[:, None, None]                                          # log gamma (n, L, S)
            gam = lg.exp()
            # expected transitions, accumulated one position at a time to keep memory at O(n S^2)
            lxi = torch.full((S, S), -math.inf, dtype=xs.dtype)
            for t in range(xs.shape[1] - 1):
                m = la[:, t, :, None] + (le[:, t + 1] + lb[:, t + 1])[:, None, :] - ll[:, None, None]
                lxi = torch.logaddexp(lxi, torch.logsumexp(m, dim=0))
            xi = (lxi + log_A).exp()
            log_pi = (gam[:, 0].sum(0) + 1e-6).log() - math.log(gam.shape[0] + S * 1e-6)
            A = xi + 1e-6
            log_A = (A / A.sum(1, keepdim=True)).log()
            # emissions: one Gauss-Newton step of the weighted circular mean, then weighted covariance
            w = gam.reshape(-1, S)                                                    # (nL, S)
            pts = xs.reshape(-1, self.p)
            wsum = w.sum(0).clamp_min(1e-8)
            for _ in range(2):
                d = _wrap(pts[:, None] - mu[None])                                    # (nL, S, p)
                mu = torch.remainder(mu + (w[..., None] * d).sum(0) / wsum[:, None], TWO_PI)
            d = _wrap(pts[:, None] - mu[None])
            cov = torch.einsum("ns,nsi,nsj->sij", w, d, d) / wsum[:, None, None]
            cov = cov + 1e-6 * torch.eye(self.p, dtype=pts.dtype)
            chol = torch.linalg.cholesky(cov)
            cur = ll.mean().item()
            if abs(cur - prev) < self.tol * max(1.0, abs(cur)):
                break
            prev = cur
        log_tr = self._log_trunc(chol, n_mc=100000, seed=seed + 999)
        ll = self._forward(log_pi, log_A, self._emission_logpdf(xs, mu, chol, log_tr))[1]
        return log_pi, log_A, mu, chol, log_tr, ll.sum().item()

    def fit(self, x: Tensor, seed: int = 0):
        xs = self._sites(x.double())
        n = xs.shape[0]
        self.selection_log, best = [], None
        cands = [self.max_states] if self.select == "fixed" else [s for s in self.state_candidates if s <= self.max_states]
        for S in cands:
            # several random initialisations; keep the one with the highest likelihood (EM has local optima)
            runs = [self._em(xs, S, seed + 1000 * r) for r in range(self.n_init)]
            log_pi, log_A, mu, chol, log_tr, ll = max(runs, key=lambda o: o[-1])
            k = (S - 1) + S * (S - 1) + S * (self.p + self.p * (self.p + 1) // 2)
            crit = -2 * ll + k * math.log(n)
            self.selection_log.append((S, crit))
            if best is None or crit < best[0]:
                best = (crit, S, log_pi, log_A, mu, chol, log_tr)
        _, S, log_pi, log_A, mu, chol, log_tr = best
        dt = self.mu.dtype
        self.log_pi.fill_(-1e30); self.log_A.fill_(-1e30)
        self.log_pi[:S] = log_pi.to(dt)
        self.log_A[:S, :S] = log_A.to(dt)
        self.mu[:S] = mu.to(dt)
        self.chol[:S] = chol.to(dt)
        self.log_trunc_fit[:S] = log_tr.to(dt)
        inflated = chol * math.sqrt(self.cov_scale)
        self.log_trunc_prior[:S] = self._log_trunc(inflated, n_mc=100000, seed=seed + 7).to(dt)
        self.n_active.fill_(S)
        self.fitted.fill_(True)
        return self

    # ------------------------------------------------------------------ sampling
    def _params(self, inflated: bool):
        S = int(self.n_active)
        chol = self.chol[:S] * (math.sqrt(self.cov_scale) if inflated else 1.0)
        lt = (self.log_trunc_prior if inflated else self.log_trunc_fit)[:S]
        return self.log_pi[:S], self.log_A[:S, :S], self.mu[:S], chol, lt

    def _emit(self, z: Tensor, mu: Tensor, chol: Tensor) -> Tensor:
        """z: (n, L) states -> (n, L, p) angles from the truncated wrapped Gaussians."""
        out = torch.empty(z.shape + (self.p,), device=z.device, dtype=mu.dtype)
        todo = torch.ones(z.shape, dtype=torch.bool, device=z.device)
        for _ in range(100):
            if not todo.any():
                break
            idx = todo.nonzero(as_tuple=True)
            zz = z[idx]
            v = torch.einsum("nij,nj->ni", chol[zz], torch.randn(len(zz), self.p, device=z.device, dtype=mu.dtype))
            ok = (v.abs() < math.pi).all(-1)
            sel = tuple(i[ok] for i in idx)
            out[sel] = torch.remainder(mu[zz[ok]] + v[ok], TWO_PI)
            todo[sel] = False
        return out

    @torch.no_grad()
    def sample(self, n: int, device=None, L: int = None) -> Tensor:
        device = device or self.mu.device
        L = L or self.n_sites
        log_pi, log_A, mu, chol, _ = self._params(inflated=True)
        z = torch.empty(n, L, dtype=torch.long, device=device)
        z[:, 0] = torch.multinomial(log_pi.exp().to(device).expand(n, -1), 1).squeeze(1)
        P = log_A.exp().to(device)
        for t in range(1, L):
            z[:, t] = torch.multinomial(P[z[:, t - 1]], 1).squeeze(1)
        return self._emit(z, mu.to(device), chol.to(device)).reshape(n, L * self.p)

    @torch.no_grad()
    def sample_matched(self, x: Tensor) -> Tensor:
        """Posterior coupling: z ~ p(z | x) by forward filtering-backward sampling, then noise ~ emissions(z)."""
        xs = self._sites(x)
        n, L, _ = xs.shape
        if self.coupling == "independent":
            return self.sample(n, device=x.device, L=L).to(x.dtype)
        log_pi, log_A, mu, chol_fit, lt_fit = self._params(inflated=False)
        le = self._emission_logpdf(xs.to(mu.dtype), mu, chol_fit, lt_fit)
        la, _ = self._forward(log_pi, log_A, le)
        z = torch.empty(n, L, dtype=torch.long, device=x.device)
        if self.coupling == "hard":                                   # Viterbi-like: per-site MAP of the posterior
            lb = self._backward(log_A, le)
            z = (la + lb).argmax(-1)
        else:
            z[:, -1] = torch.multinomial(torch.softmax(la[:, -1], -1), 1).squeeze(1)
            for t in range(L - 2, -1, -1):
                lw = la[:, t] + log_A[:, z[:, t + 1]].T
                z[:, t] = torch.multinomial(torch.softmax(lw, -1), 1).squeeze(1)
        _, _, mu_p, chol_p, _ = self._params(inflated=True)
        return self._emit(z, mu_p, chol_p).reshape(n, L * self.p).to(x.dtype)

    @torch.no_grad()
    def log_prob(self, x: Tensor, inflated: bool = True) -> Tensor:
        xs = self._sites(x)
        log_pi, log_A, mu, chol, lt = self._params(inflated=inflated)
        return self._forward(log_pi, log_A, self._emission_logpdf(xs.to(mu.dtype), mu, chol, lt))[1].to(x.dtype)
