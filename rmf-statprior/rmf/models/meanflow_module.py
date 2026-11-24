"""
Generalised Mean Flow module.
"""

import torch
from torch import Tensor

from gfm.models import match_dims
from gfm.models import RiemannianGenerativeModule
from gfm.models.structured_prior import SphereClusterPrior
from gfm.models.hmm_prior import TorusHMMPrior
from gfm.models.stat_prior import SO3TangentGMM, SphereTangentGMM, TorusTangentGMM
from gfm.manifold import ManifoldType


class MeanFlowModule(RiemannianGenerativeModule):
    """
    Generalised Mean Flow module.

    :param prop_equal_times: Proportion of times that should be equal (`r = t`).
    :param time_distribution: Distribution to sample times from. One of "uniform" or "lognormal".
    """

    def __init__(
        self,
        prop_equal_times: float = 0.0,
        time_distribution: str = "uniform",
        prior_type: str = "uniform",
        prior_k: int = 16,
        prior_kappa_scale: float = 0.5,
        prior_coupling: str = "matched",
        prior_fit_seed: int = 0,
        prior_select: str = "bic",
        prior_cov_scale: float = 2.0,
        prior_fit_max_points: int = 10000,
        prior_em_iters: int = 100,
        *args,
        **kwargs,
    ):
        """
        Args:
            prop_equal_times (float): Proportion of times that should be equal (`r = t`).
            prior_type (str): "uniform" (original RMF) or "cluster" (vMF-mixture prior, sphere only).
            prior_k (int): number of clusters / mixture components for the cluster prior.
            prior_kappa_scale (float): scales fitted concentrations (<1 widens the prior, 0 = uniform).
            prior_coupling (str): "matched" pairs each data point with noise from its own cluster;
                "independent" draws noise from the mixture independently of the data (ablation);
                for prior_type="gmm": "soft" (posterior p(k|x)), "hard" (MAP) or "independent".
            prior_type="gmm": tangent-space Gaussian mixture fitted by EM (sphere only);
                prior_k is the maximal number of components, prior_select in {"bic","cv","fixed"},
                prior_cov_scale inflates the fitted covariances.
        """
        # assert len(in_shape) >= 2, f"Input shape must have at least 2 dimensions `(n_products, manifold_dims...)`, got shape: {in_shape}"
        super().__init__(*args, **kwargs)
        assert prior_type in ("uniform", "cluster", "gmm", "hmm"), prior_type
        assert prior_coupling in ("matched", "independent", "soft", "hard", "hot"), prior_coupling
        self.prior_type = prior_type
        self.prior_coupling = prior_coupling
        self.prior_fit_seed = prior_fit_seed
        self.prior_fit_max_points = prior_fit_max_points
        self.structured_prior = None
        if prior_type == "cluster":
            if self.manifold.get_type() != ManifoldType.SPHERE:
                raise NotImplementedError("cluster prior is implemented for the sphere only")
            self.structured_prior = SphereClusterPrior(
                dim=int(self.hparams.in_shape[-1]),
                n_clusters=prior_k,
                kappa_scale=prior_kappa_scale,
            )
        elif prior_type == "gmm":
            mtype = self.manifold.get_type()
            extra = {}
            if mtype == ManifoldType.SPHERE:
                gmm_cls = SphereTangentGMM
            elif mtype == ManifoldType.SO3:
                gmm_cls = SO3TangentGMM
            elif mtype == ManifoldType.FLAT_TORUS:
                gmm_cls = TorusTangentGMM
                extra = dict(dim=int(self.hparams.in_shape[-1]))
            else:
                raise NotImplementedError(f"gmm prior not implemented for {mtype}")
            cands = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if k <= prior_k]
            self.structured_prior = gmm_cls(
                **extra,
                max_k=prior_k,
                n_iter=prior_em_iters,
                k_candidates=cands,
                select=prior_select,
                cov_scale=prior_cov_scale,
                coupling=prior_coupling if prior_coupling in ("soft", "hard", "independent", "hot") else "soft",
            )
        elif prior_type == "hmm":
            # hidden-Markov prior on (T^2)^L: sites are consecutive (phi, psi) pairs
            if self.manifold.get_type() != ManifoldType.FLAT_TORUS:
                raise NotImplementedError("hmm prior is implemented for the flat torus only")
            d = int(self.hparams.in_shape[-1])
            assert d % 2 == 0, "hmm prior expects (phi, psi) pairs"
            self.structured_prior = TorusHMMPrior(
                site_dim=2, n_sites=d // 2, max_states=prior_k,
                state_candidates=[k for k in (2, 4, 8, 12, 16, 24, 32, 48) if k <= prior_k],
                select=prior_select, cov_scale=prior_cov_scale,
                coupling=prior_coupling if prior_coupling in ("soft", "hard", "independent") else "soft",
            )
        if time_distribution == "uniform":
            self.time_distribution = torch.distributions.Uniform(
                0.0, 1.0, validate_args=False
            )
        else:
            self.time_distribution = torch.distributions.LogNormal(
                -0.4, 1.0, validate_args=False
            )

    @property
    def is_sampling_time_zero_to_one(self) -> bool:
        return False

    @torch.no_grad()
    def sample_r_t(self, n: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Sample random times `r` and `t` for the mean flow.
        `r` is sampled uniformly in [0, 1] and `t` is sampled uniformly in [0, 1].
        """
        if self.hparams.time_distribution == "uniform":
            t = torch.rand(n, device=device)
            r = torch.rand(n, device=device) * t
            return r, t
        t = self.time_distribution.sample((n,)).to(device)
        r = self.time_distribution.sample((n,)).to(device)
        msk = r > t
        tmp = r[msk]
        r[msk] = t[msk]
        t[msk] = tmp
        return r.sigmoid(), t.sigmoid()

    # come from alpha-flow
    @torch.no_grad()
    def alpha_sample(self):
        k_s = 0
        k_e = self.epoch
        gamma = 25
        eta = 0.005
        k = self.current_epoch + 1
        scale = 1/(k_e - k_s)
        mid = - (k_s + k_e) / 2 / (k_e - k_s)
        alpha = 1 - (1 / (1 + torch.exp(torch.tensor((scale * k + mid) * gamma))))
        alpha = 1 if alpha > 1 - eta else (0 if alpha < eta else alpha)
        return alpha

    # ------------------------------------------------------------------ prior
    def _noise_for(self, x: Tensor) -> Tensor:
        """Noise paired with the data batch `x` during training."""
        if self.structured_prior is None:
            return self.manifold.prior(shape=x.shape, device=x.device)
        if self.prior_type in ("gmm", "hmm") or self.prior_coupling == "matched":
            return self.structured_prior.sample_matched(x)
        return self.structured_prior.sample(x.shape[0], device=x.device).reshape(x.shape).to(x.dtype)

    def _prior_samples(self, shape: tuple, device) -> Tensor:
        """Fresh prior samples used as starting points at generation time."""
        if self.structured_prior is None:
            return self.manifold.prior(shape=shape, device=device)
        out = self.structured_prior.sample(shape[0], device=device)
        if isinstance(self.structured_prior, SO3TangentGMM):
            return out  # (n,3,3), same as SO3.prior (which ignores in_shape)
        return out.reshape(shape)

    def log_prob_prior(self, x: Tensor) -> Tensor:
        if self.structured_prior is None:
            return self.manifold.logp0(x)
        return self.structured_prior.log_prob(x)

    def _training_points(self) -> Tensor:
        dm = self.trainer.datamodule
        ds = dm.data_train
        ds = getattr(ds, "dset", ds)  # unwrap ExpandDataset
        pts = torch.stack([torch.as_tensor(ds[i]) for i in range(len(ds))]).float()
        return pts[..., : int(self.hparams.in_shape[-1])]

    def on_fit_start(self) -> None:
        if self.structured_prior is not None and not bool(self.structured_prior.fitted):
            pts = self._training_points()
            if self.prior_type in ("gmm", "hmm") and len(pts) > self.prior_fit_max_points:
                g = torch.Generator().manual_seed(self.prior_fit_seed)
                pts = pts[torch.randperm(len(pts), generator=g)[: self.prior_fit_max_points]]
            self.structured_prior.fit(pts.cpu(), seed=self.prior_fit_seed)
            self.structured_prior.to(self.device)
            if self.prior_type == "hmm":
                sel = ", ".join(f"S={k}:{c:.0f}" for k, c in self.structured_prior.selection_log)
                print(f"[hmm prior] fitted on {len(pts)} sequences of {self.structured_prior.n_sites} sites: "
                      f"selected S={int(self.structured_prior.n_active)} states ({self.structured_prior.select}; {sel})")
            elif self.prior_type == "gmm":
                sel = ", ".join(f"K={k}:{c:.0f}" for k, c in self.structured_prior.selection_log)
                print(f"[gmm prior] fitted on {len(pts)} points: selected K={int(self.structured_prior.n_active)} "
                      f"({self.structured_prior.select}; {sel})")
            else:
                w = self.structured_prior.log_weights.exp()
                print(
                    f"[cluster prior] fitted on {len(pts)} points: K={int((w > 0).sum())}, "
                    f"kappa range [{self.structured_prior.kappa.min():.1f}, {self.structured_prior.kappa.max():.1f}]"
                )

    def on_test_end(self) -> None:
        """Dump final test metrics next to the hydra outputs, for scripts/collect_results.py."""
        import json, os
        out = {k: float(v) for k, v in self.trainer.callback_metrics.items() if k.startswith("test/")}
        out.update(
            prior_type=self.prior_type,
            prior_coupling=self.prior_coupling,
            prior_k=int(self.hparams.get("prior_k", 0)),
            prior_kappa_scale=float(self.hparams.get("prior_kappa_scale", 0.0)),
            prior_select=str(self.hparams.get("prior_select", "")),
            prior_cov_scale=float(self.hparams.get("prior_cov_scale", 0.0)),
            prior_k_selected=int(self.structured_prior.n_active) if self.prior_type in ("gmm", "hmm") else None,
            my_method=self.my_method,
            dataset=str(getattr(self.trainer.datamodule.hparams, "dataset_file", None)
                        or getattr(self.trainer.datamodule.hparams, "data_type", "")),
        )
        path = os.path.join(self.trainer.default_root_dir, "test_metrics.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[results] wrote {path}")

    def get_loss(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            eps = self._noise_for(x)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )

        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )
        with torch.no_grad():
            # both v and dudt are tangent vectors already
            u_tgt = v - (t - r) * dudt.detach()

        diff = u - u_tgt
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        return self.manifold.inner(xt, diff, diff).mean()

    def get_loss_imf(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(0.5 * x.shape[0])
            eps = self._noise_for(x)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

        xt, v = torch.func.jvp(
            geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
        )
        v_tt = self.forward(xt, t, t)
        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v_tt, torch.zeros_like(r), torch.ones_like(t)),
        )
        V = u + (t - r) * dudt.detach()

        diff = V - v
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        return self.manifold.inner(xt, diff, diff).mean()


    def get_loss_alpha_p(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            # n_equal = int(0 * x.shape[0])
            eps = self._noise_for(x)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )

        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )
        with torch.no_grad():
            # both v and dudt are tangent vectors already
            u_tgt = v - (t - r) * dudt.detach()
            alpha = self.alpha_sample()

        diff = u - v
        diff_dudt = (t - r) * dudt.detach()
        xt_1 = xt
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        l_1 = self.manifold.inner(xt, diff, diff).mean()
        if diff_dudt.isnan().any():
            nans = diff_dudt.isnan().any(dim=(-1, -2))
            xt_1 = xt_1[~nans]
            diff_dudt = diff_dudt[~nans]
            u = u[~nans]
        l_2 = (1 - alpha) * self.manifold.inner(xt_1, u, 2 * diff_dudt).mean()
        return l_1 + l_2

    def get_loss_alpha_flow(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            # n_equal = int(0 * x.shape[0])
            eps = self._noise_for(x)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )

        with torch.no_grad():
            # alpha flow edit
            alpha = self.alpha_sample()
            if alpha == 0:
                u, dudt = torch.func.jvp(
                    self.forward,
                    (xt, r, t),
                    (v, torch.zeros_like(r), torch.ones_like(t)),
                )
                u_tgt = v - (t - r) * dudt
            else:
                s = r * alpha + (1 - alpha) * t
                xs, v_s = torch.func.jvp(
                    geodesic, primals=(s,), tangents=(torch.ones_like(s),), has_aux=False
                )
                u_tgt = alpha * v_s + (1 - alpha) * self.forward(xs, s, t)
        diff = self.forward(xt, r, t) - u_tgt.detach()
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        return self.manifold.inner(xt, diff, diff).mean()

    def get_loss_pc_grad(self, x: Tensor):
        if isinstance(x, list):
            x = x[0]
        with torch.no_grad():
            n_equal = int(self.hparams.prop_equal_times * x.shape[0])
            eps = self._noise_for(x)
            r, t = self.sample_r_t(x.shape[0], x.device)
            t = match_dims(t, x.shape)
            r = match_dims(r, x.shape)
            if n_equal > 0:
                # set some times equal to the input time
                r[:n_equal] = t[:n_equal]

            def geodesic(_t: Tensor) -> Tensor:
                return self.manifold.geodesic_interpolant(
                    x, eps, _t
                )  # this is the correct order!

            xt, v = torch.func.jvp(
                geodesic, primals=(t,), tangents=(torch.ones_like(t),), has_aux=False
            )
        u, dudt = torch.func.jvp(
            self.forward,
            (xt, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )
        diff = u - v
        diff_dudt = (t - r) * dudt.detach()
        xt_1 = xt
        if diff.isnan().any():
            nans = diff.isnan().any(dim=(-1, -2))
            xt = xt[~nans]
            diff = diff[~nans]
        l_1 = self.manifold.inner(xt, diff, diff).mean()
        if diff_dudt.isnan().any():
            nans = diff_dudt.isnan().any(dim=(-1, -2))
            xt_1 = xt_1[~nans]
            diff_dudt = diff_dudt[~nans]
            u = u[~nans]
        l_2 = self.manifold.inner(xt_1, u, 2 * diff_dudt).mean()
        return l_1, l_2

    @torch.inference_mode()
    def sample_batch(self, steps: int, sz: int) -> Tensor:
        z = self._prior_samples((sz,) + tuple(self.hparams.in_shape), self.device)
        ts = torch.linspace(0, 1, steps + 1, device=self.device).flip(0)

        for r, t in zip(ts[1:], ts[:-1]):
            t = match_dims(t.expand(sz), z.shape)
            r = match_dims(r.expand(sz), z.shape)
            z = self.manifold.exp(z, -(t - r) * self.forward(z, r, t))
            z = self.manifold.project(z)

        return z

    def name(self) -> str:
        return "mf"

    def forward(self, x: Tensor, s: Tensor, t: Tensor) -> Tensor:
        return self.net(x, s, t)


class AminogfmModule(MeanFlowModule):
    @property
    def val_nll(self):
        return True


class ImageMeanFlowModule(MeanFlowModule):
    @property
    def val_images(self):
        return True
