"""
Generalised Flow Maps (Davis et al., 2026) with a statistical prior.

The network, the self-distillation losses (Lagrangian, Eulerian, Progressive) and the sampler are those of
FlowMapModule. Only two things change, as in the MeanFlow module of this repository:

  * training: the noise x0 paired with a data point x is drawn by the coupling of the fitted prior (posterior
    coupling by default), instead of independently from the uniform distribution;
  * sampling: starting points are drawn from the fitted prior.

With prior_type="uniform" the module behaves exactly like FlowMapModule. MeanFlow-specific arguments of the shared
experiment configs (prop_equal_times, time_distribution, prior_kappa_scale) are accepted and ignored, so that the
MeanFlow experiment configs can be reused with model._target_ set to this class.
"""
import json
import os

import torch
from torch import Tensor

from gfm.manifold import ManifoldType
from gfm.models.flowmap_module import FlowMapModule
from gfm.models.stat_prior import SO3TangentGMM, SphereTangentGMM, TorusTangentGMM


class StatFlowMapModule(FlowMapModule):
    def __init__(
        self,
        prior_type: str = "uniform",
        prior_k: int = 32,
        prior_coupling: str = "soft",
        prior_fit_seed: int = 0,
        prior_select: str = "bic",
        prior_cov_scale: float = 2.0,
        prior_fit_max_points: int = 10000,
        prior_em_iters: int = 100,
        prop_equal_times: float = 0.0,        # MeanFlow-only, ignored
        time_distribution: str = "uniform",   # MeanFlow-only, ignored
        prior_kappa_scale: float = 0.5,       # vMF-only, ignored
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert prior_type in ("uniform", "gmm"), prior_type
        assert prior_coupling in ("soft", "hard", "independent", "hot", "matched"), prior_coupling
        self.prior_type = prior_type
        self.prior_coupling = "soft" if prior_coupling == "matched" else prior_coupling
        self.prior_fit_seed = prior_fit_seed
        self.prior_fit_max_points = prior_fit_max_points
        self.structured_prior = None
        self._pair_with = None
        if prior_type == "gmm":
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
            self.structured_prior = gmm_cls(**extra, max_k=prior_k, n_iter=prior_em_iters, k_candidates=cands,
                                            select=prior_select, cov_scale=prior_cov_scale,
                                            coupling=self.prior_coupling)

    # ------------------------------------------------------------------ noise
    @torch.no_grad()
    def sample_prior(self, shape: tuple, device: torch.device) -> Tensor:
        if self.structured_prior is None:
            return self.manifold.prior(shape=shape, device=device)
        if self._pair_with is not None:                       # training: coupling with the current batch
            return self.structured_prior.sample_matched(self._pair_with).to(self._pair_with.dtype)
        out = self.structured_prior.sample(shape[0], device=device)   # generation: fresh prior samples
        if isinstance(self.structured_prior, SO3TangentGMM):
            return out
        return out.reshape(shape)

    def get_loss(self, x):
        self._pair_with = x[0] if isinstance(x, list) else x
        try:
            return super().get_loss(x)
        finally:
            self._pair_with = None

    # ------------------------------------------------------------------ prior fitting
    def _training_points(self) -> Tensor:
        ds = self.trainer.datamodule.data_train
        ds = getattr(ds, "dset", ds)                          # unwrap ExpandDataset
        pts = torch.stack([torch.as_tensor(ds[i]) for i in range(len(ds))]).float()
        if self.manifold.get_type() == ManifoldType.SO3:
            return pts
        return pts[..., : int(self.hparams.in_shape[-1])]

    def on_fit_start(self) -> None:
        if self.structured_prior is None or bool(self.structured_prior.fitted):
            return
        pts = self._training_points()
        if len(pts) > self.prior_fit_max_points:
            g = torch.Generator().manual_seed(self.prior_fit_seed)
            pts = pts[torch.randperm(len(pts), generator=g)[: self.prior_fit_max_points]]
        self.structured_prior.fit(pts.cpu(), seed=self.prior_fit_seed)
        self.structured_prior.to(self.device)
        sel = ", ".join(f"K={k}:{c:.0f}" for k, c in self.structured_prior.selection_log)
        print(f"[gmm prior, GFM] fitted on {len(pts)} points: selected K={int(self.structured_prior.n_active)} "
              f"({self.structured_prior.select}; {sel})")

    # ------------------------------------------------------------------ results file read by the evaluation scripts
    def on_test_end(self) -> None:
        out = {k: float(v) for k, v in self.trainer.callback_metrics.items() if k.startswith("test/")}
        out.update(
            prior_type=self.prior_type,
            prior_coupling=self.prior_coupling,
            prior_k=int(self.hparams.get("prior_k", 0)),
            prior_select=str(self.hparams.get("prior_select", "")),
            prior_cov_scale=float(self.hparams.get("prior_cov_scale", 0.0)),
            prior_k_selected=int(self.structured_prior.n_active) if self.structured_prior is not None else None,
            my_method=f"gfm-{self.hparams.underlying_loss}",
            dataset=str(getattr(self.trainer.datamodule.hparams, "dataset_file", None)
                        or getattr(self.trainer.datamodule.hparams, "data_type", "")),
        )
        path = os.path.join(self.trainer.default_root_dir, "test_metrics.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[results] wrote {path}")

    def name(self) -> str:
        return f"flowmap-{self.hparams.underlying_loss}-{self.prior_type}"
