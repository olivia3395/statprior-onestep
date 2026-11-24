"""
Residue-type-conditioned MeanFlow on the flat torus.

Each training example is a 3-vector (phi, psi, class index). The network receives the angles together with a one-hot
encoding of the class (net.d_in = 2 + n_classes, net.d_out = 2). With the statistical prior, one tangent-space Gaussian
mixture is fitted per class, and noise is drawn from the mixture of the requested class, both for the posterior
coupling during training and at generation time. With the uniform prior, the noise is uniform for every class.

The class of the samples being generated is set through `set_condition` (a LongTensor of class indices) before calling
`sample_batch`. Validation and testing are disabled; evaluation is done by scripts/eval_cond.py.
"""
import copy

import torch
import torch.nn.functional as F
from torch import Tensor

from .meanflow_module import AminogfmModule


class CondAminoModule(AminogfmModule):
    def __init__(self, *args, n_classes: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = int(n_classes)
        self._cond = None
        self.cond_priors = None
        if self.structured_prior is not None:
            # one independent copy of the (unfitted) mixture prior per class
            self.cond_priors = torch.nn.ModuleList([copy.deepcopy(self.structured_prior) for _ in range(self.n_classes)])
            self.structured_prior = None  # the per-class priors replace the single prior everywhere below

    # ---------------------------------------------------------------- conditioning
    def set_condition(self, c: Tensor) -> None:
        self._cond = c.long().to(self.device)

    def forward(self, x: Tensor, s: Tensor, t: Tensor) -> Tensor:
        # the inner network sees (angles, one-hot class); its output is projected as in the unconditional wrapper
        onehot = F.one_hot(self._cond, self.n_classes).to(x.dtype)
        inner = getattr(self.net, "net", self.net)
        out = inner(torch.cat([x, onehot], dim=-1), s, t)
        if hasattr(self.net, "manifold"):
            out = self.manifold.project_tangent(x, out)
            if self.net.metric_normalize:
                out = self.manifold.metric_normalize(x, out)
        return out.reshape_as(x)

    def _split(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        self._cond = x[:, -1].long()
        return x[:, :-1]

    def get_loss(self, batch) -> Tensor:
        return super().get_loss(self._split(batch))

    # ---------------------------------------------------------------- priors
    def _noise_for(self, x: Tensor) -> Tensor:
        if self.cond_priors is None:
            return self.manifold.prior(shape=x.shape, device=x.device)
        out = torch.empty_like(x)
        for k in range(self.n_classes):
            idx = self._cond == k
            if idx.any():
                out[idx] = self.cond_priors[k].sample_matched(x[idx]).to(x.dtype)
        return out

    def _prior_samples(self, shape: tuple, device) -> Tensor:
        if self.cond_priors is None:
            return self.manifold.prior(shape=shape, device=device)
        out = torch.empty(shape, device=device)
        for k in range(self.n_classes):
            idx = self._cond == k
            n = int(idx.sum())
            if n:
                out[idx] = self.cond_priors[k].sample(n, device=device).reshape((n,) + tuple(shape[1:])).to(out.dtype)
        return out

    def on_fit_start(self) -> None:
        if self.cond_priors is None or bool(self.cond_priors[0].fitted):
            return
        ds = self.trainer.datamodule.data_fit  # labelled training points without expansion
        pts = torch.stack([torch.as_tensor(ds[i]) for i in range(len(ds))]).float()
        for k in range(self.n_classes):
            pk = pts[pts[:, -1] == k][:, :-1]
            if len(pk) > self.prior_fit_max_points:
                g = torch.Generator().manual_seed(self.prior_fit_seed)
                pk = pk[torch.randperm(len(pk), generator=g)[: self.prior_fit_max_points]]
            self.cond_priors[k].fit(pk.cpu(), seed=self.prior_fit_seed)
            self.cond_priors[k].to(self.device)
            print(f"[cond prior] class {k}: fitted on {len(pk)} points, selected K={int(self.cond_priors[k].n_active)}")

    # ---------------------------------------------------------------- no built-in validation or testing
    def validation_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self) -> None:
        return None

    def test_step(self, batch, batch_idx):
        return None

    def on_test_end(self) -> None:
        return None
