"""
Lightning callback that measures the computational overhead of the prior:
  - time to fit the prior (between on_fit_start of the callback and on_train_start, i.e. the module's prior fit),
  - mean wall-clock time per training step (after a warm-up) and peak GPU memory,
  - time of the noise pairing for one minibatch: posterior (soft) coupling and hierarchical OT with several block sizes.
Prints one line starting with [bench] and, for mixture priors, one line starting with [bench-coupling].
"""
import time

import torch
from lightning.pytorch.callbacks import Callback


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StepTimer(Callback):
    def __init__(self, warmup: int = 20, chunks=(32, 64, 128, 256, 512), reps: int = 5):
        self.warmup, self.chunks, self.reps = warmup, tuple(chunks), reps
        self.times, self.t_fit0, self.prior_fit_s, self.t0 = [], None, 0.0, None

    def on_fit_start(self, trainer, pl_module):
        self.t_fit0 = time.perf_counter()

    def on_train_start(self, trainer, pl_module):
        self.prior_fit_s = time.perf_counter() - self.t_fit0
        self._coupling_bench(trainer, pl_module)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        _sync(); self.t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _sync(); self.times.append(time.perf_counter() - self.t0)

    def on_train_end(self, trainer, pl_module):
        t = self.times[self.warmup:] or self.times
        mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
        hp = pl_module.hparams
        sp = getattr(pl_module, "structured_prior", None)
        k = int(sp.n_active) if sp is not None and hasattr(sp, "n_active") else 0
        print(f"[bench] prior={hp.get('prior_type')} coupling={hp.get('prior_coupling')} K={k} "
              f"prior_fit_s={self.prior_fit_s:.2f} step_ms={1e3 * sum(t) / len(t):.2f} "
              f"peak_mem_mb={mem:.0f} steps={len(t)}", flush=True)

    @torch.no_grad()
    def _coupling_bench(self, trainer, pl_module):
        sp = getattr(pl_module, "structured_prior", None)
        if sp is None or not hasattr(sp, "ot_chunk"):
            return
        batch = next(iter(trainer.datamodule.train_dataloader()))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(pl_module.device)
        old = (sp.coupling, sp.ot_chunk)
        res = []
        for name, chunk in [("soft", None)] + [("hot", c) for c in self.chunks]:
            sp.coupling = name
            if chunk is not None:
                sp.ot_chunk = chunk
            ts = []
            for _ in range(self.reps):
                _sync(); t0 = time.perf_counter(); pl_module._noise_for(x); _sync()
                ts.append(time.perf_counter() - t0)
            res.append(f"{name}{'' if chunk is None else chunk}={1e3 * sorted(ts)[len(ts) // 2]:.1f}ms")
        sp.coupling, sp.ot_chunk = old
        print(f"[bench-coupling] batch={x.shape[0]} K={int(sp.n_active)} " + " ".join(res), flush=True)
