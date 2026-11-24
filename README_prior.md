# Statistical prior for Riemannian MeanFlow (Woo et al., ICML 2026)

This copy of the official JAX code adds the statistical prior and posterior coupling of "Start Where the Data Are:
Statistical Priors for One-Step Generation on Manifolds". Without `+prior_type=gmm` the code behaves exactly like
the original.

| File | Change |
|:---|:---|
| `src/utils/stat_prior.py` | Tangent-space Gaussian mixtures (copied from the RMF repository, PyTorch) |
| `src/utils/stat_prior_jax.py` | `StatPrior` with the interface of `UniformPrior` plus `couple(x_1)`; `make_prior` |
| `src/experiments/flowmap_learning/mean_flow.py` | Prior constructed by `make_prior`; `export_samples` writes 1/2/4-step samples |
| `src/experiments/flowmap_learning/{mean_flow,lagrange,split_flow}.py` | Training noise from `couple(x_1)` when available (8 sites) |
| `src/experiments/base_class.py` | Calls `export_samples` after training |
| `run_woo.sh`, `run_woo_more.sh` | Eulerian/Lagrangian/semigroup x nine real datasets x uniform/statistical x two seeds |
| `woo_eval.py` | Evaluation with the kernels of the RMF repository (run in the RMF environment: `python woo_eval.py --rmf_root ../rmf-statprior`) |

Noise is drawn outside the jitted training step, so the prior runs in PyTorch on the CPU and is passed to JAX as arrays.
