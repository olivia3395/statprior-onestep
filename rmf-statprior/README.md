# Statistical Priors for One-Step Generation on Manifolds

Code for the paper **Start Where the Data Are: Statistical Priors for One-Step Generation on Manifolds**.

Anonymous authors (CVPR 2027 submission).

The code builds on the implementation of Riemannian MeanFlow (Zhong et al., ICML 2026). The network, the training
objectives and the one-step sampler are those of that implementation. We add priors estimated from the training data
and couplings between noise and data that preserve these priors.

## What is new

| File | Content |
|:---|:---|
| `rmf/models/stat_prior.py` | Tangent-space Gaussian mixtures on $S^2$, SO(3) and $\mathbb T^d$, fitted by EM with BIC; posterior, hard, independent and hierarchical-OT couplings |
| `rmf/models/hmm_prior.py` | Hidden Markov prior over sites for products of tori, with posterior coupling by forward filtering and backward sampling |
| `rmf/models/structured_prior.py` | Isotropic von Mises–Fisher mixture prior on $S^2$ (the vMF baseline) |
| `rmf/models/meanflow_module.py` | Prior options for the MeanFlow module (`prior_type`, `prior_coupling`, ...) |
| `rmf/models/cond_module.py` | Conditional generation with one statistical prior per class (residue types) |
| `rmf/models/stat_flowmap_module.py` | Generalised Flow Maps (Davis et al., 2026) with the statistical prior and posterior coupling |
| `rmf/data/amino_datamodule.py` | Torsion windows, residue-type-conditioned data (`cond4`) and side chains (`chi14`) |
| `scripts/` | Data preparation, evaluation, analysis and plotting |
| `scripts/experiments/` | The training runs behind every table and figure |

## Installation

```bash
conda env create -f environment.yaml
conda activate rmf
pip install pandas pyyaml
python -c "import gfm.models as M; print(M.AminogfmModule, M.CondAminoModule)"
```

`gfm` is a symbolic link to the package directory `rmf/`, as in the original repository. All commands below are run
from this directory.

## Data

The Earth datasets (`data/earth_data`), the Top500 dihedral angles (`data/top500`) and the RNA torsion angles
(`data/rna`) are included. The SO(3) benchmarks (`cone`, `fisher24`, `line`, `peak`) are not included because of
their size; copy the `*.npy` files of the original Riemannian MeanFlow repository into `data/raw/`.

The other datasets are generated:

```bash
python scripts/make_synth_sphere.py --kappas 5 25 100 250     # vMF mixtures on S^2
python scripts/make_synth_shapes.py                            # thin rings and arcs on S^2
python scripts/torus_synth.py --dims 2 5 10 20 50              # banana mixtures on T^d
python scripts/make_peptide_windows.py                         # torsion windows and fragments (k = 2, 3, 5, 10, 20)
python scripts/make_sidechain.py                               # side chains (chi1, chi2); downloads PDB entries from RCSB
```

## Training

The prior is selected through Hydra overrides of the MeanFlow module.

```bash
# uniform prior (original Riemannian MeanFlow)
python -m rmf.train experiment=meanflow/meanflow_amino_prepro trainer=gpu model.prior_type=uniform

# statistical prior with posterior coupling (the default of the paper)
python -m rmf.train experiment=meanflow/meanflow_amino_prepro trainer=gpu \
    model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0
```

| Option | Values |
|:---|:---|
| `model.prior_type` | `uniform`, `cluster` (vMF mixture, sphere), `gmm` (tangent-space mixture), `hmm` (products of tori) |
| `model.prior_coupling` | `soft` (posterior), `hard`, `independent`, `hot` (hierarchical OT) |
| `model.prior_k` | maximal number of mixture components |
| `model.prior_select` | `bic` or `fixed` |
| `model.prior_cov_scale` | inflation $s$ of the fitted covariances (paper default: 2) |
| `model.my_method` | objective: `mf` (MeanFlow), `imf` (improved MeanFlow), `alpha-f` ($\alpha$-Flow), `pcgrad` (RMF-MT) |

## Reproducing the paper

The run scripts in `scripts/experiments/` were written for a cluster with one GPU per job and use relative paths.

| Paper | Training | Evaluation |
|:---|:---|:---|
| Controlled study of the uniform prior (Sec. 4.2) | `run_prior_pilot.sh` on the synthetic data | `eval_synth.py`, `analyze_jacobian.py` |
| Comparison with existing generators (Sec. 4.3, Tab. 2--4) | `run_prior_pilot.sh`, `run_objectives.sh`, `run_obj_seed1.sh`, `run_aflow_*.sh`, `run_gfm.sh`, `run_gfm_more.sh` | `eval_real.py`, `eval_manifold.py`, `eval_gfm.py` |
| Structure validation (Sec. 4.3, Fig. 5) | `run_cond.sh`, `run_sidechain.sh` | `downstream_fragments.py`, `eval_sidechain.py` |
| Higher dimensions and structured data (Sec. 4.4, Tab. 5) | `run_dim.sh`, `run_pepwin.sh`, `run_pepwin_hmm.sh`, `run_pepwin_long.sh`, `run_cond.sh` | `eval_torus_synth.py`, `eval_backbone.py`, `eval_cond.py`, `kde_baseline.py` |
| Ablation and sensitivity (Sec. 4.5, Tab. 6) | `run_decomp.sh`, `run_decomp_imf_seed1.sh`, `run_hot.sh`, `run_s_select.sh`, `run_sens.sh`, `run_em_robust.sh` | `eval_real.py`, `eval_synth.py`, `analyze_coupling.py` |
| Computational cost (Supp.) | `run_bench.sh` with `bench_callback.py` | |
| Evaluation protocol (Supp.) | | `audit_eval.py`, `kernel_pd_check.py`, `eval_gamma_ll.py` |

The three formulations of the Riemannian MeanFlow of Woo et al. are in `../woo-rmf-statprior`.

## Figures

`scripts/figures/make_figs.py` draws the figures that only need numbers reported in the tables. Figures that need
trained models are drawn by `scripts/experiments/make_model_figs.sh` and by the `plot_*.py` scripts in `scripts/`
(`plot_folding.py`, `plot_globe.py`, `plot_torus3d.py`, `plot_cond.py`, `plot_fragments.py`, `plot_sidechain.py`).
`scripts/figures/render_fragments.pml` renders the PDB files written by `plot_fragments.py` with PyMOL.

## Licence and citation

The parts of this code that come from the Riemannian MeanFlow repository remain under the licence of that repository.
If you use this code, please cite the paper and Riemannian MeanFlow:

```bibtex
@inproceedings{zhong2026rmf,
  title     = {Riemannian MeanFlow for One-Step Generation on Manifolds},
  author    = {Zhong, Zichen and Sun, Haoliang and Zhao, Yukun and Gong, Yongshun and Yin, Yilong},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```
