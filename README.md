# Start Where the Data Are: Statistical Priors for One-Step Generation on Manifolds

**Yuyao Wang**<sup>1</sup>, **Haoliang Sun**<sup>2</sup>, **Xiankai Lu**<sup>2</sup>, **Xinxin Zhang**<sup>2</sup>, **Yongshun Gong**<sup>2</sup>

<sup>1</sup>Department of Mathematics and Statistics, Boston University  
<sup>2</sup>School of Software, Shandong University  

This repository contains the code for the paper. The method changes only how the noise of a one-step generator is
drawn and paired with the data: a statistical prior is fitted once to the training data, and each data point is paired
with noise from a mixture component drawn from its posterior. Network, objective and sampler are those of each
generator.


| Folder | Content |
|:---|:---|
| `rmf-statprior/` | Riemannian MeanFlow (Zhong et al.) and generalized flow maps (G-LSD) with the statistical prior, the posterior coupling, the hidden Markov prior, data preparation, evaluation and plotting. Main entry point; see its `README.md`. |
| `woo-rmf-statprior/` | The Eulerian, Lagrangian and semigroup formulations of the Riemannian MeanFlow of Woo et al. with the same prior and coupling (JAX). See `README_prior.md`. |

## Quick start

```bash
cd rmf-statprior
conda env create -f environment.yaml && conda activate rmf
pip install pandas pyyaml

# uniform prior (original generator)
python -m rmf.train experiment=meanflow/meanflow_amino_prepro trainer=gpu model.prior_type=uniform

# statistical prior with posterior coupling (paper default: K by BIC up to 32, s = 2)
python -m rmf.train experiment=meanflow/meanflow_amino_prepro trainer=gpu \
    model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0
```

The core of the method is in `rmf-statprior/rmf/models/stat_prior.py` (tangent-space Gaussian mixtures, EM with BIC,
posterior coupling) and `rmf-statprior/rmf/models/hmm_prior.py` (hidden Markov prior with forward filtering and backward
sampling). The table in `rmf-statprior/README.md` maps each section, table and figure of the paper to its run and
evaluation scripts.

## Data

The Earth event catalogs, the Top500 backbone angles and the RNA torsion angles are included. The SO(3) benchmarks are
taken from the Riemannian MeanFlow repository; synthetic data, torsion windows, fragments and side chains are generated
by the scripts listed in `rmf-statprior/README.md`. All datasets are split 80/10/10 with a fixed seed.

## Licence

Code taken from the Riemannian MeanFlow, generalized flow map and Woo et al. repositories remains under the licences of
those repositories.
