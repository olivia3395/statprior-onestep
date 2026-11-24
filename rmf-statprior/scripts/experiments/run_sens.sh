#!/usr/bin/env bash
# Sensitivity to the number of mixture components K and of hidden-Markov states S (seed 0). The values selected by BIC
# in the main experiments (Pre-Pro K=8, Fire K=32, fragments S=32) are covered by the existing runs.
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="hydra.run.dir=logs/train/runs/\${now:%Y-%m-%d_%H-%M-%S}_sens trainer=gpu data.num_workers=0 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1 seed=0 model.my_method=mf"
FIX="model.prior_select=fixed model.prior_coupling=soft model.prior_cov_scale=2.0"
for K in 2 4 16 32; do
  echo ">>> Pre-Pro K=$K"; python -m rmf.train experiment=meanflow/meanflow_amino_prepro trainer.max_epochs=1000 $C model.prior_type=gmm model.prior_k=$K $FIX
done
for K in 8 16 64; do
  echo ">>> Fire K=$K"; python -m rmf.train experiment=meanflow/meanflow_earth_fire trainer.max_epochs=100 $C model.prior_type=gmm model.prior_k=$K $FIX
done
for S in 8 16 48; do
  echo ">>> fragments k=10 S=$S"
  python -m rmf.train experiment=meanflow/meanflow_torus_synth data.data_dir=data/torus_windows data.data_type=pepwin_k10 \
    model.net.d_in=20 model.in_shape=[20] trainer.max_epochs=500 $C model.prior_type=hmm model.prior_k=$S $FIX \
    model.prior_fit_max_points=6000
done
