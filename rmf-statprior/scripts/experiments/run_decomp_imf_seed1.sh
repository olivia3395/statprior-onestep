#!/usr/bin/env bash
# Second seed for the 'prior only' setting of iMF in the decomposition (statistical prior, independent coupling).
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="hydra.run.dir=logs/train/runs/\${now:%Y-%m-%d_%H-%M-%S}_decomp1 trainer=gpu data.num_workers=0 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1 seed=1 model.my_method=imf"
IND="model.prior_type=gmm model.prior_k=32 model.prior_coupling=independent model.prior_cov_scale=2.0"
run() { echo ">>> $*"; python -m rmf.train $@ $C $IND; }
run experiment=meanflow/meanflow_so3 data.dataset_file=cone trainer.max_epochs=200
run experiment=meanflow/meanflow_amino_prepro trainer.max_epochs=1000
run experiment=meanflow/meanflow_earth_fire trainer.max_epochs=100
