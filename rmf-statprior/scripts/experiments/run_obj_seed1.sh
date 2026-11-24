#!/usr/bin/env bash
# Second seed for the objectives in Fig. (base vs objective): iMF and alpha-Flow on Cone, Pre-Pro, Fire, and RMF-MT on
# Cone, each with the uniform and the statistical base (posterior coupling).
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
# unique run directory per batch job: two jobs starting in the same second would otherwise collide
C="hydra.run.dir=\${paths.log_dir}/\${task_name}/runs/\${now:%Y-%m-%d}_\${now:%H-%M-%S}_${JOB_ID:-local} trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1 seed=1"
G="model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
CONE="experiment=meanflow/meanflow_so3 data.dataset_file=cone trainer.max_epochs=200"
PREPRO="experiment=meanflow/meanflow_amino_prepro trainer.max_epochs=1000"
FIRE="experiment=meanflow/meanflow_earth_fire trainer.max_epochs=100"
both() { echo ">>> $* uniform"; python -m rmf.train $@ $C model.prior_type=uniform
         echo ">>> $* gmm";     python -m rmf.train $@ $C $G; }
for M in imf alpha-f; do
  both $CONE model.my_method=$M
  both $PREPRO model.my_method=$M
  both $FIRE model.my_method=$M
done
both $CONE model.my_method=pcgrad
