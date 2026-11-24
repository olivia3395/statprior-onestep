#!/usr/bin/env bash
# Base vs coupling: statistical base with INDEPENDENT coupling (noise drawn from the fitted prior, ignoring the data
# point), same settings as the main runs. Together with the existing uniform + independent and statistical + posterior
# runs this isolates the effect of the base marginal.
#   Part 1 (MeanFlow): Cone seed 1 (seed 0 exists), Pre-Pro seeds 0 1, Fire seeds 0 1.
#   Part 2 (iMF, seed 0): Cone, Pre-Pro, Fire.
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
# unique run directory per batch job: two jobs starting in the same second would otherwise collide
C="hydra.run.dir=\${paths.log_dir}/\${task_name}/runs/\${now:%Y-%m-%d}_\${now:%H-%M-%S}_${JOB_ID:-local} trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1"
IND="model.prior_type=gmm model.prior_k=32 model.prior_coupling=independent model.prior_cov_scale=2.0"
CONE="experiment=meanflow/meanflow_so3 data.dataset_file=cone trainer.max_epochs=200"
PREPRO="experiment=meanflow/meanflow_amino_prepro trainer.max_epochs=1000"
FIRE="experiment=meanflow/meanflow_earth_fire trainer.max_epochs=100"
run() { echo ">>> $*"; python -m rmf.train $@ $C $IND; }
run $CONE   seed=1 model.my_method=mf
run $PREPRO seed=0 model.my_method=mf
run $PREPRO seed=1 model.my_method=mf
run $FIRE   seed=0 model.my_method=mf
run $FIRE   seed=1 model.my_method=mf
run $CONE   seed=0 model.my_method=imf
run $PREPRO seed=0 model.my_method=imf
run $FIRE   seed=0 model.my_method=imf
