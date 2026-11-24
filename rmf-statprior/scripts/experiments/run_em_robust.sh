#!/usr/bin/env bash
# Robustness to an imperfect EM fit (MeanFlow, statistical prior, posterior coupling, s = 2). The runs are kept in a
# separate directory (logs/train/runs_emrob) so that they never mix with the main experiments.
#   em3:  EM stopped after 3 iterations          fit500: prior fitted on 500 training points
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="trainer=gpu data.num_workers=0 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1 seed=0 model.my_method=mf model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
for D in "Pre-Pro|experiment=meanflow/meanflow_amino_prepro trainer.max_epochs=1000" "fire|experiment=meanflow/meanflow_earth_fire trainer.max_epochs=100"; do
  N=${D%%|*}; E=${D#*|}
  echo ">>> $N, EM stopped after 3 iterations"
  python -m rmf.train $E $C model.prior_em_iters=3 "hydra.run.dir=logs/train/runs_emrob/\${now:%Y-%m-%d_%H-%M-%S}_em3"
  echo ">>> $N, prior fitted on 500 points"
  python -m rmf.train $E $C model.prior_fit_max_points=500 "hydra.run.dir=logs/train/runs_emrob/\${now:%Y-%m-%d_%H-%M-%S}_fit500"
done
