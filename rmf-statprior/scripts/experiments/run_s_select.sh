#!/usr/bin/env bash
# Prior width s chosen on the validation split. Fisher, Line, Flood and Pre-Pro already have s in {1, 2} with
# K <= 128 (run_fix.sh); this adds the same two settings for Cone and Swiss roll (seed 0).
# Run from rmf/:  bash scripts/run_s_select.sh > s_select.log 2>&1
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="trainer=gpu data.num_workers=2 model.my_method=mf ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1 trainer.max_epochs=200"
for CS in 1.0 2.0; do
  G="model.prior_type=gmm model.prior_k=128 model.prior_coupling=soft model.prior_cov_scale=$CS"
  echo ">>> s=$CS K<=128 cone";  python -m rmf.train experiment=meanflow/meanflow_so3 data.dataset_file=cone seed=0 $C $G
  echo ">>> s=$CS K<=128 swiss"; python -m rmf.train experiment=meanflow/meanflow_so3_swiss seed=0 $C $G
done
