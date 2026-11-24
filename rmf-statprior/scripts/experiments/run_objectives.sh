#!/usr/bin/env bash
# Does the statistical prior help other one-step objectives? iMF and alpha-Flow (both implemented in the
# RMF code base), uniform vs statistical prior, on SO(3), torus and Earth data. 1 seed.
# Run from rmf/:  nohup bash scripts/run_objectives.sh > objectives.log 2>&1 &
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1"
G="model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
S=${SEED:-0}
for M in ${METHODS:-imf alpha-f}; do
  for D in "meanflow/meanflow_so3 data.dataset_file=cone trainer.max_epochs=200" \
           "meanflow/meanflow_so3_swiss trainer.max_epochs=200" \
           "meanflow/meanflow_amino_prepro trainer.max_epochs=1000" \
           "meanflow/meanflow_earth_fire trainer.max_epochs=100"; do
    EXP=${D%% *}; EXTRA=${D#"$EXP"}
    echo ">>> $M $EXP$EXTRA uniform"; python -m rmf.train experiment=$EXP $EXTRA seed=$S model.my_method=$M $C model.prior_type=uniform
    echo ">>> $M $EXP$EXTRA gmm";     python -m rmf.train experiment=$EXP $EXTRA seed=$S model.my_method=$M $C $G
  done
done
