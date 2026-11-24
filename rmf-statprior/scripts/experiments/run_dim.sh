#!/usr/bin/env bash
# Dimension-scaling study on synthetic torus mixtures (exact densities).
# Run from rmf/:  nohup bash scripts/run_dim.sh > dim.log 2>&1 &
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
python scripts/torus_synth.py --dims ${DIMS:-2 5 10 20 50}
C="trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.max_epochs=${EPOCHS:-500}"
G="model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
for S in ${SEEDS:-0 1}; do
  for D in ${DIMS:-2 5 10 20 50}; do
    X="experiment=meanflow/meanflow_torus_synth data.data_type=tsynth_d$D model.net.d_in=$D model.in_shape=[$D] seed=$S"
    echo ">>> d=$D seed=$S uniform"; python -m rmf.train $X $C model.prior_type=uniform
    echo ">>> d=$D seed=$S gmm";     python -m rmf.train $X $C $G
  done
done
