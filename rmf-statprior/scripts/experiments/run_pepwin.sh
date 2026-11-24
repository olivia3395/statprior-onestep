#!/usr/bin/env bash
# Real higher-dimensional torus data: torsions of k consecutive residues (T^4, T^6, T^10).
# Run from rmf/:  nohup bash scripts/run_pepwin.sh > pepwin.log 2>&1 &
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
[ -f data/torus_windows/pepwin_k5.npy ] || python scripts/make_peptide_windows.py --ks 2 3 5
C="trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.max_epochs=${EPOCHS:-500}"
G="model.prior_type=gmm model.prior_k=64 model.prior_coupling=soft model.prior_cov_scale=2.0"
for S in ${SEEDS:-0 1}; do
  for K in ${KS:-2 3 5}; do
    D=$((2*K))
    X="experiment=meanflow/meanflow_torus_synth data.data_dir=data/torus_windows data.data_type=pepwin_k$K model.net.d_in=$D model.in_shape=[$D] seed=$S"
    echo ">>> pepwin k=$K (T^$D) seed=$S uniform"; python -m rmf.train $X $C model.prior_type=uniform
    echo ">>> pepwin k=$K (T^$D) seed=$S gmm";     python -m rmf.train $X $C $G
  done
done
