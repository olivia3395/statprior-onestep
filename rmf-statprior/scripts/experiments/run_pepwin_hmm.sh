#!/usr/bin/env bash
# Hidden-Markov prior on backbone fragments of 10 and 20 residues (T^20, T^40). Same setting as
# run_pepwin_long.sh (uniform and flat-mixture priors), so the three priors can be compared directly.
# Run from rmf/:  nohup bash scripts/run_pepwin_hmm.sh > pepwin_hmm.log 2>&1 &
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
[ -f data/torus_windows/pepwin_k20.npy ] || python scripts/make_peptide_windows.py --ks 10 20
C="trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.max_epochs=${EPOCHS:-500}"
H="model.prior_type=hmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0 model.prior_fit_max_points=6000"
for S in ${SEEDS:-0 1}; do
  for K in ${KS:-10 20}; do
    D=$((2*K))
    echo ">>> fragments k=$K (T^$D) seed=$S hmm"
    python -m rmf.train experiment=meanflow/meanflow_torus_synth data.data_dir=data/torus_windows data.data_type=pepwin_k$K \
      model.net.d_in=$D model.in_shape=[$D] seed=$S $C $H
  done
done
