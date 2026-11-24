#!/usr/bin/env bash
# Computational overhead of the prior: 220 training steps per setting (20 warm-up steps) with the StepTimer callback,
# without validation or testing. The [bench] lines are collected in bench_summary.txt.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$PWD/scripts:$PYTHONPATH
B="hydra.run.dir=logs/bench/\${now:%Y-%m-%d_%H-%M-%S} trainer=gpu data.num_workers=0 seed=0 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false ~callbacks.early_stopping ~callbacks.model_checkpoint ++trainer.max_steps=220 trainer.max_epochs=100000 +trainer.limit_val_batches=0 test=False +callbacks.bench._target_=bench_callback.StepTimer"
G="model.prior_type=gmm model.prior_k=32 model.prior_cov_scale=2.0"
run() { echo ">>> $1"; shift; python -m rmf.train "$@" $B 2>&1 | grep -E "^\[bench|^\[gmm prior\]|^\[hmm prior\]|Error" ; }
for D in "fire|experiment=meanflow/meanflow_earth_fire" "cone|experiment=meanflow/meanflow_so3 data.dataset_file=cone" "Pre-Pro|experiment=meanflow/meanflow_amino_prepro"; do
  N=${D%%|*}; E=${D#*|}
  run "$N uniform" $E model.prior_type=uniform
  run "$N statistical (posterior coupling)" $E $G model.prior_coupling=soft
  run "$N statistical (hierarchical OT)" $E $G model.prior_coupling=hot
done
F="experiment=meanflow/meanflow_torus_synth data.data_dir=data/torus_windows data.data_type=pepwin_k10 model.net.d_in=20 model.in_shape=[20]"
run "fragments k=10 uniform" $F model.prior_type=uniform
run "fragments k=10 hidden-Markov (posterior coupling)" $F model.prior_type=hmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0 model.prior_fit_max_points=6000
