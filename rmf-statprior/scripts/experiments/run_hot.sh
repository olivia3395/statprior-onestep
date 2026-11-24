#!/usr/bin/env bash
# Hierarchical-OT coupling ("hot"): posterior component + optimal matching within each component.
# Compare with the soft-coupling runs of stages 5-7 (same datasets, same prior settings).
# Run from rmf/:  SEEDS="0" nohup bash scripts/run_hot.sh > hot.log 2>&1 &
set -e
SEEDS=${SEEDS:-"0"}
C="trainer=gpu model.my_method=mf ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true"
H="model.prior_type=gmm model.prior_coupling=hot"
run () { for s in $SEEDS; do echo ">>> $1 seed=$s $2"; python -m rmf.train experiment=$1 seed=$s $C $2; done; }

# 1) synthetic sphere: the cases where the network hurt an accurate prior
run meanflow/meanflow_earth_synth "data.dataset_file=synth_arc30  $H model.prior_k=32 model.prior_cov_scale=1.0"
run meanflow/meanflow_earth_synth "data.dataset_file=synth_arc30  $H model.prior_k=32 model.prior_cov_scale=2.0"
run meanflow/meanflow_earth_synth "data.dataset_file=synth_ring30 $H model.prior_k=32 model.prior_cov_scale=2.0"
# 2) SO(3) benchmarks
run meanflow/meanflow_so3        "data.dataset_file=cone $H model.prior_k=32 model.prior_cov_scale=2.0"
run meanflow/meanflow_so3_fisher "$H model.prior_k=32 model.prior_cov_scale=2.0"
run meanflow/meanflow_so3        "data.dataset_file=line $H model.prior_k=32 model.prior_cov_scale=2.0"
run meanflow/meanflow_so3_swiss  "$H model.prior_k=32 model.prior_cov_scale=2.0"
# 3) real data (Fire), same budget as the other Fire runs
run meanflow/meanflow_earth_fire "trainer.max_epochs=100 ~callbacks.early_stopping $H model.prior_k=32 model.prior_cov_scale=2.0"
