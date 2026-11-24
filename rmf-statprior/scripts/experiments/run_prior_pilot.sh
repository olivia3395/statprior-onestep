#!/usr/bin/env bash
# Pilot experiment: uniform prior (original RMF) vs structured cluster prior.
# Run from the rmf/ directory:  bash scripts/run_prior_pilot.sh 1   (stage 1)  or 2 (stage 2)
set -e
STAGE=${1:-1}
SEEDS=${SEEDS:-"0 1"}
EPOCHS=${EPOCHS:-300}           # paper uses up to 2000 for Volcano; 300 is enough for a go/no-go
METHOD=${METHOD:-mf}            # mf = plain RMF loss (cleanest comparison); pcgrad = RMF-MT
COMMON="trainer=gpu model.my_method=$METHOD ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true"

run () {  # $1 experiment  $2 extra overrides; skips runs that already finished
  for s in $SEEDS; do
    if python scripts/find_done.py experiment=$1 seed=$s $2; then
      continue
    fi
    echo ">>> $1 seed=$s $2"
    python -m rmf.train experiment=$1 seed=$s $COMMON $2
  done
}

if [ "$STAGE" = "1" ]; then
  # Real data: concentrated (Volcano) vs spread-out (Fire)
  for DS in volcano fire; do
    run meanflow/meanflow_earth_$DS "trainer.max_epochs=$EPOCHS model.prior_type=uniform"
    run meanflow/meanflow_earth_$DS "trainer.max_epochs=$EPOCHS model.prior_type=cluster model.prior_k=16"
  done
fi

if [ "$STAGE" = "2" ]; then
  # Key prediction: the gain from the structured prior grows with data concentration
  python scripts/make_synth_sphere.py --kappas 5 25 100 250
  for K in 5 25 100 250; do
    run meanflow/meanflow_earth_synth "data.dataset_file=synth_k$K model.prior_type=uniform"
    run meanflow/meanflow_earth_synth "data.dataset_file=synth_k$K model.prior_type=cluster model.prior_k=8"
  done
fi

if [ "$STAGE" = "3" ]; then
  # Ablations on Volcano: number of clusters, prior width, coupling
  for K in 4 8 32; do
    run meanflow/meanflow_earth_volcano "trainer.max_epochs=$EPOCHS model.prior_type=cluster model.prior_k=$K"
  done
  for S in 0.25 1.0; do
    run meanflow/meanflow_earth_volcano "trainer.max_epochs=$EPOCHS model.prior_type=cluster model.prior_k=16 model.prior_kappa_scale=$S"
  done
  run meanflow/meanflow_earth_volcano "trainer.max_epochs=$EPOCHS model.prior_type=cluster model.prior_k=16 model.prior_coupling=independent"
fi

if [ "$STAGE" = "4" ]; then
  # Misspecified shapes (NOT vMF): thin rings and long arcs, two widths each (100 and 30 mrad)
  python scripts/make_synth_shapes.py --shapes ring arc --widths 100 30
  for D in synth_ring100 synth_ring30 synth_arc100 synth_arc30; do
    run meanflow/meanflow_earth_synth "data.dataset_file=$D model.prior_type=uniform"
    run meanflow/meanflow_earth_synth "data.dataset_file=$D model.prior_type=cluster model.prior_k=${PRIOR_K:-8}"
  done
fi

if [ "$STAGE" = "5" ]; then
  # Statistical prior: tangent-space Gaussian mixture, EM + BIC, posterior (soft) coupling.
  # Compare with stage 4 (uniform, vMF-cluster) on the same misspecified shapes.
  python scripts/make_synth_shapes.py --shapes ring arc --widths 100 30
  for D in synth_ring100 synth_ring30 synth_arc100 synth_arc30; do
    run meanflow/meanflow_earth_synth "data.dataset_file=$D model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=${COV_SCALE:-2.0}"
  done
fi

if [ "$STAGE" = "6" ]; then
  # SO(3) benchmarks (Table 3 of the RMF paper): uniform prior vs statistical (gmm) prior.
  # The paper's MMD is discriminative on SO(3) (see audit_eval.py), so test/mmd-001 is meaningful here.
  for DS in "meanflow/meanflow_so3 data.dataset_file=cone" "meanflow/meanflow_so3_fisher" \
            "meanflow/meanflow_so3 data.dataset_file=line" "meanflow/meanflow_so3_swiss"; do
    EXP=${DS%% *}; EXTRA=${DS#"$EXP"}
    run $EXP "$EXTRA model.prior_type=uniform"
    run $EXP "$EXTRA model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=${COV_SCALE:-2.0}"
  done
fi

if [ "$STAGE" = "7" ]; then
  # Ablation: division of labour between the statistical prior and the network.
  # (a) prior width: cov_scale 1 (prior = fitted mixture), 4, 8 (cov_scale 2 is stage 5)
  for D in synth_arc30 synth_ring30; do
    for CS in 1.0 4.0 8.0; do
      run meanflow/meanflow_earth_synth "data.dataset_file=$D model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=$CS"
    done
  done
  # (b) larger candidate set for K on the thin ring (BIC hit the upper limit 32 in stage 5)
  run meanflow/meanflow_earth_synth "data.dataset_file=synth_ring30 model.prior_type=gmm model.prior_k=64 model.prior_coupling=soft model.prior_cov_scale=2.0"
fi

if [ "$STAGE" = "8" ]; then
  # Fire (real data) with the statistical prior: 100 epochs x SEEDS, plus 400 epochs x seed 0.
  # Same settings as run_fire_gmm.sh, so runs already finished there are skipped.
  G="model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
  run meanflow/meanflow_earth_fire "trainer.max_epochs=100 ~callbacks.early_stopping $G"
  SEEDS="0" run meanflow/meanflow_earth_fire "trainer.max_epochs=400 ~callbacks.early_stopping $G"
fi

if [ "$STAGE" = "8" ]; then
  # Ablation: coupling between data and prior noise, with the statistical (gmm) prior fixed.
  #   soft        = component k ~ p(k|x) (stage 5 / stage 6 runs, the default)
  #   hard        = MAP component argmax_k p(k|x)
  #   independent = noise drawn from the mixture regardless of x (prior only, no coupling)
  G="model.prior_type=gmm model.prior_k=32 model.prior_cov_scale=2.0"
  python scripts/make_synth_shapes.py --shapes ring arc --widths 30
  for CP in hard independent; do
    run meanflow/meanflow_earth_synth "data.dataset_file=synth_arc30 $G model.prior_coupling=$CP"
    run meanflow/meanflow_earth_synth "data.dataset_file=synth_ring30 $G model.prior_coupling=$CP"
    run meanflow/meanflow_so3 "data.dataset_file=cone $G model.prior_coupling=$CP"
  done
fi

python scripts/collect_results.py
