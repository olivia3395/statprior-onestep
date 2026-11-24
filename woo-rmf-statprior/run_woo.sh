#!/usr/bin/env bash
# Uniform vs statistical prior with the three formulations of Woo et al. (rmf: Eulerian, lmf: Lagrangian,
# smf: semigroup) on Fire and Pre-Pro, two seeds. The authors' configurations are used unchanged; only the prior is
# added. Each run writes samples.npz (1, 2 and 4 steps) into its directory; finished runs are skipped.
set -e
export WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false
for S in ${SEEDS:-0 1}; do
  for V in ${VARIANTS:-rmf lmf smf}; do
    for D in "fire|${V}_earth data=fire" "prepro|${V}_amino data.train.amino=Pre-Pro data.val.amino=Pre-Pro"; do
      NAME=${D%%|*}; ARGS=${D#*|}; EXP=${ARGS%% *}; EXTRA=${ARGS#"$EXP"}
      for P in uniform gmm; do
        OUT=runs/${V}_${NAME}_${P}_s${S}
        if [ -f "$OUT/samples.npz" ]; then echo ">>> skip (finished): $OUT"; continue; fi
        PR=""; [ "$P" = gmm ] && PR="+prior_type=gmm +prior_k=32 +prior_cov_scale=2.0 +prior_coupling=soft"
        echo ">>> $OUT"
        uv run main.py experiment=$EXP $EXTRA seed=$S $PR hydra.run.dir=$OUT
      done
    done
  done
done
