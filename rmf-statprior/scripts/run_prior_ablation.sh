#!/usr/bin/env bash
# Go/no-go experiment for the structured-prior idea.
#
# Three settings, everything else identical:
#   uniform   : original RMF (uniform prior, independent pairing)      <- baseline
#   mix_indep : vMF-mixture prior, independent pairing                 <- effect of the prior alone
#   mix_clust : vMF-mixture prior, cluster-matched pairing             <- full method
#
# Usage (from the rmf/ directory):
#   bash scripts/run_prior_ablation.sh                      # defaults below
#   DATASETS="volcano fire" SEEDS="0 1 2" EPOCHS=300 METHOD=mf bash scripts/run_prior_ablation.sh
#   python scripts/summarize_runs.py                        # table of results
#
# EXTRA:  any additional hydra overrides, e.g. EXTRA="data.num_workers=8"
# METHOD: mf (plain RMF loss, recommended to isolate the prior effect) or pcgrad (RMF-MT).
# EPOCHS: the paper trains 600-2000 epochs; a shorter budget is fine for a relative
#         comparison as long as every setting gets the same budget.

set -euo pipefail

DATASETS=${DATASETS:-"volcano earthquake flood fire"}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-300}
METHOD=${METHOD:-mf}
K=${K:-8}
KAPPA_SCALE=${KAPPA_SCALE:-0.5}
TRAINER=${TRAINER:-gpu}
SETTINGS=${SETTINGS:-"uniform mix_indep mix_clust"}
OUT=${OUT:-logs/prior_ablation}

for DS in $DATASETS; do
  case "$DS" in
    volcano|earthquake|flood|fire) EXP="meanflow/meanflow_earth_${DS}" ;;
    *)                             EXP="meanflow/meanflow_earth_fire" ;;   # synthetic sets
  esac
  for SETTING in $SETTINGS; do
    case "$SETTING" in
      uniform)   PRIOR="model.prior_type=uniform" ;;
      mix_indep) PRIOR="model.prior_type=vmf_mixture model.prior_kwargs.coupling=independent" ;;
      mix_clust) PRIOR="model.prior_type=vmf_mixture model.prior_kwargs.coupling=cluster" ;;
      *) echo "unknown setting $SETTING"; exit 1 ;;
    esac
    for SEED in $SEEDS; do
      RUN_DIR="${OUT}/${DS}/${SETTING}_K${K}_ks${KAPPA_SCALE}_${METHOD}/seed${SEED}"
      if [ -f "${RUN_DIR}/metrics.json" ]; then echo "skip ${RUN_DIR}"; continue; fi
      echo "=== ${DS} | ${SETTING} | seed ${SEED} ==="
      python -m rmf.train experiment="${EXP}" trainer="${TRAINER}" seed="${SEED}" \
        data.dataset_file="${DS}" model.my_method="${METHOD}" \
        trainer.max_epochs="${EPOCHS}" trainer.min_epochs="${EPOCHS}" \
        ${PRIOR} model.prior_kwargs.n_components="${K}" model.prior_kwargs.kappa_scale="${KAPPA_SCALE}" \
        model.prior_kwargs.seed="${SEED}" \
        hydra.run.dir="${RUN_DIR}" ${EXTRA:-}
    done
  done
done
