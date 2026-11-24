#!/usr/bin/env bash
# Woo et al. (Eulerian rmf, Lagrangian lmf, semigroup smf) with the uniform and the statistical prior on the remaining
# real datasets: Earthquake, Flood, Volcano (S^2), Glycine, Proline, General (T^2) and RNA (T^7). The authors'
# configurations are used unchanged except for the Earth training length, which is set to that of the Fire runs.
# Each run writes samples.npz (1, 2 and 4 steps) into its directory; finished runs are skipped.
#
# Environment variables (all optional):
#   VARIANTS      default "rmf lmf smf"
#   DATASETS      default "earthquake flood volcano glycine proline general rna"
#   SEEDS         default "0 1"
#   EARTH_EPOCHS  default 100: MUST equal num_epochs of the Fire runs (see README)
#   AMINO_EPOCHS  default empty = the authors' 10000 epochs on T^2
set -e
export WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false
EARTH_EPOCHS=${EARTH_EPOCHS:-100}
AE=""; [ -n "$AMINO_EPOCHS" ] && AE="num_epochs=$AMINO_EPOCHS"
spec() {   # dataset -> "task|extra hydra overrides"
  case $1 in
    earthquake) echo "earth|data=earthsquake num_epochs=$EARTH_EPOCHS" ;;   # the authors' file is called earthsquake.yaml
    flood)      echo "earth|data=flood num_epochs=$EARTH_EPOCHS" ;;
    volcano)    echo "earth|data=volcano num_epochs=$EARTH_EPOCHS" ;;
    glycine)    echo "amino|data.train.amino=Glycine data.val.amino=Glycine $AE" ;;
    proline)    echo "amino|data.train.amino=Proline data.val.amino=Proline $AE" ;;
    general)    echo "amino|data.train.amino=General data.val.amino=General $AE" ;;
    rna)        echo "rna|" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}
for S in ${SEEDS:-0 1}; do
  for V in ${VARIANTS:-rmf lmf smf}; do
    for NAME in ${DATASETS:-earthquake flood volcano glycine proline general rna}; do
      SP=$(spec $NAME); TASK=${SP%%|*}; EXTRA=${SP#*|}
      for P in uniform gmm; do
        OUT=runs/${V}_${NAME}_${P}_s${S}
        if [ -f "$OUT/samples.npz" ]; then echo ">>> skip (finished): $OUT"; continue; fi
        PR=""; [ "$P" = gmm ] && PR="+prior_type=gmm +prior_k=32 +prior_cov_scale=2.0 +prior_coupling=soft"
        echo ">>> $OUT  (experiment=${V}_${TASK} $EXTRA)"
        uv run main.py experiment=${V}_${TASK} $EXTRA seed=$S $PR hydra.run.dir=$OUT
      done
    done
  done
done
