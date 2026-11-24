#!/usr/bin/env bash
# Generalised flow maps (G-LSD) with the uniform and the statistical prior on the remaining real datasets, with the
# budgets of the main experiments of the paper: 100 epochs on the Earth data and 1000 epochs on the torus.
# Uses the same module, overrides and completion check (gfm_done.py) as run_gfm.sh; finished runs are skipped.
# Run from the repository root:  nohup bash scripts/run_gfm_more.sh > gfm_more_train.log 2>&1 &
#   DATASETS  default "earthquake flood volcano Glycine Proline General RNA"   (names as in the data configs)
#   SEEDS     default "0 1"
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1"
GFM="model._target_=gfm.models.StatFlowMapModule model.my_method=mf +model.fm_loss_weight=1.0 +model.eval_nll_exact=false"
G="model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
spec() {   # dataset -> experiment and budget
  case $1 in
    earthquake) echo "meanflow/meanflow_earth_earthquake trainer.max_epochs=100" ;;
    flood)      echo "meanflow/meanflow_earth_flood trainer.max_epochs=100" ;;
    volcano)    echo "meanflow/meanflow_earth_volcano trainer.max_epochs=100" ;;
    Glycine)    echo "meanflow/meanflow_amino_glycine trainer.max_epochs=1000" ;;
    Proline)    echo "meanflow/meanflow_amino_proline trainer.max_epochs=1000" ;;
    General)    echo "meanflow/meanflow_amino_general trainer.max_epochs=1000" ;;
    RNA)        echo "meanflow/meanflow_amino_rna trainer.max_epochs=1000" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}
L=lsd
for S in ${SEEDS:-0 1}; do
  for DS in ${DATASETS:-earthquake flood volcano Glycine Proline General RNA}; do
    REST=$(spec $DS); EXP=${REST%% *}; EXTRA=${REST#"$EXP"}
    for P in uniform gmm; do
      if python scripts/gfm_done.py "$DS" "$P" "$S" "$L"; then
        echo ">>> skip (finished): GFM-$L $DS $P seed=$S"; continue
      fi
      PR="model.prior_type=uniform"; [ $P = gmm ] && PR="$G"
      echo ">>> GFM-$L $EXP$EXTRA $P seed=$S"
      python -m rmf.train experiment=$EXP $EXTRA seed=$S $C $GFM +model.underlying_loss=$L $PR \
          "hydra.run.dir=logs/train/runs/\${now:%Y-%m-%d_%H-%M-%S}_gfm"
    done
  done
done
