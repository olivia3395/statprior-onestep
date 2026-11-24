#!/usr/bin/env bash
# Statistical prior with Generalised Flow Maps (Davis et al., 2026): uniform vs statistical prior on Cone, Pre-Pro and
# Fire, with the same networks, data and budgets as the comparison of objectives (run_objectives.sh).
# LOSSES selects the GFM self-distillation losses (default: lsd; all three: LOSSES="lsd esd psd").
# Finished runs are skipped, so the script can simply be restarted after an interruption.
# Run from the repository root:  nohup bash scripts/run_gfm.sh > gfm_train.log 2>&1 &
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1"
GFM="model._target_=gfm.models.StatFlowMapModule model.my_method=mf +model.fm_loss_weight=1.0 +model.eval_nll_exact=false"
G="model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
for S in ${SEEDS:-0 1}; do
  for L in ${LOSSES:-lsd}; do
    for D in "cone|meanflow/meanflow_so3 data.dataset_file=cone trainer.max_epochs=200" \
             "Pre-Pro|meanflow/meanflow_amino_prepro trainer.max_epochs=1000" \
             "fire|meanflow/meanflow_earth_fire trainer.max_epochs=100"; do
      DS=${D%%|*}; REST=${D#*|}; EXP=${REST%% *}; EXTRA=${REST#"$EXP"}
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
done
