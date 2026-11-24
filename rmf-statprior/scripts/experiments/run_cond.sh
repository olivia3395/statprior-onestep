#!/usr/bin/env bash
# Residue-type-conditioned generation: one model for General, Glycine, Proline and Pre-Pro.
# Uniform prior vs one statistical prior per residue type (posterior coupling, s = 2); two seeds each.
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="experiment=meanflow/meanflow_amino_cond trainer=gpu data.num_workers=0 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false ~callbacks.early_stopping callbacks.model_checkpoint.monitor=null callbacks.model_checkpoint.save_last=true +trainer.limit_val_batches=0 test=False trainer.max_epochs=300"
for SEED in 0 1; do
  echo ">>> uniform prior, seed $SEED"
  python -m rmf.train $C seed=$SEED model.prior_type=uniform "hydra.run.dir=logs/train/runs/\${now:%Y-%m-%d_%H-%M-%S}_cond"
  echo ">>> statistical prior, seed $SEED"
  python -m rmf.train $C seed=$SEED model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0 "hydra.run.dir=logs/train/runs/\${now:%Y-%m-%d_%H-%M-%S}_cond"
done
