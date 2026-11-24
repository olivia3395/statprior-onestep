#!/usr/bin/env bash
# alpha-Flow tuned equally for both priors: curriculum length model.epoch (the transition is centred at model.epoch/2)
# and, on Fire, the training budget. Seed 0. Existing runs with the default curriculum (model.epoch=100) complete the
# candidate set. Part d.
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} TORCHINDUCTOR_COMPILE_THREADS=1
C="hydra.run.dir=logs/train/runs/\${now:%Y-%m-%d_%H-%M-%S}_aflowd trainer=gpu data.num_workers=2 ~callbacks.rich_progress_bar +trainer.enable_progress_bar=false callbacks.model_checkpoint.save_last=true ~callbacks.early_stopping trainer.min_epochs=1 seed=0 model.my_method=alpha-f"
G="model.prior_type=gmm model.prior_k=32 model.prior_coupling=soft model.prior_cov_scale=2.0"
both() { echo ">>> $* uniform"; python -m rmf.train $@ $C model.prior_type=uniform
         echo ">>> $* statistical"; python -m rmf.train $@ $C $G; }
one() { echo ">>> $* uniform"; python -m rmf.train $@ $C model.prior_type=uniform; }
one experiment=meanflow/meanflow_earth_fire trainer.max_epochs=400 model.epoch=400
