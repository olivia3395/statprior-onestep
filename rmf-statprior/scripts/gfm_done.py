"""
Exit 0 if a GFM run with the given dataset, prior, seed and loss has finished (its test_metrics.json exists), else 1.
Used by run_gfm.sh to skip finished runs when it is restarted.

    python scripts/gfm_done.py <dataset> <prior_type> <seed> <loss>
"""
import glob, os, sys
from omegaconf import OmegaConf

ds, prior, seed, loss = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
for cf in glob.glob("logs/train/runs/*_gfm/.hydra/config.yaml"):
    run = os.path.dirname(os.path.dirname(cf))
    if not os.path.exists(os.path.join(run, "test_metrics.json")):
        continue
    c = OmegaConf.load(cf)
    d = str(c.data.get("dataset_file", None) or c.data.get("data_type", ""))
    if d == ds and str(c.model.get("prior_type")) == prior and int(c.seed) == seed \
            and str(c.model.get("underlying_loss")) == loss:
        sys.exit(0)
sys.exit(1)
