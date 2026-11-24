"""
Exit 0 if a finished run (test_metrics.json present) exists whose hydra overrides contain
ALL the given override tokens; exit 1 otherwise. Used by run_prior_pilot.sh to skip runs
that already completed (e.g. when resubmitting after a session expired).

    python scripts/find_done.py experiment=meanflow/meanflow_earth_synth seed=0 data.dataset_file=synth_arc30 ...
"""
import glob, os, sys
import yaml

want = [t for t in sys.argv[1:] if t]
for f in glob.glob("logs/train/runs/*/.hydra/overrides.yaml"):
    run = os.path.dirname(os.path.dirname(f))
    if not os.path.exists(os.path.join(run, "test_metrics.json")):
        continue
    have = set(yaml.safe_load(open(f)) or [])
    if all(t in have for t in want):
        print(f"[skip] already done in {run}")
        sys.exit(0)
sys.exit(1)
