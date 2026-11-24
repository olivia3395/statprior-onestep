"""
Exit 0 if a finished run (with test_metrics.json) exists whose hydra overrides contain
ALL the given key=value tokens; exit 1 otherwise. Used by run_prior_pilot.sh to skip
runs that already completed, so an interrupted stage can simply be restarted.

    python scripts/is_done.py experiment=meanflow/meanflow_earth_synth seed=0 data.dataset_file=synth_arc30 ...
"""
import glob, os, sys

tokens = [t for t in sys.argv[1:] if t and not t.startswith("~") and not t.startswith("+")
          and not t.startswith("trainer=") and not t.startswith("callbacks.")]
for f in glob.glob(os.path.join("logs", "train", "runs", "*", ".hydra", "overrides.yaml")):
    run_dir = os.path.dirname(os.path.dirname(f))
    if not os.path.exists(os.path.join(run_dir, "test_metrics.json")):
        continue
    have = {line.strip()[2:].strip() for line in open(f) if line.strip().startswith("- ")}
    if all(t in have for t in tokens):
        sys.exit(0)
sys.exit(1)
