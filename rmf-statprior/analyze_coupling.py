"""
Does the posterior coupling preserve the prior in practice?

For every statistical-prior model (MeanFlow, s = 2, posterior coupling) the script compares
  - the mixture weights w_k with the average posterior responsibilities on the training data
    (equal at a fixed point of EM): total-variation distance and largest absolute difference;
  - the noise produced by the coupling for the training data with fresh samples from the prior,
    by MMD^2 at the finest kernel scale, relative to the MMD^2 between two fresh prior samples (noise floor).
EM settings (number of iterations, number of fitting points) are reported, so that models fitted with an
imperfect EM (logs/train/runs_emrob) can be compared with the main models.

    python scripts/analyze_coupling.py --roots logs/train/runs logs/train/runs_emrob --csv coupling.csv
"""
import argparse, glob, json, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
import eval_manifold as EM
import eval_real as ER
from eval_synth import load_model

SPACE = {**{n: "sphere" for n in ["volcano", "earthquake", "flood", "fire"]},
         **{n: "so3" for n in ["cone", "fisher24", "line", "swiss"]},
         **{n: "torus" for n in ["General", "Glycine", "Proline", "Pre-Pro", "RNA"]}}
GAMMA = {"sphere": 50, "so3": 20, "torus": 10}


def mmd(x, y, space):
    if space == "sphere":
        return ER.mmd2_unbiased(x, y, "lap", GAMMA[space])
    return EM.mmd2(x, y, space, "lap", GAMMA[space])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["logs/train/runs", "logs/train/runs_emrob"])
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--csv", default="coupling.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rows = []
    for root in a.roots:
        for f in sorted(glob.glob(os.path.join(root, "*", "test_metrics.json"))):
            meta = json.load(open(f)); name = meta.get("dataset")
            if name not in SPACE or meta.get("prior_type") != "gmm":
                continue
            run = os.path.dirname(f); cfg = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
            m = cfg.model
            if str(m.get("my_method", "mf")) != "mf" or str(m.get("prior_coupling")) != "soft" \
                    or float(m.get("prior_cov_scale", 2.0)) != 2.0 or int(m.get("prior_k", 32)) > 32 \
                    or str(m.get("prior_select", "bic")) != "bic":
                continue
            space = SPACE[name]
            model, _, cfg = load_model(run, dev, "last")
            sp = model.structured_prior
            train, _, _ = (ER.split_tensors(cfg) if space == "sphere" else EM.split_tensors(cfg))
            x = train[torch.randperm(len(train))[: a.n]].to(dev).float()
            with torch.no_grad():
                k = int(sp.n_active)
                w = torch.exp(sp.log_weights[:k].double()); w = w / w.sum()
                r = sp.responsibilities(sp._flat(x)).double().mean(0)[:k]
                tv = 0.5 * (w - r).abs().sum().item(); mx = (w - r).abs().max().item()
                vals, floor = [], []
                for _ in range(a.reps):
                    eps = sp.sample_matched(x).float()
                    fresh = sp.sample(x.shape[0], device=dev).reshape(eps.shape).float()
                    fresh2 = sp.sample(x.shape[0], device=dev).reshape(eps.shape).float()
                    vals.append(mmd(eps, fresh, space)); floor.append(mmd(fresh2, fresh, space))
            row = dict(dataset=name, K=k, em_iters=int(m.get("prior_em_iters", 100)),
                       fit_points=int(m.get("prior_fit_max_points", 10000)), seed=int(cfg.seed),
                       tv_weights=tv, max_weight_diff=mx, mmd2=1e3 * float(np.mean(vals)),
                       floor_sd=1e3 * float(np.std(floor)), ratio=float(np.mean(vals) / max(np.std(floor), 1e-12)),
                       run=os.path.relpath(run))
            rows.append(row)
            print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
    if not rows:
        print("no statistical-prior runs found"); return
    df = pd.DataFrame(rows); df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.drop(columns=["run"]).round(4).to_string(index=False))
    print(f"\nper-model results written to {a.csv}")


if __name__ == "__main__":
    main()
