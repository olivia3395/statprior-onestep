"""
Geometry of the learned one-step map T(z) = Exp_z(-u(z, 0, 1)) (projected), for the uniform and the statistical prior.

For each dataset and prior, on N prior samples z ~ p0:
  mean_logdet   E_{p0}[log |det J_T(z)|], with J_T restricted to tangent spaces (oriented orthonormal bases)
  fold          fraction of points with det J_T < 0 (orientation reversed: the map folds)
  smax_med/p95  median and 95th percentile of the largest singular value of J_T (local Lipschitz constant of T)
  inv_p99       99th percentile of 1 / smallest singular value (local Lipschitz constant of T^{-1})
  disp          mean geodesic displacement d(z, T(z))
and the quantities predicted by Proposition 1:
  req_lb        lower bound on the required average log-contraction H(p0) - H(p1):
                H(p0) - H_hat, with H_hat = -E_test[log q] >= H(p1) for the fitted prior density q
                (uniform: H(p0) = log vol(M); statistical: H(p0) estimated by Monte Carlo)
  lip_bound     lower bound on Lip(T^{-1}) = exp(req_lb / d)
For an injective map, -mean_logdet = H(p0) - H(p_model) is expected to be close to H(p0) - H(p1) >= req_lb.

    python scripts/analyze_jacobian.py --csv jacobian.csv > analyze_jacobian.log 2>&1
"""
import argparse, glob, json, math, os, sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT); sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
import eval_manifold as EM
import eval_real as ER
from eval_synth import load_model
from plot_density import MAIN_EPOCHS, is_main_setting

SPHERE = ["volcano", "earthquake", "flood", "fire"]
TORUS = ["General", "Glycine", "Proline", "Pre-Pro", "RNA"]


def pick(root, names, any_epochs=False):
    """Newest main-setting MeanFlow run per (dataset, prior type), preferring seed 0."""
    best = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "test_metrics.json"))):
        meta = json.load(open(f)); name = meta.get("dataset")
        if name not in names or meta.get("prior_type") not in ("uniform", "gmm"):
            continue
        run = os.path.dirname(f); cfg = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
        if str(cfg.model.get("my_method", "mf")) != "mf":
            continue
        if not any_epochs and int(cfg.trainer.max_epochs) != MAIN_EPOCHS.get(name, -1):
            continue
        if meta["prior_type"] == "gmm" and str(cfg.model.get("prior_coupling", "soft")) not in ("soft", "matched"):
            continue
        if not is_main_setting(cfg, meta):
            continue
        key = (name, meta["prior_type"])
        if key not in best or int(cfg.seed) == 0 or int(OmegaConf.load(os.path.join(best[key], ".hydra", "config.yaml")).seed) != 0:
            best[key] = run
    return best


def oriented_basis(x):
    """Orthonormal tangent basis (b1, b2) at unit vectors x, oriented so that b1 x b2 = x. x: (n, 3)."""
    e = torch.eye(3, dtype=x.dtype, device=x.device)[torch.argmin(x.abs(), dim=1)]
    b1 = torch.nn.functional.normalize(e - (e * x).sum(1, keepdim=True) * x, dim=1)
    b2 = torch.cross(x, b1, dim=1)
    return torch.stack([b1, b2], dim=2)            # (n, 3, 2)


def one_step(model, z):
    sz = z.shape[0]
    r = torch.zeros(sz, *([1] * (z.dim() - 1)), device=z.device, dtype=z.dtype)
    t = torch.ones_like(r)
    return model.manifold.project(model.manifold.exp(z, -model.forward(z, r, t)))


def jacobians(model, z):
    out = []
    for i in range(z.shape[0]):
        J = torch.autograd.functional.jacobian(lambda v: one_step(model, v[None])[0], z[i])
        out.append(J)
    return torch.stack(out)                         # (n, D, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/train/runs")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--csv", default="jacobian.csv")
    ap.add_argument("--any_epochs", action="store_true", help="ignore the training budget (local tests only)")
    ap.add_argument("--datasets", nargs="+", default=None, help="subset of datasets (default: all)")
    ap.add_argument("--best_for", nargs="+", default=[], help="datasets evaluated at the best validation checkpoint")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    runs = pick(a.root, SPHERE + TORUS, a.any_epochs)
    rows = []
    for name in (a.datasets or SPHERE + TORUS):
        ck = "best" if name in a.best_for else "last"
        if (name, "uniform") not in runs or (name, "gmm") not in runs:
            print(f"[skip] {name}: need both priors"); continue
        sphere = name in SPHERE
        # fitted prior density q of the statistical model, for the entropy bounds
        ms, _, cfg = load_model(runs[(name, "gmm")], dev, ck)
        _, _, test = (ER.split_tensors(cfg) if sphere else EM.split_tensors(cfg))
        test = test.to(dev).float()
        d = 2 if sphere else int(test.shape[-1])
        logvol = math.log(4 * math.pi) if sphere else d * math.log(2 * math.pi)
        with torch.no_grad():
            H_hat = -ms.structured_prior.log_prob(test).mean().item()
            zs = ms.structured_prior.sample(20000, device=dev)
            H_stat = -ms.structured_prior.log_prob(zs.reshape(20000, -1)).mean().item()
        for prior in ["uniform", "gmm"]:
            model, _, cfg = load_model(runs[(name, prior)], dev, ck)
            z = model._prior_samples((a.n,) + tuple(model.hparams.in_shape), dev).float()
            J = jacobians(model, z)
            with torch.no_grad():
                y = one_step(model, z)
            if sphere:
                Bz, By = oriented_basis(z), oriented_basis(y)
                Jt = By.transpose(1, 2) @ J @ Bz
                disp = 2 * torch.asin(((y - z).norm(dim=1) / 2).clamp(max=1))
            else:
                Jt = J
                diff = torch.remainder(y - z + math.pi, 2 * math.pi) - math.pi
                disp = diff.norm(dim=1)
            det = torch.linalg.det(Jt.double())
            sv = torch.linalg.svdvals(Jt.double())
            H0 = logvol if prior == "uniform" else H_stat
            req = H0 - H_hat
            rows.append(dict(dataset=name, prior=prior, d=d, ckpt=ck, run=os.path.basename(runs[(name, prior)]),
                             mean_logdet=det.abs().log().mean().item(), fold=(det < 0).double().mean().item(),
                             smax_med=sv[:, 0].median().item(), smax_p95=sv[:, 0].quantile(0.95).item(),
                             inv_p99=(1 / sv[:, -1]).quantile(0.99).item(), disp=disp.mean().item(),
                             req_lb=req, lip_bound=math.exp(max(req, 0) / d)))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    if not rows:
        print("no dataset with both a uniform-prior and a statistical-prior run"); return
    df = pd.DataFrame(rows); df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.drop(columns=["run"]).round(3).to_string(index=False))
    u = df[df.prior == "uniform"]
    if len(u) > 2:
        print("\nSpearman correlation with req_lb (uniform prior):",
              {c: round(float(u[["req_lb", c]].corr(method="spearman").iloc[0, 1]), 2)
               for c in ["mean_logdet", "smax_p95", "inv_p99", "disp", "fold"]})
    print(f"\nper-model results written to {a.csv}")


if __name__ == "__main__":
    main()
