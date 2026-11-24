"""
Classical baseline: geodesic kernel density estimation (KDE), sampled directly, without any network.

A KDE sample picks a training point x_i uniformly and perturbs it by an isotropic Gaussian in the tangent
space, x = Exp_{x_i}(v), v ~ N(0, h^2 I). The bandwidth h is chosen by maximising the log-likelihood of the
validation split. Samples are evaluated exactly like the generative models (same splits, same kernels,
memorisation ratio), and, for synthetic data with known density, by the true log-likelihood.

Run from rmf/ (CPU is enough):
    python scripts/kde_baseline.py --datasets fire earthquake flood volcano General Glycine Proline Pre-Pro RNA cone fisher24 line
    python scripts/kde_baseline.py --datasets synth_arc30 synth_ring30 synth_arc100 synth_ring100
"""
import argparse
import math
import os
import sys

RMF_ROOT = os.getcwd()
sys.path.insert(0, RMF_ROOT)
sys.path.insert(0, os.path.join(RMF_ROOT, "scripts"))

import numpy as np
import pandas as pd
import torch

import geodesic as GD  # numerically stable float64 distances
from hydra import compose, initialize_config_dir

import eval_manifold as EM
import eval_real as ER

SPECS = {  # name -> (space, experiment, extra overrides)
    "fire": ("sphere", "meanflow/meanflow_earth_fire", []),
    "earthquake": ("sphere", "meanflow/meanflow_earth_earthquake", []),
    "flood": ("sphere", "meanflow/meanflow_earth_flood", []),
    "volcano": ("sphere", "meanflow/meanflow_earth_volcano", []),
    "General": ("torus", "meanflow/meanflow_amino_general", []),
    "Glycine": ("torus", "meanflow/meanflow_amino_glycine", []),
    "Proline": ("torus", "meanflow/meanflow_amino_proline", []),
    "Pre-Pro": ("torus", "meanflow/meanflow_amino_prepro", []),
    "RNA": ("torus", "meanflow/meanflow_amino_rna", []),
    "cone": ("so3", "meanflow/meanflow_so3", ["data.dataset_file=cone"]),
    "fisher24": ("so3", "meanflow/meanflow_so3_fisher", []),
    "line": ("so3", "meanflow/meanflow_so3", ["data.dataset_file=line"]),
    "swiss": ("so3", "meanflow/meanflow_so3_swiss", []),
}
for s in ["synth_arc30", "synth_ring30", "synth_arc100", "synth_ring100"]:
    SPECS[s] = ("sphere", "meanflow/meanflow_earth_synth", [f"data.dataset_file={s}"])
for _k in [2, 3, 5]:
    SPECS[f"pepwin_k{_k}"] = ("torus", "meanflow/meanflow_torus_synth",
                              [f"data.data_type=pepwin_k{_k}", "data.data_dir=data/torus_windows",
                               f"model.net.d_in={2 * _k}", f"model.in_shape=[{2 * _k}]"])
for _d in [2, 5, 10, 20, 50]:
    SPECS[f"tsynth_d{_d}"] = ("torus", "meanflow/meanflow_torus_synth",
                              [f"data.data_type=tsynth_d{_d}", f"model.net.d_in={_d}", f"model.in_shape=[{_d}]"])


# ------------------------------------------------------------------ geometry
def sphere_basis(x):
    a = torch.where(x[:, :1].abs() < 0.9, torch.tensor([1.0, 0, 0]).expand_as(x), torch.tensor([0, 1.0, 0]).expand_as(x))
    e1 = torch.nn.functional.normalize(a - (a * x).sum(1, keepdim=True) * x, dim=1)
    return e1, torch.cross(x, e1, dim=1)


def sphere_exp(x, v2):
    e1, e2 = sphere_basis(x)
    v = v2[:, :1] * e1 + v2[:, 1:] * e2
    r = v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cos(r) * x + torch.sin(r) * v / r


def so3_hat(v):
    O = torch.zeros(len(v), 3, 3, dtype=v.dtype)
    O[:, 0, 1], O[:, 0, 2], O[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    return O - O.transpose(1, 2)


def so3_exp(v):
    th = v.norm(dim=1).clamp_min(1e-12)[:, None, None]
    K = so3_hat(v / th[:, :, 0])
    I = torch.eye(3, dtype=v.dtype).expand(len(v), 3, 3)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def geodesic_r(a, b, space):
    """distances between every a_i and b_j (for the KDE likelihood)."""
    if space == "sphere":
        return GD.sphere_dist(a, b)
    if space == "so3":
        return GD.so3_angle(a, b)
    return EM._wrap(a[:, None] - b[None]).norm(dim=-1)


def log_jac(r, space):
    if space == "sphere":
        return torch.log(torch.sinc(r / math.pi).clamp_min(1e-12))          # sin r / r
    if space == "so3":
        return torch.log((2 * (1 - torch.cos(r)) / r.clamp_min(1e-6) ** 2).clamp_min(1e-12))
    return torch.zeros_like(r)


def kde_loglik(y, centers, h, space, d, block=200):
    out = []
    for i in range(0, len(y), block):                      # blocks keep memory small in high dimension
        r = geodesic_r(y[i:i + block], centers, space)
        lk = -0.5 * (r / h) ** 2 - d * math.log(h) - 0.5 * d * math.log(2 * math.pi) - log_jac(r, space)
        out.append(torch.logsumexp(lk, 1) - math.log(len(centers)))
    return torch.cat(out).mean().item()


def kde_sample(train, n, h, space, gen):
    idx = torch.randint(len(train), (n,), generator=gen)
    x = train[idx]
    if space == "sphere":
        return sphere_exp(x, h * torch.randn(n, 2, generator=gen, dtype=x.dtype))
    if space == "so3":
        return x.reshape(n, 3, 3) @ so3_exp(h * torch.randn(n, 3, generator=gen, dtype=x.dtype))
    return torch.remainder(x + h * torch.randn(x.shape, generator=gen, dtype=x.dtype), 2 * math.pi)


# ------------------------------------------------------------------ main
def load_split(name):
    space, exp, extra = SPECS[name]
    with initialize_config_dir(config_dir=os.path.join(RMF_ROOT, "configs"), version_base="1.3"):
        cfg = compose("train.yaml", overrides=[f"experiment={exp}", "seed=0"] + extra)
    if space == "sphere":
        tr, va, te = ER.split_tensors(cfg)
    else:
        tr, va, te = EM.split_tensors(cfg)
    if space == "so3":
        tr, va, te = [t.reshape(len(t), 3, 3) for t in (tr, va, te)]
    return space, tr.double(), va.double(), te.double()


def metrics(space, gen, test, train):
    out = {}
    if space == "sphere":
        for kind, g in [("gauss", 1.0), ("lap", 1.0), ("lap", 10.0), ("lap", 50.0)]:
            out[f"{kind}{g:g}"] = 1e3 * ER.mmd2_unbiased(gen.float(), test.float(), kind, g)
        nn = lambda a: ER.nn_mean(a.float(), train.float())
    else:
        for kind, g in EM.kernels(space):
            out[f"{'rmf' if kind == 'paper' else 'lap'}{g:g}"] = 1e3 * EM.mmd2(gen.float(), test.float(), space, kind, g)
        nn = lambda a: EM.nn_mean(a.float(), train.float(), space, block=256)
    out["nn_ratio"] = nn(gen) / nn(test)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max_centers", type=int, default=5000, help="kernel centres used for bandwidth selection")
    ap.add_argument("--csv", default="kde_eval.csv")
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(0)
    rows = []
    for name in a.datasets:
        space, tr, va, te = load_split(name)
        d = {"sphere": 2, "so3": 3}.get(space, tr.shape[-1])
        centers = tr[torch.randperm(len(tr), generator=gen)[: a.max_centers]]
        vsub = va[: 2000]
        grid = np.exp(np.linspace(np.log(0.003), np.log(0.6), 25))
        lls = [kde_loglik(vsub, centers, h, space, d) for h in grid]
        h = float(grid[int(np.argmax(lls))])
        ref = tr[torch.randperm(len(tr), generator=gen)[:20000]]
        print(f"[{name}] space={space} n_train={len(tr)} n_test={len(te)} bandwidth h={h:.4f} rad "
              f"(val LL {max(lls):.3f})", flush=True)
        for rep in range(a.reps):
            x = kde_sample(tr, a.n, h, space, gen)
            if name.startswith("tsynth_"):      # exact density: log-likelihood lost per dimension
                import torus_synth as TS
                floor = -d * math.log(2 * math.pi) - 20.0
                lp = TS.logpdf(name, x.numpy())
                below = (~np.isfinite(lp)) | (lp < floor)
                lp = np.maximum(np.nan_to_num(lp, neginf=floor), floor)
                ll_t = TS.logpdf(name, TS.sample(name, 20000, np.random.default_rng(7))).mean()
                rows.append(dict(dataset=name, space=space, h=h, rep=rep, gap_per_dim=(ll_t - lp.mean()) / d,
                                 below_floor=below.mean()))
                print("   ", rows[-1], flush=True)
                continue
            r = dict(dataset=name, space=space, h=h, rep=rep, **metrics(space, x, te, ref))
            if name.startswith("synth_"):
                import synth_shapes as SS
                ll_x = SS.logpdf(name, x.numpy()).mean()
                ll_t = SS.logpdf(name, SS.sample(name, 20000, np.random.default_rng(7))).mean()
                r["ll_gap"] = ll_t - ll_x
            rows.append(r)
            print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if k not in ("space",)}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.csv, index=False)
    pd.set_option("display.width", 220)
    num = [c for c in df.columns if c not in ("dataset", "space", "rep")]
    print("\nKDE baseline (mean over draws; MMD^2 x1e-3 against the held-out split)")
    print(df.groupby("dataset")[num].mean().round(4).to_string())


if __name__ == "__main__":
    main()
