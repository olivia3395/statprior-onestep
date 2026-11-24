"""
Write non-vMF synthetic sphere datasets (rings / arcs, see synth_shapes.py) as
LATITUDE,LONGITUDE csv files that EarthDataModule can load.

    python scripts/make_synth_shapes.py                       # rings & arcs, widths 100 and 30 mrad
    python scripts/make_synth_shapes.py --shapes ring --widths 100 50 25
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_shapes as S

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", nargs="+", default=["ring", "arc"])
ap.add_argument("--widths", type=int, nargs="+", default=[100, 30], help="band width in milliradians")
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/earth_data")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
for shape in a.shapes:
    for w in a.widths:
        name = f"synth_{shape}{w}"
        x = S.sample(name, a.n, rng)
        lat = np.degrees(np.arcsin(np.clip(x[:, 2], -1, 1)))
        lon = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
        path = os.path.join(a.out, f"{name}.csv")
        np.savetxt(path, np.stack([lat, lon], 1), delimiter=",", header="LATITUDE,LONGITUDE", comments="", fmt="%.6f")
        print(f"wrote {path}")
