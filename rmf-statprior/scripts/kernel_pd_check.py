# Run from the rmf/ directory: python scripts/kernel_pd_check.py
import numpy as np, csv
rng = np.random.default_rng(0)
def latlon_to_xyz(ll):
    lat, lon = np.deg2rad(ll[:,0]), np.deg2rad(ll[:,1])
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)],1)
def load(name):
    rows = list(csv.reader(open(f"data/earth_data/{name}.csv")))[1:]
    return latlon_to_xyz(np.array(rows, dtype=float))
def sphere_dist(X):
    return np.arccos(np.clip(X@X.T, -1, 1))
def so3_dist(R):
    # R: n x 3 x 3
    tr = np.einsum('iab,jab->ij', R, R)  # trace(Ri^T Rj)
    return np.arccos(np.clip((tr-1)/2, -1, 1))
def report(name, D, n):
    for label, K in [("geodesic Gaussian exp(-d^2)", np.exp(-D**2)),
                     ("geodesic Laplace  exp(-d)  ", np.exp(-D))]:
        ev = np.linalg.eigvalsh(K)
        print(f"{name:22s} n={n:5d} {label}: min eig = {ev.min(): .3e}, #neg(<-1e-10) = {(ev<-1e-10).sum()}")
for name in ["volcano","earthquake","flood","fire"]:
    X = load(name)
    if len(X) > 3000: X = X[rng.choice(len(X), 3000, replace=False)]
    report("earth/"+name, sphere_dist(X), len(X))
U = rng.normal(size=(2000,3)); U /= np.linalg.norm(U,axis=1,keepdims=True)
report("sphere uniform", sphere_dist(U), 2000)
for name in ["cone","fisher24","line"]:
    R = np.load(f"data/raw/{name}_test.npy")
    R = R.reshape(-1,3,3)[rng.choice(len(R), 2000, replace=False)]
    report("so3/"+name, so3_dist(R), 2000)
