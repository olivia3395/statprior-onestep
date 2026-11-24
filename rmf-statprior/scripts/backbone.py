"""
Rebuild protein backbone fragments from (phi, psi) torsions and compute physical metrics.

Reconstruction: NeRF (natural extension reference frame) with ideal bond lengths and angles
(Engh & Huber), omega = 180 deg (trans peptide bonds). Given k residues with torsions
(phi_1, psi_1, ..., phi_k, psi_k), atoms are placed in the order N_i, CA_i, C_i; phi_1 and psi_k
do not affect the coordinates. Real and generated fragments are rebuilt the same way, so the
comparison isolates the distribution of the torsions.

Metrics (per fragment):
  clash      any pair of CA atoms at least 3 residues apart closer than 3.0 A (severe steric clash; real
             fragments rebuilt with ideal geometry rarely go below this, while residue-independent angles do)
  rg         radius of gyration of the CA atoms (compactness)
  e2e        CA_1 -- CA_k end-to-end distance
  helix      fraction of residues in the right-handed alpha region
  sheet      fraction of residues in the beta region
  helix_run  whether the fragment contains >= 4 consecutive residues in the alpha region (a helical turn);
             this depends on the joint distribution of neighbouring residues, not only on their marginals
"""
import math

import numpy as np

B_N_CA, B_CA_C, B_C_N = 1.458, 1.525, 1.329                    # bond lengths (A)
A_N_CA_C, A_CA_C_N, A_C_N_CA = np.radians([111.2, 116.2, 121.7])  # bond angles
OMEGA = math.pi


def _place(a, b, c, bond, angle, torsion):
    """Position of d such that |cd| = bond, angle(b,c,d) = angle, dihedral(a,b,c,d) = torsion (batched)."""
    bc = c - b
    bc /= np.linalg.norm(bc, axis=-1, keepdims=True)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    m = np.cross(n, bc)
    d2 = np.stack([-bond * np.cos(angle) * np.ones_like(torsion),
                   bond * np.sin(angle) * np.cos(torsion),
                   bond * np.sin(angle) * np.sin(torsion)], -1)
    return c + d2[..., :1] * bc + d2[..., 1:2] * m + d2[..., 2:3] * n


def build_backbone(torsions):
    """torsions: (n, 2k) array (phi_1, psi_1, ..., phi_k, psi_k) in radians -> (n, k, 3, 3) N/CA/C coordinates."""
    t = np.asarray(torsions, dtype=np.float64)
    n, k = t.shape[0], t.shape[1] // 2
    phi, psi = t[:, 0::2], t[:, 1::2]
    X = np.zeros((n, k, 3, 3))
    X[:, 0, 0] = [0.0, 0.0, 0.0]
    X[:, 0, 1] = [B_N_CA, 0.0, 0.0]
    X[:, 0, 2] = X[:, 0, 1] + B_CA_C * np.array([-math.cos(A_N_CA_C), math.sin(A_N_CA_C), 0.0])
    for i in range(k - 1):
        N, CA, C = X[:, i, 0], X[:, i, 1], X[:, i, 2]
        N1 = _place(N, CA, C, B_C_N, A_CA_C_N, psi[:, i])
        CA1 = _place(CA, C, N1, B_N_CA, A_C_N_CA, np.full(n, OMEGA))
        C1 = _place(C, N1, CA1, B_CA_C, A_N_CA_C, phi[:, i + 1])
        X[:, i + 1, 0], X[:, i + 1, 1], X[:, i + 1, 2] = N1, CA1, C1
    return X


def _wrapdeg(a):
    return (np.degrees(a) + 180.0) % 360.0 - 180.0


def fragment_metrics(torsions, clash_cutoff=3.0, min_sep=3, run=4):
    X = build_backbone(torsions)
    ca = X[:, :, 1]
    k = ca.shape[1]
    d = np.linalg.norm(ca[:, :, None] - ca[:, None, :], axis=-1)
    iu = np.triu_indices(k, min_sep)
    clash = (d[:, iu[0], iu[1]] < clash_cutoff).any(1) if len(iu[0]) else np.zeros(len(ca), bool)
    rg = np.sqrt(((ca - ca.mean(1, keepdims=True)) ** 2).sum(-1).mean(1))
    e2e = np.linalg.norm(ca[:, -1] - ca[:, 0], axis=-1)
    phi, psi = _wrapdeg(np.asarray(torsions)[:, 0::2]), _wrapdeg(np.asarray(torsions)[:, 1::2])
    helix = ((phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)).mean(1)
    in_alpha = (phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)
    sheet = ((phi > -180) & (phi < -45) & ((psi > 90) | (psi < -150))).mean(1)
    cnt = np.zeros(len(ca), int); helix_run = np.zeros(len(ca), bool)
    for j in range(in_alpha.shape[1]):
        cnt = np.where(in_alpha[:, j], cnt + 1, 0); helix_run |= cnt >= run
    return dict(clash=clash.astype(float), rg=rg, e2e=e2e, helix=in_alpha.mean(1), sheet=sheet,
                helix_run=helix_run.astype(float))


def wasserstein1(a, b):
    """1-D Wasserstein distance between two samples (quantile matching)."""
    q = np.linspace(0, 1, 1001)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def compare(gen_torsions, ref_torsions):
    """Summary of physical plausibility of generated fragments against reference fragments."""
    g, r = fragment_metrics(gen_torsions), fragment_metrics(ref_torsions)
    return dict(
        clash=float(g["clash"].mean()), clash_ref=float(r["clash"].mean()),
        helix_run=float(g["helix_run"].mean()), helix_run_ref=float(r["helix_run"].mean()),
        helix=float(g["helix"].mean()), helix_ref=float(r["helix"].mean()),
        sheet=float(g["sheet"].mean()), sheet_ref=float(r["sheet"].mean()),
        W1_e2e=wasserstein1(g["e2e"], r["e2e"]), W1_rg=wasserstein1(g["rg"], r["rg"]),
    )
