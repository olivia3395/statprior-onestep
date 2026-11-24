"""
Numerically stable distances for the evaluation scripts (float64, computed from coordinate differences).

arccos of an inner product loses all precision for nearby points: in float32, <x, y> rounds to 1 for points closer
than about 3e-4 rad, so kernel values and nearest-neighbour distances of concentrated data are distorted. Here
distances are obtained from the chord |x - y| (sphere) or |R - R'|_F (SO(3)), which is accurate at all scales:
    sphere:  d(x, y)      = 2 asin(|x - y| / 2)
    SO(3):   theta(R, R') = 2 asin(|R - R'|_F / (2 sqrt 2))      (rotation angle)
"""
import math

import torch


def _chord(a, b):
    a = a.reshape(len(a), -1).double()
    b = b.reshape(len(b), -1).double()
    return torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")


def sphere_dist(a, b):
    """Geodesic distance between unit vectors a (n, 3) and b (m, 3); returns (n, m) float64."""
    return 2.0 * torch.asin((_chord(a, b) / 2.0).clamp(max=1.0))


def so3_chord(a, b):
    """Frobenius distance |R - R'|_F between rotation matrices a (n, 3, 3) and b (m, 3, 3)."""
    return _chord(a, b)


def so3_angle(a, b):
    """Rotation angle of R^T R' for rotation matrices a (n, 3, 3) and b (m, 3, 3)."""
    return 2.0 * torch.asin((so3_chord(a, b) / (2.0 * math.sqrt(2.0))).clamp(max=1.0))
