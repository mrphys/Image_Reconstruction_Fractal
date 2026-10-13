"""Through-plane encoders (backend-free helpers).

``none``          2D sampling, one in-plane trajectory per frame.
``cartesian_kz``  stack-of-* : fully sampled Cartesian kz (FFT along Z), in-plane trajectory per
                  partition. ``kz_rotation='aligned'`` uses the same star for every partition
                  (classic stack-of-stars); ``'golden'`` rotates partition ``k`` by ``k`` times the
                  golden angle (pi * 0.618...) to spread spokes through kz.
"""

import numpy as np

GOLDEN_ANGLE = np.pi * (np.sqrt(5) - 1) / 2

THROUGH_PLANE = ("none", "cartesian_kz")


def kz_trajectories(traj, n_partitions, mode="aligned"):
    """Per-partition in-plane trajectories ``(T, Z, R, 2)`` or ``None`` when all partitions share ``traj``."""
    if mode == "aligned":
        return None
    if mode != "golden":
        raise ValueError(f"unknown kz_rotation {mode!r}")
    traj = np.asarray(traj)
    angles = (np.arange(n_partitions) * GOLDEN_ANGLE) % np.pi
    c, s = np.cos(angles), np.sin(angles)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)          # (Z, 2, 2)
    out = np.einsum("zij,trj->tzri", rot, traj)
    # keep within [-pi, pi] (rotation preserves radius; guard float rounding)
    return np.clip(out, -np.pi, np.pi).astype(np.float32)
