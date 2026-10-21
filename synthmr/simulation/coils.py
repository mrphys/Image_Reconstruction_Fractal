"""3D coil sensitivity simulation (backend-free)."""

import math

import numpy as np

from .phase import background_phase_3d


def _u(gen, lo, hi, shape=(), dtype="float32"):
    return np.asarray(gen.uniform(shape=shape, minval=lo, maxval=hi, dtype=dtype))


def simulate_coils_3d(gen, shape, sigma_coil, num_coils, add_phase=0, ngrid=2, slice_spacing=1.0):
    """Coil maps ``(C, Z, H, W)`` complex64 with max root-sum-of-squares 1.

    Each coil is an anisotropic 3D Gaussian (widths ``base / U(sigma_coil)`` per axis, as in 2D,
    measured in in-plane pixel units) centred outside the in-plane centre (same offset
    distribution as the 2D simulator) and anywhere along the stack (+-75% of its height),
    with random intensity, global phase and an optional smooth 3D background phase.
    ``num_coils`` is ``(n,)`` or ``(lo, hi)``; candidates are drawn then a random subset kept.
    """
    z, h, w = shape
    base = float(max(h, w))
    if len(num_coils) == 2:
        total = int(_u(gen, num_coils[0], num_coils[1], dtype="int32"))
        nsim = int(num_coils[1])
    else:
        total = nsim = int(num_coils[0])
    zz = (np.arange(z) - (z - 1) / 2) * slice_spacing
    yy = np.arange(h) - h // 2
    xx = np.arange(w) - w // 2
    maps = []
    for _ in range(nsim):
        if len(sigma_coil) == 2:
            sig = base / _u(gen, sigma_coil[0], sigma_coil[1], shape=(3,))
        else:
            sig = np.full(3, base / sigma_coil[0])
        phase0 = float(_u(gen, -math.pi, math.pi))
        intensity = float(_u(gen, 0.1, 1.0))
        offset = _u(gen, base // 5, base // 2, shape=(2,)).astype(np.float64)
        signs = (-1.0) ** _u(gen, -1, 9, shape=(2,), dtype="int32")
        cy, cx = offset * signs
        cz = float(_u(gen, -0.75, 0.75)) * z * slice_spacing / 2
        g = (np.exp(-(zz - cz) ** 2 / (2 * sig[0] ** 2))[:, None, None]
             * np.exp(-(yy - cy) ** 2 / (2 * sig[1] ** 2))[None, :, None]
             * np.exp(-(xx - cx) ** 2 / (2 * sig[2] ** 2))[None, None, :])
        coil = intensity * g * np.exp(1j * phase0)
        if add_phase > 0:
            coil = coil * np.exp(1j * background_phase_3d(shape, gen, ngrid=ngrid, scaling=add_phase,
                                                          slice_spacing=slice_spacing))
        maps.append(coil)
    keep = np.argsort(_u(gen, 0, 1, shape=(nsim,)))[:total]
    maps = np.stack(maps)[keep]
    maps /= np.sqrt((np.abs(maps) ** 2).sum(0)).max()
    return maps.astype(np.complex64)
