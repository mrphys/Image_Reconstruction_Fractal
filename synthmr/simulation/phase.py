"""Smooth random background phase (backend-free)."""

import math

import numpy as np
from scipy.ndimage import zoom


def background_phase_3d(shape, gen, ngrid=6, scaling=1.0, slice_spacing=1.0):
    """Smooth random phase map ``(Z, H, W)`` in radians.

    A coarse uniform random grid (``ngrid`` points in-plane, proportionally fewer along z
    given the stack's physical height) is tricubically upsampled to the volume, plus a
    random global offset in [0, 2 pi). The same map is applied to every frame.
    """
    z, h, w = shape
    nz = max(2, int(math.ceil(ngrid * z * slice_spacing / max(h, w)))) if z > 1 else 1
    coarse = np.asarray(gen.uniform(shape=(nz, ngrid, ngrid)), dtype=np.float64) * 2 * scaling
    fine = zoom(coarse, (z / nz, h / ngrid, w / ngrid), order=3, mode="nearest", grid_mode=True)
    fine = fine[:z, :h, :w]
    offset = float(np.asarray(gen.uniform(shape=(1,)))[0]) * 2 * math.pi
    return (fine + offset).astype(np.float32)
