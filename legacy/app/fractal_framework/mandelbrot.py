"""Quaternion Mandelbrot set, used to pick interesting Julia ``c`` parameters.

Julia sets are most detailed when ``c`` lies near the Mandelbrot boundary, i.e.
where the orbit of 0 escapes neither immediately nor never.
"""

import numpy as np

from ._kernels import mandelbrot_grid


def mandelbrot_escape(cs, max_iter=50, bailout=4.0):
    """Escape counts for quaternion parameters ``cs`` of shape ``(..., 4)``."""
    cs = np.ascontiguousarray(cs, dtype=np.float64)
    flat = cs.reshape(-1, 4)
    return mandelbrot_grid(flat, int(max_iter), float(bailout)).reshape(cs.shape[:-1])


def mandelbrot_grid_4d(resolution=100, extent=1.5):
    """Regular ``(R, R, R, R, 4)`` grid of ``c`` values over ``[-extent, extent]^4``."""
    axis = np.linspace(-extent, extent, resolution)
    return np.stack(np.meshgrid(axis, axis, axis, axis, indexing="ij"), axis=-1)


def candidate_c(n=None, resolution=40, extent=1.5, max_iter=50, bailout=4.0,
                min_iter=10, max_escape=30, rng=None):
    """Sample Julia ``c`` values near the quaternion Mandelbrot boundary.

    Evaluates a 4-D grid and keeps parameters whose escape count lies in
    ``(min_iter, max_escape)``. Returns all candidates ``(M, 4)`` if ``n`` is
    None, otherwise ``n`` rows drawn at random (with replacement if needed).
    """
    cs = mandelbrot_grid_4d(resolution, extent).reshape(-1, 4)
    esc = mandelbrot_escape(cs, max_iter, bailout)
    candidates = cs[(esc > min_iter) & (esc < max_escape)]
    if n is None:
        return candidates
    if len(candidates) == 0:
        raise ValueError("no candidates in range; widen min_iter/max_escape or raise resolution")
    rng = np.random.default_rng(rng)
    idx = rng.choice(len(candidates), size=n, replace=n > len(candidates))
    return candidates[idx]
