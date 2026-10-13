"""Escape-time quaternion Julia/Mandelbrot code from the base repo (``utils/fractal_utils.py``).

Kept verbatim (numerics unchanged) for the original 2D+t path and the entropy
complexity filter. Quaternions here are ``(x, y, z, w)`` with ``x`` the real part,
which is the same Hamilton product as ``_kernels._qmul`` with ``x -> w``.
"""

import numpy as np
from numba import njit, prange


@njit
def quaternion_multiply_scalar(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        x1*x2 - y1*y2 - z1*z2 - w1*w2,
        x1*y2 + y1*x2 + z1*w2 - w1*z2,
        x1*z2 - y1*w2 + z1*x2 + w1*y2,
        x1*w2 + y1*z2 - z1*y2 + w1*x2
    ])


@njit
def quaternion_abs_scalar(q):
    return np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)


@njit(parallel=True)
def julia_escape(grid, c, max_iter=50, escape_radius=4.0):
    """Escape times ``(Nx, Ny, Nz, Nw)`` of the Julia set for ``c`` over a 4-D ``grid`` ``(..., 4)``."""
    shape = grid.shape[:-1]
    escape_times = np.full(shape, max_iter, dtype=np.int32)
    c = np.broadcast_to(c, grid.shape)
    Nx, Ny, Nz, Nw = shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nw):
                    q = grid[i, j, k, l]
                    c_val = c[i, j, k, l]
                    for it in range(max_iter):
                        q = quaternion_multiply_scalar(q, q) + c_val
                        if quaternion_abs_scalar(q) > escape_radius:
                            escape_times[i, j, k, l] = it
                            break
    return escape_times


@njit(parallel=True)
def mandelbrot_escape(c_grid, max_iter=50, escape_radius=4.0):
    """Escape times of the quaternion Mandelbrot set over a 4-D grid of ``c`` values."""
    shape = c_grid.shape[:-1]
    escape_times = np.full(shape, max_iter, dtype=np.int32)
    Nx, Ny, Nz, Nw = shape
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nw):
                    c = c_grid[i, j, k, l]
                    q = np.zeros(4, dtype=np.float64)
                    for it in range(max_iter):
                        q = quaternion_multiply_scalar(q, q)
                        q = q + c
                        if quaternion_abs_scalar(q) > escape_radius:
                            escape_times[i, j, k, l] = it
                            break
    return escape_times


def build_grid(limits, sizes):
    """Grid ``(*sizes, len(sizes))`` spanning ``[-lim, lim]`` per dimension."""
    coords = [np.linspace(-lim, lim, size) for lim, size in zip(limits, sizes)]
    return np.stack(list(np.meshgrid(*coords, indexing='ij')), axis=-1)


def test_complexity(fractal, threshold=1.7):
    """True if the histogram entropy of ``fractal`` is at least ``threshold``."""
    hist, _ = np.histogram(fractal, bins=len(np.unique(fractal)))
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log(prob + 1e-9))
    return bool(entropy >= threshold)


test_complexity.__test__ = False  # not a pytest test
