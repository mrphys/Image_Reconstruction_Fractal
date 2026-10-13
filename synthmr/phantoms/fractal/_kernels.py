"""Numba-compiled grid kernels for the built-in fields.

Each kernel takes flattened ``(N, 4)`` quaternion coordinates plus scalar
parameters and returns an ``(N,)`` float64 array. Points are evaluated in
parallel with ``prange``. Quaternions are kept as four scalars to avoid
allocating small arrays inside the hot loop.
"""

import math
import numpy as np
from numba import njit, prange

MODE_PLAIN = 0
MODE_POWER = 1
MODE_PERTURB = 2

TRAP_POINT = 0
TRAP_LINE = 1
TRAP_PLANE = 2


@njit(inline="always")
def _qmul(aw, ax, ay, az, bw, bx, by, bz):
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )


@njit(cache=True)
def _iterate_point(q, c, max_iter, bailout, p, perturb, epsilon):
    """Return ``(n, r)``: escape iteration (``max_iter`` if bounded) and final norm."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    for n in range(max_iter):
        if p == 2:
            w, x, y, z = _qmul(w, x, y, z, w, x, y, z)
        else:
            pw, px, py, pz = w, x, y, z
            for _ in range(p - 1):
                pw, px, py, pz = _qmul(pw, px, py, pz, w, x, y, z)
            w, x, y, z = pw, px, py, pz
        w += c[0]; x += c[1]; y += c[2]; z += c[3]
        if perturb:
            w += epsilon * math.sin(w)
            x += epsilon * math.sin(x)
            y += epsilon * math.sin(y)
            z += epsilon * math.sin(z)
        r = math.sqrt(w*w + x*x + y*y + z*z)
        if r > bailout:
            return n, r
    return max_iter, math.sqrt(w*w + x*x + y*y + z*z)


@njit(parallel=True, cache=True)
def escape_grid(qs, c, max_iter, bailout, p, perturb, epsilon, smooth, interior):
    """Escape count. ``smooth`` gives the continuous count for escaped points;
    ``interior`` replaces bounded points with ``max_iter * |x0|``."""
    out = np.empty(qs.shape[0])
    for i in prange(qs.shape[0]):
        n, r = _iterate_point(qs[i], c, max_iter, bailout, p, perturb, epsilon)
        if n >= max_iter:
            out[i] = max_iter * abs(qs[i, 1]) if interior else max_iter
        elif smooth:
            out[i] = n + 1 - math.log(math.log(max(r, 1.0000001))) / math.log(2.0)
        else:
            out[i] = n
    return out


@njit(parallel=True, cache=True)
def distance_grid(qs, c, max_iter, bailout, floor):
    out = np.empty(qs.shape[0])
    for i in prange(qs.shape[0]):
        w, x, y, z = qs[i, 0], qs[i, 1], qs[i, 2], qs[i, 3]
        dw, dx, dy, dz = 1.0, 0.0, 0.0, 0.0
        r = 0.0
        for _ in range(max_iter):
            dw, dx, dy, dz = _qmul(w, x, y, z, dw, dx, dy, dz)
            dw *= 2.0; dx *= 2.0; dy *= 2.0; dz *= 2.0
            w, x, y, z = _qmul(w, x, y, z, w, x, y, z)
            w += c[0]; x += c[1]; y += c[2]; z += c[3]
            r = math.sqrt(w*w + x*x + y*y + z*z)
            if r > bailout:
                break
        dr = math.sqrt(dw*dw + dx*dx + dy*dy + dz*dz)
        if dr <= floor or r <= 0.0:
            out[i] = 0.0
        else:
            out[i] = 0.5 * r * math.log(r) / dr
    return out


@njit(parallel=True, cache=True)
def orbit_trap_grid(qs, c, max_iter, bailout, trap, axes):
    """``axes`` is an int array: the axes summed for a line trap, or
    ``axes[0]`` for a plane trap."""
    out = np.empty(qs.shape[0])
    for i in prange(qs.shape[0]):
        v = np.empty(4)
        v[:] = qs[i]
        minimum = np.inf
        for _ in range(max_iter):
            w, x, y, z = _qmul(v[0], v[1], v[2], v[3], v[0], v[1], v[2], v[3])
            v[0] = w + c[0]; v[1] = x + c[1]; v[2] = y + c[2]; v[3] = z + c[3]
            if trap == TRAP_POINT:
                d = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2)
            elif trap == TRAP_LINE:
                s = 0.0
                for a in axes:
                    s += v[a] * v[a]
                d = math.sqrt(s)
            else:
                d = abs(v[axes[0]])
            minimum = min(minimum, d)
            if math.sqrt(v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2) > bailout:
                break
        out[i] = minimum
    return out


@njit(parallel=True, cache=True)
def mandelbrot_grid(cs, max_iter, bailout):
    """Quaternion Mandelbrot escape count: iterate ``q -> q^2 + c`` from ``q = 0``
    for each row of ``cs``. Bounded points get ``max_iter``."""
    out = np.empty(cs.shape[0], dtype=np.int64)
    zero = np.zeros(4)
    for i in prange(cs.shape[0]):
        n, _ = _iterate_point(zero, cs[i], max_iter, bailout, 2, False, 0.0)
        out[i] = n
    return out
