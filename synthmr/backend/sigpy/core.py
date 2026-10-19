"""Core MRI / array ops for the sigpy backend (numpy on CPU, cupy on GPU).

Same semantics as ``backend/tf/core.py`` (docs/BACKEND.md). The tfmri algorithms are ported from
tensorflow-mri 0.22 (trajectories, radial density, Walsh, SVD coil compression, crop/pad); NUFFTs
use ``sigpy.nufft`` with pixel-unit coordinates as in mikgroup/extreme_mri.
"""

import contextlib
import math

import numpy as np
import sigpy as sp

from synthmr.rng import NumpyGenerator

NAME = "sigpy"

# NUFFT accuracy: sigpy's defaults (1.25, 4) give ~1e-2 relative error; (2, 6) ~5e-5 (tfft: ~1e-6).
NUFFT_OVERSAMP = 2.0
NUFFT_WIDTH = 6
# Walsh: max elements of one (pixels, C, C) covariance slab.
WALSH_SLAB_ELEMENTS = 2 ** 26


# --- devices ----------------------------------------------------------------
_device_stack = []


def _gpu_available():
    if not sp.config.cupy_enabled:
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - no driver / no device
        return False


def _parse_device(spec):
    if spec is None:
        return sp.Device(0) if _gpu_available() else sp.cpu_device
    spec = spec.lstrip("/").lower()
    kind, _, idx = spec.partition(":")
    if kind == "cpu":
        return sp.cpu_device
    if kind == "gpu":
        return sp.Device(int(idx or 0))
    raise ValueError(f"unknown device {spec!r}; use 'cpu', 'gpu' or 'gpu:N'")


def current_device():
    """Device new arrays go to: innermost ``device(...)`` context, else GPU 0 if available, else CPU."""
    return _device_stack[-1] if _device_stack else _parse_device(None)


@contextlib.contextmanager
def device(spec):
    """Context manager placing arrays/ops on ``spec`` (``"gpu"``, ``"gpu:1"``, ``"cpu"``); ``None`` = default."""
    dev = _parse_device(spec)
    _device_stack.append(dev)
    try:
        with dev:
            yield dev
    finally:
        _device_stack.pop()


def current_xp():
    """Array module (numpy / cupy) of :func:`current_device`."""
    return current_device().xp


def xp_of(x):
    return sp.get_array_module(x)


def ndimage_of(x):
    """``scipy.ndimage`` or ``cupyx.scipy.ndimage`` matching the array's device."""
    if xp_of(x) is np:
        import scipy.ndimage as ndi
    else:
        import cupyx.scipy.ndimage as ndi
    return ndi


# --- dtype / conversion -----------------------------------------------------
def as_dtype(dtype):
    """Accept ``"float32"``, ``np.float32`` or a TF dtype."""
    return np.dtype(getattr(dtype, "as_numpy_dtype", dtype))


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if xp_of(x) is not np:
        return sp.to_device(x, sp.cpu_device)
    return np.asarray(x)


def from_numpy(x, dtype=None):
    if xp_of(x) is np:
        x = np.asarray(x, dtype=None if dtype is None else as_dtype(dtype))
    elif dtype is not None:
        x = x.astype(as_dtype(dtype))
    return sp.to_device(x, current_device())


# --- random numbers ---------------------------------------------------------
# Host-side numpy philox streams (the backend-free 3D modules ``np.asarray`` every draw).
Generator = NumpyGenerator


# --- Fourier transforms -----------------------------------------------------
def fft(x, axes, norm="ortho", shift=True):
    """Centred n-D FFT over ``axes``."""
    return sp.fft(x, axes=tuple(axes), center=shift, norm=norm)


def ifft(x, axes, norm="ortho", shift=True):
    """Centred n-D inverse FFT over ``axes``."""
    return sp.ifft(x, axes=tuple(axes), center=shift, norm=norm)


def _coord(traj, grid_shape):
    """radians/pixel [-pi, pi] -> sigpy pixel units [-N/2, N/2] (``traj[..., i]`` <-> image axis ``-ndim + i``)."""
    xp = xp_of(traj)
    scale = xp.asarray([n / (2 * math.pi) for n in grid_shape], dtype=traj.dtype)
    return traj * scale


def _over_traj_batches(fn, data, data_core_ndim, traj):
    """Apply ``fn(data_b, traj_b)`` with ``traj_b`` ``(R, ndim)``, looping only over the batch axes in which
    ``traj`` varies (sigpy's NUFFT does not broadcast batch axes of ``coord``). Batch axes of ``data``
    and ``traj`` (``traj.shape[:-2]``) broadcast like tfft."""
    xp = xp_of(data)
    if traj.ndim == 2:
        return fn(data, traj)
    data_batch, core = data.shape[:data.ndim - data_core_ndim], data.shape[data.ndim - data_core_ndim:]
    batch = tuple(np.broadcast_shapes(data_batch, traj.shape[:-2]))
    tb = (1,) * (len(batch) - (traj.ndim - 2)) + tuple(traj.shape[:-2])
    traj = traj.reshape(tb + traj.shape[-2:])
    data = xp.broadcast_to(data, batch + core)
    loop = [a for a, n in enumerate(tb) if n > 1]
    out = None
    for idx in np.ndindex(*[batch[a] for a in loop]):
        dsl, tsl = [slice(None)] * len(batch), [0] * len(batch)
        for a, i in zip(loop, idx):
            dsl[a] = tsl[a] = i
        res = fn(xp.ascontiguousarray(data[tuple(dsl)]), traj[tuple(tsl)])
        if out is None:
            out = xp.empty(batch + res.shape[len(batch) - len(loop):], dtype=res.dtype)
        out[tuple(dsl)] = res
    return out


def nufft_forward(image, traj):
    """Type-2 NUFFT (grid -> samples), unscaled. ``traj`` ``(..., R, ndim)`` in [-pi, pi]."""
    nd = traj.shape[-1]
    grid = image.shape[-nd:]
    gain = math.sqrt(math.prod(grid))

    def fn(x, t):
        return sp.nufft(x, _coord(t, grid), oversamp=NUFFT_OVERSAMP, width=NUFFT_WIDTH) * gain

    return _over_traj_batches(fn, image, nd, traj)


def nufft_adjoint(kspace, traj, grid_shape):
    """Type-1 NUFFT (samples -> grid), unscaled."""
    grid = tuple(int(n) for n in grid_shape)
    gain = math.sqrt(math.prod(grid))

    def fn(k, t):
        return sp.nufft_adjoint(k, _coord(t, grid), oshape=k.shape[:-1] + grid,
                                oversamp=NUFFT_OVERSAMP, width=NUFFT_WIDTH) * gain

    return _over_traj_batches(fn, kspace, 1, traj)


# --- image ops --------------------------------------------------------------
def crop_or_pad(x, shape):
    """Centred crop/pad of the trailing ``len(shape)`` axes; ``-1`` keeps a dimension (tfmri offsets)."""
    xp = xp_of(x)
    target = [-1] * (x.ndim - len(shape)) + [int(s) for s in shape]
    pads = [(max(t - n, 0) // 2, (max(t - n, 0) + 1) // 2) if t >= 0 else (0, 0) for t, n in zip(target, x.shape)]
    if any(p != (0, 0) for p in pads):
        x = xp.pad(x, pads)
    sl = tuple(slice((n - t) // 2, (n - t) // 2 + t) if 0 <= t < n else slice(None) for t, n in zip(target, x.shape))
    return x[sl]


def scale_by_min_max(x):
    """Rescale to [0, 1]; complex input: magnitude rescaled, phase kept."""
    xp = xp_of(x)

    def rescale(v):
        return (v - v.min()) / (v.max() - v.min())

    if xp.iscomplexobj(x):
        return (rescale(xp.abs(x)) * xp.exp(1j * xp.angle(x))).astype(x.dtype)
    return rescale(x)


_KEYS_TABLE_SIZE = 1024


def _keys_table(a=-0.5):
    """``tf.image.resize`` bicubic coefficient table (tensorflow/core/kernels/image/resize_bicubic_op.cc)."""
    i = np.arange(_KEYS_TABLE_SIZE + 1, dtype=np.float64)
    x = (i / _KEYS_TABLE_SIZE).astype(np.float32).astype(np.float64)
    t0 = ((a + 2) * x - (a + 3)) * x * x + 1
    x1 = x + 1.0
    t1 = ((a * x1 - 5 * a) * x1 + 8 * a) * x1 - 4 * a
    return np.stack([t0, t1], -1).astype(np.float32)


def _bicubic_matrix(n_in, n_out):
    """``(n_out, n_in)`` interpolation matrix: half-pixel centres, Keys a=-0.5, out-of-range taps dropped and
    weights renormalised (TF ResizeBicubic with ``half_pixel_centers=True``)."""
    table = _keys_table()
    scale = np.float32(n_in) / np.float32(n_out)
    m = np.zeros((n_out, n_in), dtype=np.float32)
    for o in range(n_out):
        src = (np.float32(o) + np.float32(0.5)) * scale - np.float32(0.5)
        loc = int(math.floor(src))
        off = int(np.rint(np.float32(src - loc) * _KEYS_TABLE_SIZE))
        w = [table[off, 1], table[off, 0], table[_KEYS_TABLE_SIZE - off, 0], table[_KEYS_TABLE_SIZE - off, 1]]
        idx = [loc - 1, loc, loc + 1, loc + 2]
        w = [wk if 0 <= ik < n_in else np.float32(0) for wk, ik in zip(w, idx)]
        total = np.float32(sum(w))
        if abs(total) >= 1000 * np.finfo(np.float32).tiny:
            w = [wk / total for wk in w]
        for wk, ik in zip(w, idx):
            m[o, min(max(ik, 0), n_in - 1)] += wk
    return m


def resize(images, size, method="bicubic"):
    """``tf.image.resize(bicubic)`` on ``(N, H, W, C)`` or ``(H, W, C)`` images (upsampling)."""
    if method != "bicubic":
        raise NotImplementedError(f"resize method {method!r}")
    xp = xp_of(images)
    h, w = images.shape[-3], images.shape[-2]
    my = xp.asarray(_bicubic_matrix(h, int(size[0])))
    mx = xp.asarray(_bicubic_matrix(w, int(size[1])))
    out = xp.einsum("oh,...hwc->...owc", my, images.astype(np.float32))
    return xp.einsum("pw,...owc->...opc", mx, out)


def translate(images, translations, interpolation="bilinear"):
    """Shift ``(H, W, C)`` images by ``translations`` ``[dx, dy]`` pixels; zero fill (``tfa.image.translate``)."""
    if interpolation != "bilinear":
        raise NotImplementedError(f"translate interpolation {interpolation!r}")
    ndi = ndimage_of(images)
    dx, dy = (float(v) for v in to_numpy(translations).reshape(-1)[:2])
    shift = [0.0] * images.ndim
    shift[-3], shift[-2] = dy, dx
    return ndi.shift(images, shift, order=1, mode="grid-constant", cval=0.0, prefilter=False)


# --- coils ------------------------------------------------------------------
class _CoilCompressorSVD:
    """SVD coil compression (tfmri ``CoilCompressorSVD``): ``x @ V[:, :out_coils]`` along ``coil_axis``."""

    def __init__(self, coil_axis=-1, out_coils=None):
        self.coil_axis, self.out_coils, self.matrix = coil_axis, out_coils, None

    def fit(self, kspace):
        xp = xp_of(kspace)
        flat = xp.moveaxis(kspace, self.coil_axis, -1)
        flat = flat.reshape(-1, flat.shape[-1])
        u, _, vh = xp.linalg.svd(flat, full_matrices=False)
        matrix = vh.conj().T if flat.shape[0] > flat.shape[1] else u
        self.matrix = matrix[:, :self.out_coils] if self.out_coils is not None else matrix
        return self

    def transform(self, kspace):
        xp = xp_of(kspace)
        x = xp.moveaxis(kspace, self.coil_axis, -1)
        return xp.moveaxis(x @ self.matrix.astype(x.dtype), -1, self.coil_axis)

    def fit_transform(self, kspace):
        return self.fit(kspace).transform(kspace)


def coil_compressor_svd(coil_axis, out_coils):
    """Object with ``fit(x)`` / ``transform(x)`` (SVD coil compression)."""
    return _CoilCompressorSVD(coil_axis=coil_axis, out_coils=out_coils)


def _walsh(images, filter_size=5):
    """Walsh maps for ``images`` ``(*spatial, C)``: dominant eigenvector of the box-filtered coil covariance.

    Same as tfmri ``_estimate_walsh`` (zero-padded 'SAME' uniform filter); processed in slabs along the
    first spatial axis (halo ``filter_size // 2``) to bound memory.
    """
    xp, ndi = xp_of(images), ndimage_of(images)
    spatial, nc = images.shape[:-1], images.shape[-1]
    halo = filter_size // 2
    per_row = math.prod(spatial[1:]) * nc * nc
    rows = max(1, WALSH_SLAB_ELEMENTS // per_row - 2 * halo)
    size = [filter_size] * len(spatial) + [1, 1]
    maps = xp.empty(images.shape, dtype=np.complex64)
    for start in range(0, spatial[0], rows):
        stop = min(start + rows, spatial[0])
        lo, hi = max(start - halo, 0), min(stop + halo, spatial[0])
        x = images[lo:hi].astype(np.complex64)
        cov = x[..., :, None] * xp.conj(x[..., None, :])
        cov = (ndi.uniform_filter(cov.real, size=size, mode="constant")
               + 1j * ndi.uniform_filter(cov.imag, size=size, mode="constant")).astype(np.complex64)
        cov = cov[start - lo:start - lo + (stop - start)]
        _, vecs = xp.linalg.eigh(cov.reshape(-1, nc, nc))
        maps[start:stop] = vecs[..., -1].reshape(cov.shape[:-1])
    return maps


def estimate_sensitivities(x, coil_axis=-1, method="walsh"):
    """Coil maps; all non-coil axes are spatial (rank 2 or 3)."""
    if method != "walsh":
        raise NotImplementedError(f"estimate_sensitivities method {method!r} (use coil_maps='jsense' for JSENSE)")
    xp = xp_of(x)
    if x.ndim - 1 not in (2, 3):
        raise ValueError(f"Walsh needs spatial rank 2 or 3, got input shape {x.shape}")
    return xp.moveaxis(_walsh(xp.moveaxis(x, coil_axis, -1)), -1, coil_axis)


def combine_coils(x, maps=None, coil_axis=-1, keepdims=False):
    """Coil combination: adaptive with ``maps``, root-sum-of-squares without."""
    xp = xp_of(x)
    if maps is None:
        return xp.sqrt(xp.sum(x * xp.conj(x), axis=coil_axis, keepdims=keepdims))
    num = xp.sum(x * xp.conj(maps), axis=coil_axis, keepdims=keepdims)
    den = xp.sum(maps * xp.conj(maps), axis=coil_axis, keepdims=keepdims)
    return divide_no_nan(num, den)


def divide_no_nan(a, b):
    """``a / b`` with 0 where ``b == 0`` (``tf.math.divide_no_nan``)."""
    xp = xp_of(a)
    nz = b != 0
    return xp.where(nz, a / xp.where(nz, b, 1), 0).astype(xp.result_type(a, b))


# --- trajectories -----------------------------------------------------------
def _radial_waveform(base_resolution, readout_os=2.0):
    samples = int(np.float32(base_resolution) * np.float32(readout_os) + np.float32(0.5))
    wave = np.arange((-samples) // 2, samples // 2, dtype=np.float32) / np.float32(samples)
    wave = np.stack([wave, np.zeros_like(wave)], -1)
    return wave * np.float32(2.0 * math.pi)


def _trajectory_angles(views, phases, ordering, tiny_number=7):
    """Angles ``(phases, views)`` (tfmri ``_trajectory_angles``, ``angle_range='full'``)."""
    f32 = np.float32
    pi, pi2 = f32(math.pi), f32(2.0 * math.pi)
    phi = f32(2.0) / (f32(1.0) + np.sqrt(f32(5.0)))
    phi_n = f32(1.0) / (phi + f32(tiny_number))

    def angles_2d(delta, amax, interleave=False):
        a = np.mod(np.arange(views * phases, dtype=np.float32) * f32(delta), f32(amax))
        return a.reshape(views, phases).T if interleave else a.reshape(phases, views)

    if ordering == "linear":
        return angles_2d(pi2 / f32(views * phases), pi2, interleave=True)
    if ordering == "golden":
        return angles_2d(phi * pi2, pi2)
    if ordering == "golden_half":
        return angles_2d(phi * pi, pi2)
    if ordering == "sorted":
        return np.sort(angles_2d(phi * pi2, pi2), axis=1)
    if ordering == "sorted_half":
        a = np.sort(angles_2d(phi * pi, pi), axis=1)
        out = [a[0]]
        for curr in a[1:]:
            curr = np.concatenate([curr, curr + f32(math.pi)])
            d = curr - out[-1][-1]
            start = int(np.argmin(np.where(d < 0, np.inf, d)))
            out.append(np.roll(curr, -start)[:views])
        return np.stack(out).astype(np.float32)
    if ordering == "tiny":
        return angles_2d(phi_n * pi2, pi2)
    if ordering == "tiny_half":
        return angles_2d(phi_n * pi, pi2)
    raise ValueError(f"unknown/unsupported radial ordering {ordering!r}")


def radial_trajectory(base_resolution, views, phases, ordering):
    """Radial k-space trajectory ``(phases, views, readout, 2)`` in [-pi, pi]."""
    wave = _radial_waveform(base_resolution)                                   # (S, 2), y = 0
    ang = _trajectory_angles(views, phases, ordering)[..., None]              # (P, V, 1)
    traj = np.stack([np.cos(ang) * wave[:, 0], np.sin(ang) * wave[:, 0]], -1).astype(np.float32)
    return from_numpy(traj)


def flatten_trajectory(traj):
    """``(..., views, readout, 2)`` -> ``(..., views*readout, 2)``."""
    return traj.reshape(traj.shape[:-3] + (-1, traj.shape[-1]))


def estimate_radial_density(traj, readout_os=2.0):
    """tfmri ``estimate_radial_density``: geometry-only density of ``(..., views, samples, rank)`` points."""
    xp = xp_of(traj)
    views, samples, rank = traj.shape[-3:]
    if rank != 2:
        raise NotImplementedError("only 2D radial density")
    radius = xp.linalg.norm(traj, axis=-1) / np.float32(math.pi) * np.float32(samples // 2)
    weights = xp.where(radius < 0.5, np.float32(1.0), np.float32(4.0) * radius).astype(np.float32)
    weights /= np.float32(views * readout_os ** 2)
    return divide_no_nan(xp.ones_like(weights), weights)
