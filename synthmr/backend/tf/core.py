"""Core MRI / array ops for the TensorFlow backend.

Every function wraps exactly the TF call used by the legacy code so that
outputs stay bit-identical. Semantics are documented in docs/BACKEND.md.
"""

import contextlib

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_mri as tfmri
import tensorflow_nufft as tfft

NAME = "tf"


# --- dtype / conversion -----------------------------------------------------
def as_dtype(dtype):
    """Accept ``"float32"``, ``np.float32`` or ``tf.float32``."""
    return tf.as_dtype(dtype)


def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def from_numpy(x, dtype=None):
    return tf.convert_to_tensor(x, dtype=None if dtype is None else as_dtype(dtype))


def device(spec):
    """Context manager placing ops on ``spec`` (``"gpu"``, ``"gpu:1"``, ``"cpu"``); ``None`` = default."""
    if spec is None:
        return contextlib.nullcontext()
    spec = spec if spec.startswith("/") else "/" + spec
    return tf.device(spec if ":" in spec else spec + ":0")


# --- random numbers ---------------------------------------------------------
class Generator:
    """Seeded counter-based (philox) generator, same algorithm as the legacy ``rg``.

    Method signatures mirror ``tf.random.Generator`` so legacy call sites work unchanged.
    """

    def __init__(self, seed):
        self.seed = seed
        self._gen = tf.random.Generator.from_seed(seed, alg="philox")

    def reset(self, seed=None):
        self.seed = self.seed if seed is None else seed
        self._gen.reset_from_seed(self.seed)

    def uniform(self, shape=(), minval=0, maxval=None, dtype="float32"):
        return self._gen.uniform(shape, minval=minval, maxval=maxval, dtype=as_dtype(dtype))

    def normal(self, shape=(), mean=0.0, stddev=1.0, dtype="float32"):
        return self._gen.normal(shape=shape, mean=mean, stddev=stddev, dtype=as_dtype(dtype))


# --- Fourier transforms -----------------------------------------------------
def fft(x, axes, norm="ortho", shift=True):
    """Centred n-D FFT over ``axes``."""
    return tfmri.signal.fft(x, axes=axes, norm=norm, shift=shift)


def ifft(x, axes, norm="ortho", shift=True):
    """Centred n-D inverse FFT over ``axes``."""
    return tfmri.signal.ifft(x, axes=axes, norm=norm, shift=shift)


def nufft_forward(image, traj):
    """Type-2 NUFFT (grid -> samples), unscaled. ``traj`` ``(..., R, ndim)`` in [-pi, pi]."""
    return tfft.nufft(image, traj, transform_type="type_2", fft_direction="forward")


def nufft_adjoint(kspace, traj, grid_shape):
    """Type-1 NUFFT (samples -> grid), unscaled."""
    return tfft.nufft(kspace, traj, grid_shape=grid_shape, transform_type="type_1",
                      fft_direction="backward")


# --- image ops --------------------------------------------------------------
def crop_or_pad(x, shape):
    """Centred crop/pad; ``-1`` keeps a dimension."""
    return tfmri.resize_with_crop_or_pad(x, shape=shape)


def scale_by_min_max(x):
    return tfmri.scale_by_min_max(x)


def resize(images, size, method="bicubic"):
    """``tf.image.resize`` on ``(N, H, W, C)`` or ``(H, W, C)`` images."""
    return tf.image.resize(images, size, method)


def translate(images, translations, interpolation="bilinear"):
    """Shift ``(H, W, C)`` / ``(N, H, W, C)`` images by ``translations`` ``[dx, dy]`` pixels."""
    return tfa.image.translate(images, translations, interpolation)


# --- coils ------------------------------------------------------------------
def coil_compressor_svd(coil_axis, out_coils):
    """Object with ``fit(x)`` / ``transform(x)`` (SVD coil compression)."""
    return tfmri.coils.CoilCompressorSVD(coil_axis=coil_axis, out_coils=out_coils)


def estimate_sensitivities(x, coil_axis=-1, method="walsh"):
    """Coil maps; all non-coil axes are spatial (rank 2 or 3)."""
    return tfmri.coils.estimate_sensitivities(x, coil_axis=coil_axis, method=method)


def combine_coils(x, maps=None, coil_axis=-1, keepdims=False):
    """Coil combination: adaptive with ``maps``, root-sum-of-squares without."""
    return tfmri.coils.combine_coils(x, maps=maps, coil_axis=coil_axis, keepdims=keepdims)


# --- trajectories -----------------------------------------------------------
def radial_trajectory(base_resolution, views, phases, ordering):
    """Radial k-space trajectory ``(phases, views, readout, 2)`` in [-pi, pi]."""
    return tfmri.sampling.radial_trajectory(base_resolution=base_resolution, views=views,
                                            phases=phases, ordering=ordering)


def flatten_trajectory(traj):
    """``(..., views, readout, 2)`` -> ``(..., views*readout, 2)``."""
    return tfmri.sampling.flatten_trajectory(traj)


def estimate_radial_density(traj):
    return tfmri.sampling.estimate_radial_density(traj)
