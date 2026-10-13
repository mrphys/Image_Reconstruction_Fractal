
"""Convert scalar fractal fields into real, complex and multichannel images."""

import numpy as np


def normalise(x, low=None, high=None, eps=1e-8):
    """Min-max normalise an array, optionally using explicit bounds."""
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x) if low is None else low)
    hi = float(np.max(x) if high is None else high)
    return (x - lo) / (hi - lo + eps)


def remap(x, mode="normalise", **kwargs):
    """Apply one of the notebook-inspired scalar remappings."""
    x = np.asarray(x, dtype=np.float32)
    if mode == "normalise":
        return normalise(x, **kwargs)
    if mode == "log":
        return np.log1p(np.maximum(x, 0))
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unknown remap mode {mode!r}")


def compose_channel(fields, weights=None, remap_mode="normalise"):
    """Combine one or more scalar fields into a single channel."""
    arr = [normalise(f) for f in fields]
    if weights is None:
        weights = np.ones(len(arr), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(weights) != len(arr):
        raise ValueError("weights must have one value per field")
    out = sum(w * f for w, f in zip(weights, arr))
    if np.sum(np.abs(weights)) > 0:
        out /= np.sum(np.abs(weights))
    return remap(out, remap_mode)


def compose_complex(real_fields, imag_fields, real_weights=None, imag_weights=None):
    """Create a complex image from independently mixed real and imaginary fields.

    This formalises the notebook pattern::

        real = norm(field_a) + norm(field_b)
        imag = norm(field_c) + norm(field_b)
        image = real + 1j * imag
    """
    real = compose_channel(real_fields, real_weights)
    imag = compose_channel(imag_fields, imag_weights)
    return real.astype(np.float32) + 1j * imag.astype(np.float32)


def complex_channels(z, include_mag_phase=True):
    """Return a channel-first representation of a complex image.

    With ``include_mag_phase=True`` the channels are
    ``[real, imag, magnitude, phase]``.
    """
    z = np.asarray(z)
    channels = [np.real(z), np.imag(z)]
    if include_mag_phase:
        channels.extend([np.abs(z), np.angle(z)])
    return np.stack(channels, axis=0).astype(np.float32)


def stack_channels(*channels):
    """Stack arbitrary scalar channels as ``(C, H, W)``."""
    arrays = [np.asarray(c, dtype=np.float32) for c in channels]
    if not arrays:
        raise ValueError("At least one channel is required")
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("All channels must have the same shape")
    return np.stack(arrays, axis=0)
