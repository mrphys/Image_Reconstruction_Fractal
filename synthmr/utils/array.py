"""Backend-free array helpers (numpy)."""

import numpy as np
from scipy.ndimage import zoom


def float_to_uint8(array, max, min):  # noqa: A002 - legacy signature
    """Scale ``[min, max]`` to integers ``[0, 255]`` (legacy: returns int, not uint8)."""
    scaled = (array - min) / (max - min)
    return (scaled * 255.0).astype(int)


def int255_to_float(array, max, min):  # noqa: A002 - legacy signature
    """Inverse of :func:`float_to_uint8`."""
    return (array.astype(float) / 255) * (max - min) + min


def scale_frames(img, required_frames=20):
    """Nearest-neighbour resample axis 0 to ``required_frames`` frames."""
    zooms = np.ones(img.ndim)
    zooms[0] = required_frames / img.shape[0]
    return zoom(img, zooms, order=0)


def circular_pad(img, padding_fraction=0.3):
    """Wrap ``padding_fraction`` of the frames from each end onto the other (axis 0)."""
    padding_size = int(img.shape[0] * padding_fraction)
    return np.concatenate([img[-padding_size:], img, img[:padding_size]], axis=0)


def crop_or_circular_pad(img, output_frames):
    """Crop axis 0 to ``output_frames`` or pad it by cyclic repetition."""
    if img.shape[0] >= output_frames:
        return img[:output_frames]
    reps = int(np.ceil(output_frames / img.shape[0]))
    return np.concatenate([img] * reps, axis=0)[:output_frames]


def complex_from_channels(x):
    """``(..., 2)`` real/imag -> complex; ``(..., 1)`` -> squeezed."""
    x = np.asarray(x)
    return x[..., 0] + 1j * x[..., 1] if x.shape[-1] == 2 else x[..., 0]


def channels_from_complex(z, axis=-1):
    """Complex -> real/imag stacked on ``axis``."""
    return np.stack([np.real(z), np.imag(z)], axis=axis)
