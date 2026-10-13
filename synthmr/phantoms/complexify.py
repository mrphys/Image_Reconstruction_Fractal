"""Turn scalar fractal fields into complex images / 2-channel "videos".

Two methods:

``sine_wave``  The base-repo (paper) method: normalise an escape-time field, map it
               through two random sine waves (real/imag) and blur+sharpen each.
               Output is a float "video" in roughly [0, 255] with channels (re, im).
``recipe``     The new framework: randomly weighted mixes of several fractal fields
               for real and imaginary parts (see ``recipes.py``), recentred to [-1, 1].

``complex_to_video`` converts a complex image to the same 2-channel [0, 255] range
consumed by the simulation stage.

Random draws use ``rng`` (a ``np.random.Generator``). Passing the ``np.random``
module reproduces the legacy code, which used the global numpy state.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .fractal.channels import compose_complex


def make_sine_wave(amplitude=1.0, frequency=1.0, phase=0.0):
    """``t -> int(128 * (1 + a sin(2 pi f t + phase)))``."""
    def sine_wave(t):
        return (128 * (1 + amplitude * np.sin(2 * np.pi * frequency * t + phase))).astype(int)
    return sine_wave


def norm(x):
    """Min-max normalisation to [0, 1]; constant input gives zeros."""
    xmin, xmax = x.min(), x.max()
    if xmax == xmin:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)


def blur_and_sharpen(image, sigma=1.0, alpha=50.0):
    """Gaussian blur then unsharp mask.

    Note (legacy behaviour, kept): for integer input ``gaussian_filter`` returns
    integers, so the blurred base and detail are truncated before sharpening.
    """
    base = gaussian_filter(image, sigma)
    detail = base - gaussian_filter(base, 0.1)
    return base + alpha * detail


def sine_wave(escape, rng=np.random):
    """Base-repo ``make_complex_fractal``: escape-time ``(..., H, W)`` -> video ``(..., H, W, 2)``."""
    ex = norm(escape)
    freq1 = rng.uniform(0.25, 1.0)
    freq2 = 0.25
    if rng.random() > 0.5:
        freq1, freq2 = freq2, freq1
    wave1 = make_sine_wave(1, freq1, rng.uniform(0, 2))
    wave2 = make_sine_wave(1, freq2, rng.uniform(0, 2))
    real = blur_and_sharpen(wave1(ex), rng.uniform(0.2, 0.4))
    imag = blur_and_sharpen(wave2(ex), rng.uniform(0.2, 0.4))
    return np.stack((real, imag), axis=-1)


def recipe_complex(fields_real, fields_imag, weights_real, weights_imag, norm_axes=(-2, -1)):
    """Complex image from rendered real/imag field lists, recentred from [0, 1] to [-1, 1].

    ``norm_axes=(-2, -1)`` normalises every 2-D frame independently (legacy
    ``julia_generator_new_grasp``); ``None`` normalises over the whole block, which
    keeps intensities consistent across slices and time for 3D+t.
    """
    field = compose_complex(fields_real, fields_imag, weights_real, weights_imag, axes=norm_axes)
    return (2 * field - (1 + 1j)).astype(np.complex128)


def complex_to_video(volume):
    """Complex ``(...)`` in [-1, 1] -> float video ``(..., 2)`` in [0, 255] (re, im)."""
    return np.stack([volume.real, volume.imag], axis=-1) * 127.5 + 127.5


def _norm_inverted(x):
    """``(max - x) / (max - min)`` as used by the app/process_data_old variants."""
    return (x.max() - x) / (x.max() - x.min())


def sine_wave_smooth(escape, rng=np.random):
    """``app/process_data_old.make_complex_fractal``: inverted norm, Gaussian blur 0.8, f = 0.25."""
    wave1 = make_sine_wave(amplitude=1, frequency=0.25, phase=rng.uniform(0, 1))
    wave2 = make_sine_wave(amplitude=1, frequency=0.25, phase=rng.uniform(0, 1))
    ex = gaussian_filter(_norm_inverted(escape).astype(float), sigma=0.8)
    return np.stack([wave1(ex), wave2(ex)], axis=-1)


def sine_wave_spiral(escape, rng=np.random):
    """``make_complex_fractal_spiral``: one blurred sine-mapped channel duplicated into re and im."""
    wave1 = make_sine_wave(amplitude=1, frequency=rng.uniform(0.2, 0.5), phase=rng.uniform(0, 1))
    temp = wave1(_norm_inverted(gaussian_filter(escape, sigma=0.8)))
    return np.stack([temp, temp], axis=-1)


METHODS = {
    "sine_wave": sine_wave,
    "sine_wave_smooth": sine_wave_smooth,
    "sine_wave_spiral": sine_wave_spiral,
}
