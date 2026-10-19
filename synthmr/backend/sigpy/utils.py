"""sigpy versions of the real-MRI helpers in ``backend/tf/utils.py``."""

import math

from . import core


def convert_to_real(complex_tensor, axis=0):
    """Concatenate real and imaginary parts along ``axis`` (legacy default: axis 0)."""
    xp = core.xp_of(complex_tensor)
    return xp.concatenate([complex_tensor.real, complex_tensor.imag], axis=axis)


def convert_to_complex(stacked_tensor):
    """Inverse of channel-last real/imag stacking."""
    n = stacked_tensor.shape[-1] // 2
    return stacked_tensor[..., :n] + 1j * stacked_tensor[..., n:]


def normalise_max_abs(image):
    """Divide by max magnitude (legacy ``normalise``)."""
    return image / core.xp_of(image).abs(image).max().astype(image.dtype)


def sample_noncartesian_kspace(image, trajectory):
    """Type-2 NUFFT scaled by ``1/sqrt(H*W)``."""
    return core.nufft_forward(image, trajectory) / math.sqrt(image.shape[-1] * image.shape[-2])


def compute_trajectory_and_density(image_shape, extra_shape, views, ordering):
    """Flattened radial trajectory and density for ``extra_shape[-1]`` phases."""
    traj = core.radial_trajectory(image_shape[-1], views, extra_shape[-1], ordering)
    density = core.estimate_radial_density(traj)
    return core.flatten_trajectory(traj), density.reshape(density.shape[:-2] + (-1,))


class LowPassScaler:
    """Not available: the TF version relies on ``tfmri.signal.filter_kspace`` / ``tfmri.recon.adjoint``."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LowPassScaler needs the TF backend")
