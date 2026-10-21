"""MRI helpers that need the backend (resolved from the active backend's ``utils`` module).

Available: ``convert_to_real``, ``convert_to_complex``, ``normalise_max_abs``,
``sample_noncartesian_kspace``, ``compute_trajectory_and_density``, ``LowPassScaler``.
"""

from synthmr.backend import get_backend


def __getattr__(name):
    return getattr(get_backend().utils, name)
