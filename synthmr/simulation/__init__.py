"""Image -> multicoil k-space simulation (contrast, mask, coils, background phase, noise)."""

from synthmr.backend import get_backend

from .config import SimulationConfig


def simulate(video, config, rng, backend=None):
    """Simulate fully-sampled multicoil k-space from a 2-channel video.

    Args:
      video: ``(T, H, W, 2)`` (2d mode) or ``(T, Z, H, W, 2)`` (3d mode) values in [0, 255].
      config: :class:`SimulationConfig`.
      rng: :class:`synthmr.rng.RNG` (streams ``transforms`` and ``noise``).

    Returns:
      dict with ``kspace`` ``(C, T, Z, H, W)`` complex64, ``smaps`` ``(C, Z, H, W)``,
      ``image`` ``(T, Z, H, W)`` (object incl. background phase) and ``mask`` ``(Z, H, W)``.
    """
    backend = backend or get_backend()
    if config.mode == "3d":
        from .sim3d import simulate_3d
        return simulate_3d(video, config, rng, backend)
    return backend.simulation.simulate(video, config, rng)


__all__ = ["SimulationConfig", "simulate"]
