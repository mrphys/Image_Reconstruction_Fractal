"""Undersampling, coil compression, coil-map estimation/combination, output formats, augmentation."""

from dataclasses import dataclass
from typing import Optional

from synthmr.backend import get_backend


@dataclass
class ReconConfig:
    """Mirrors legacy ``preprocessing_multicoil_noselect.config_base_preproc`` (+ coil-map strategy)."""

    base_resolution: int = 256
    phases: int = 32
    roll: int = 0
    input_format: Optional[str] = "coil_compressed"   # coil_compressed | abs | other (no compression)
    input_coils: int = 10
    output_format: Optional[str] = "abs"              # abs | None (complex as 2 channels)
    normalize_input: bool = True
    gfilt: Optional[float] = None
    # walsh_xyt_legacy: Walsh on (H, W, T, C), i.e. time treated as a spatial axis (legacy 2D).
    # time_avg_tukey_3d: maps from the time-averaged, Tukey-filtered low-res volume (3D+t).
    coil_maps: str = "walsh_xyt_legacy"
    tukey_alpha: float = 0.5
    tukey_width: tuple = (1.0, 0.25, 0.25)           # k-space support kept (fraction) along (Z, Y, X)
    anchor_phase: bool = True                         # 3D maps: reference phase to coil 0 (legacy 2D: unanchored)
    target_combine: str = "sos"                       # sos (magnitude) | maps (complex, same maps as input)
    # maps: zfill coil-combined with the estimated maps (1 complex channel).
    # none: coils kept, folded into channels as [re_c1..re_cC, im_c1..im_cC] (abs: C channels).
    input_combine: str = "maps"

    def __post_init__(self):
        self.tukey_width = tuple(self.tukey_width)

    @classmethod
    def from_legacy(cls, d):
        return cls(**d)


@dataclass
class AugmentConfig:
    """Legacy ``rolling_fn`` arguments."""

    phases: int = 24
    roll: int = 1
    rotation: int = 0
    z_flip: int = 0          # 3D only


def reconstruct(kspace, traj, dcw, config, through_plane="none", kz_rotation="aligned", backend=None):
    """Undersample fully-sampled ``kspace`` ``(C, T, Z, H, W)`` on ``traj`` and build the pair.

    ``through_plane``/``kz_rotation`` come from the :class:`TrajectoryConfig`
    (``none`` = 2D, ``cartesian_kz`` = stack-of-stars for 3D+t).

    Returns dict with ``zfill`` (network input), ``image`` (target), ``uskspace``,
    ``traj``, ``dcw`` and (if estimated) ``maps``.
    """
    return (backend or get_backend()).recon.reconstruct(kspace, traj, dcw, config, through_plane, kz_rotation)


def augment(zfill, image, config, rng, backend=None):
    """Shared random rotation / temporal roll of input and target (``rng.augment``).

    3D+t inputs ``(T, Z, H, W, ch)`` use the volume-consistent numpy implementation.
    """
    backend = backend or get_backend()
    if len(zfill.shape) == 5:
        from .augment3d import augment_3d
        return augment_3d(backend.to_numpy(zfill), backend.to_numpy(image), config, rng)
    return backend.recon.augment(zfill, image, config, rng)


__all__ = ["ReconConfig", "AugmentConfig", "reconstruct", "augment"]
