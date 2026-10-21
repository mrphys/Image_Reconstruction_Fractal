"""sigpy implementation of trajectories and the sampling operator (port of ``backend/tf/sampling.py``).

Spiral trajectories need the spiral-waveform library behind tfmri and are not available here;
load a precomputed spiral with ``kind='file'`` instead.
"""

import math

import numpy as np

from . import core


def radial(cfg):
    """Radial trajectory ``(phases, spokes*readout, 2)`` and dcw ``(phases, spokes*readout)``."""
    traj = core.radial_trajectory(cfg.base_resolution, cfg.spokes, cfg.phases, cfg.ordering)
    traj = core.flatten_trajectory(traj)
    dens = core.estimate_radial_density(traj)
    xp = core.xp_of(dens)
    return traj, core.divide_no_nan(xp.ones_like(dens), dens)


def make_trajectory(cfg):
    """``(traj, dcw)`` for a :class:`synthmr.sampling.TrajectoryConfig` subclass (in-plane part)."""
    if cfg.kind == "radial":
        traj, dcw = radial(cfg)
    elif cfg.kind == "spiral":
        raise NotImplementedError("spiral trajectories need the TF backend; with sigpy use kind='file' "
                                  "(see synthmr.sampling.files)")
    else:
        raise ValueError(f"unknown trajectory kind {cfg.kind!r}")
    if cfg.reverse:
        traj = core.xp_of(traj).ascontiguousarray(traj[..., ::-1])
    return traj, dcw


def tile_to_phases(traj, dcw, phases):
    """Repeat a trajectory along time to cover ``phases`` frames (legacy ``reps`` tiling)."""
    xp = core.xp_of(traj)
    reps = int(math.ceil(phases / traj.shape[0]))
    return xp.tile(traj, (reps, 1, 1))[:phases], xp.tile(dcw, (reps, 1))[:phases]


class SamplingOperator:
    """Non-Cartesian sampling of multicoil image series (same contract as the TF operator).

    Through-plane encoders:
      ``none``          2D: image ``(T, C, H, W)`` <-> k-space ``(T, C, R)``.
      ``cartesian_kz``  stack-of-*: image ``(T, C, Z, H, W)``; centred ortho FFT along Z, then the
                        in-plane trajectory per kz partition -> k-space ``(T, C, Z, R)``.
    Forward and adjoint include the legacy ``1/sqrt(H*W)`` scaling; density compensation is
    applied separately (``apply_dcw``). The NUFFT loops over frames (and partitions for ``kz_traj``).
    """

    def __init__(self, traj, dcw, image_shape, through_plane="none", kz_traj=None):
        self.traj, self.dcw = traj, dcw
        self.image_shape = [int(n) for n in image_shape]
        self.through_plane = through_plane
        # Optional per-partition trajectories (T, Z, R, 2) e.g. rotated stars; default: same star for all kz.
        self.kz_traj = kz_traj
        self._sqrt_npixel = np.float32(math.sqrt(self.image_shape[0] * self.image_shape[1]))

    def _points(self, ndim_extra):
        if self.kz_traj is not None:
            return self.kz_traj[:, None]                                    # (T, 1, Z, R, 2)
        t = self.traj[:, None]                                              # (T, 1, R, 2)
        return t[:, :, None] if ndim_extra else t                          # (T, 1, 1, R, 2)

    def forward(self, image):
        image = image.astype(np.complex64)
        kz = self.through_plane == "cartesian_kz"
        if kz:
            image = core.fft(image, axes=[-3])
        return (core.nufft_forward(image, self._points(kz)) / self._sqrt_npixel).astype(np.complex64)

    def adjoint(self, kspace):
        kz = self.through_plane == "cartesian_kz"
        image = core.nufft_adjoint(kspace.astype(np.complex64), self._points(kz), self.image_shape) / self._sqrt_npixel
        if kz:
            image = core.ifft(image, axes=[-3])
        return image.astype(np.complex64)

    def apply_dcw(self, kspace):
        dcw = self.dcw.astype(np.complex64)
        if self.through_plane == "cartesian_kz":
            return kspace * dcw[:, None, None, :]
        return kspace * dcw[..., None, :]
