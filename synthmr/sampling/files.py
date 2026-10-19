"""Load an in-plane trajectory and its density from ``.npy`` files (backend-free).

Accepted trajectory shapes (last axis = 2 coordinates, same order as the image axes ``(H, W)``):
``(R, 2)`` one frame, ``(V, S, 2)`` one frame of views x samples, ``(T, R, 2)`` and ``(T, V, S, 2)``.
The density array has the trajectory's shape without the last axis. Everything is flattened to
``traj (T, R, 2)`` in radians/pixel ([-pi, pi]) and ``dcw (T, R)``.
"""

import numpy as np


def _flatten(traj, dens):
    if traj.ndim < 2 or traj.shape[-1] != 2:
        raise ValueError(f"trajectory must have shape (..., 2), got {traj.shape}")
    if dens.shape != traj.shape[:-1]:
        raise ValueError(f"density shape {dens.shape} must equal trajectory shape without the last axis "
                         f"{traj.shape[:-1]}")
    if traj.ndim == 2:                                    # (R, 2)
        traj, dens = traj[None], dens[None]
    elif traj.ndim == 4:                                  # (T, V, S, 2)
        traj, dens = traj.reshape(traj.shape[0], -1, 2), dens.reshape(dens.shape[0], -1)
    elif traj.ndim != 3:
        raise ValueError(f"unsupported trajectory rank {traj.ndim}: use (R,2), (V,S,2), (T,R,2) or (T,V,S,2)")
    return traj, dens


def load_trajectory_files(cfg):
    """``(traj (T, R, 2) float32, dcw (T, R) float32)`` numpy arrays for a ``FileTrajectoryConfig``.

    A 3-D trajectory is read as ``(T, R, 2)`` unless ``cfg.views_axis`` is set, in which case
    ``(V, S, 2)`` (one frame of ``V`` views) is assumed.
    """
    traj = np.load(cfg.traj_path).astype(np.float64)
    dens = np.load(cfg.density_path).astype(np.float64)
    if traj.ndim == 3 and cfg.views_axis:
        traj, dens = traj.reshape(1, -1, 2), dens.reshape(1, -1)
    traj, dens = _flatten(traj, dens)
    if cfg.traj_units == "pixels":                        # sigpy / extreme_mri: [-N/2, N/2]
        traj = traj * (2 * np.pi / cfg.base_resolution)
    elif cfg.traj_units != "radians":
        raise ValueError(f"unknown traj_units {cfg.traj_units!r}; use 'radians' or 'pixels'")
    if np.abs(traj).max() > np.pi * (1 + 1e-5):
        raise ValueError(f"trajectory exceeds [-pi, pi] (max |k| = {np.abs(traj).max():.4f}); "
                         "check traj_units / base_resolution")
    if cfg.density_kind == "density":
        with np.errstate(divide="ignore"):
            dcw = np.where(dens != 0, 1.0 / np.where(dens != 0, dens, 1.0), 0.0)
    elif cfg.density_kind == "dcw":
        dcw = dens
    else:
        raise ValueError(f"unknown density_kind {cfg.density_kind!r}; use 'density' or 'dcw'")
    traj = np.clip(traj, -np.pi, np.pi)
    if cfg.reverse:
        traj = traj[..., ::-1]
    return np.ascontiguousarray(traj, dtype=np.float32), np.ascontiguousarray(dcw, dtype=np.float32)
