"""sigpy implementation of the 3D+t recon stage (port of ``backend/tf/recon3d.py``), plus JSENSE 3D maps."""

import math

import numpy as np

from synthmr.recon.coil_maps import tukey_window_nd
from synthmr.sampling.encoders import kz_trajectories

from . import core
from .recon import format_pair, jsense_maps
from .sampling import SamplingOperator, tile_to_phases


def _fs_image_3d(kspace, image_shape, phases):
    """Fully sampled image ``(T, C, Z, H, W)`` from in-plane Cartesian k-space, cropped and time-tiled."""
    xp = core.xp_of(kspace)
    image = core.ifft(kspace, axes=[-2, -1])
    image = core.crop_or_pad(image, [-1, -1, -1, *image_shape])
    reps = int(np.ceil(phases / image.shape[0]))
    return xp.tile(image, (reps, 1, 1, 1, 1))[:phases]


def _compress(kspace, uskspace, out_coils):
    """SVD coil compression fitted on undersampled data. kspace ``(T,C,Z,H,W)``, uskspace ``(T,C,Z,R)``."""
    xp = core.xp_of(kspace)
    cc = core.coil_compressor_svd(coil_axis=-3, out_coils=out_coils)
    cc.fit(xp.transpose(uskspace, (0, 2, 1, 3))[..., None])                   # (T, Z, C, R, 1)
    kspace = xp.transpose(cc.transform(xp.transpose(kspace, (0, 2, 1, 3, 4))), (0, 2, 1, 3, 4))
    usk = cc.transform(xp.transpose(uskspace, (0, 2, 1, 3))[..., None])[..., 0]
    return kspace, xp.transpose(usk, (0, 2, 1, 3))


def anchor_phase(maps, ref_coil=0):
    """Remove the unanchored eigenvector phase by referencing it to one coil."""
    xp = core.xp_of(maps)
    ref = maps[..., ref_coil:ref_coil + 1]
    return maps * xp.exp(-1j * xp.angle(ref)).astype(maps.dtype)


def time_avg_tukey_maps(zfill, cfg):
    """3D coil maps ``(Z, H, W, C)`` from the time-summed zero-filled data ``(T, C, Z, H, W)``.

    Summing over all frames fills k-space far more densely than any single frame; a Tukey
    low-pass in 3D k-space then keeps the smooth, low-resolution content from which 3D Walsh
    maps are estimated.
    """
    xp = core.xp_of(zfill)
    avg = xp.sum(zfill, axis=0)                                               # (C, Z, H, W)
    k = core.fft(avg, axes=[-3, -2, -1])
    win = core.from_numpy(tukey_window_nd(avg.shape[1:], cfg.tukey_width, cfg.tukey_alpha))
    low = core.ifft(k * win.astype(k.dtype)[None], axes=[-3, -2, -1])
    maps = core.estimate_sensitivities(xp.transpose(low, (1, 2, 3, 0)), coil_axis=-1)
    return anchor_phase(maps) if cfg.anchor_phase else maps


def jsense_maps_3d(uskspace, traj, kz_traj, dcw, image_shape, z, cfg):
    """3D JSENSE maps ``(Z, H, W, C)`` from stack-of-* data ``(T, C, Z, R)`` of all frames.

    Partition ``p`` holds kz = ``p - Z // 2`` (centred FFT), so the 3D coordinates are
    ``[kz, ky, kx]`` in pixel units, matching the image axes ``(Z, H, W)``.
    """
    xp = core.xp_of(uskspace)
    t, _, _, r = uskspace.shape
    inplane = kz_traj if kz_traj is not None else xp.broadcast_to(traj[:, None], (t, z, r, 2))
    inplane = inplane * np.float32(cfg.base_resolution / (2 * math.pi))
    kz = xp.broadcast_to((xp.arange(z, dtype=np.float32) - z // 2)[None, :, None, None], (t, z, r, 1))
    coord = xp.concatenate([kz, inplane], axis=-1)                            # (T, Z, R, 3)
    maps = jsense_maps(xp.transpose(uskspace, (1, 0, 2, 3)), coord, dcw[:, None, :], [z, *image_shape], cfg)
    maps = xp.transpose(maps, (1, 2, 3, 0))                                   # (Z, H, W, C)
    return anchor_phase(maps) if cfg.anchor_phase else maps


def combine_with_maps(x, maps):
    """Adaptive combination of ``x`` ``(..., Z, H, W, C)`` with maps ``(Z, H, W, C)`` -> ``(..., Z, H, W)``."""
    xp = core.xp_of(x)
    num = xp.sum(xp.conj(maps) * x, axis=-1)
    den = xp.sum(xp.abs(maps) ** 2, axis=-1).astype(num.dtype)
    return core.divide_no_nan(num, den)


def reconstruct_3d(kspace, traj, dcw, cfg, kz_rotation="aligned"):
    """3D+t pair from canonical k-space ``(C, T, Z, H, W)``.

    Returns zfill ``(T, Z, H, W, ch)``, image ``(T, Z, H, W)`` (abs) or ``(T, Z, H, W, 2)``.
    With ``input_combine='none'`` the coils stay in the zfill channels (ch = 2C, or C for abs).
    """
    if cfg.coil_maps not in ("time_avg_tukey_3d", "jsense"):
        raise ValueError(f"3D path supports coil_maps='time_avg_tukey_3d' or 'jsense', got {cfg.coil_maps!r}")
    if cfg.input_combine not in ("maps", "none"):
        raise ValueError(f"unknown input_combine {cfg.input_combine!r}; use 'maps' or 'none'")
    xp = core.xp_of(kspace)
    kspace = xp.transpose(kspace, (1, 0, 2, 3, 4))                             # (T, C, Z, H, W)
    image_shape = [cfg.base_resolution, cfg.base_resolution]
    z = kspace.shape[2]
    traj, dcw = tile_to_phases(traj, dcw, cfg.phases)
    kz_traj = kz_trajectories(core.to_numpy(traj), z, kz_rotation)
    kz_traj = None if kz_traj is None else core.from_numpy(kz_traj)
    op = SamplingOperator(traj, dcw, image_shape, through_plane="cartesian_kz", kz_traj=kz_traj)

    image = _fs_image_3d(kspace, image_shape, cfg.phases)
    uskspace = op.apply_dcw(op.forward(image))                                 # (T, C, Z, R)
    if cfg.input_format == 'coil_compressed':
        kspace, uskspace = _compress(kspace, uskspace, cfg.input_coils)
        image = _fs_image_3d(kspace, image_shape, cfg.phases)

    sos = xp.sqrt(xp.sum(xp.abs(image) ** 2, axis=1))                          # (T, Z, H, W)
    scale = sos.max().astype(image.dtype)
    image, uskspace = image / scale, uskspace / scale
    zfill = op.adjoint(uskspace)                                               # (T, C, Z, H, W)
    if cfg.normalize_input:
        zfill = core.scale_by_min_max(zfill)
    if cfg.coil_maps == "jsense":
        maps = jsense_maps_3d(uskspace, traj, kz_traj, dcw, image_shape, z, cfg)
    else:
        maps = time_avg_tukey_maps(zfill, cfg)                                 # (Z, H, W, C)
    zfill = xp.transpose(zfill, (0, 2, 3, 4, 1))                               # (T, Z, H, W, C)
    if cfg.input_combine == "maps":
        zfill = combine_with_maps(zfill, maps)[..., None]                      # (T, Z, H, W, 1)
    if cfg.target_combine == "maps":
        target = combine_with_maps(xp.transpose(image, (0, 2, 3, 4, 1)), maps)
    else:
        target = (sos / scale.real).astype(image.dtype)
    zfill, target = format_pair(zfill, target, cfg)
    return {'zfill': zfill, 'image': target, 'uskspace': uskspace, 'traj': traj, 'dcw': dcw, 'maps': maps}
