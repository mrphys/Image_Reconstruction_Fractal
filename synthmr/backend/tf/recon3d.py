"""TF implementation of the 3D+t recon stage (stack-of-* sampling, 3D coil maps)."""

import numpy as np
import tensorflow as tf
import tensorflow_mri as tfmr

from synthmr.recon.coil_maps import tukey_window_nd
from synthmr.sampling.encoders import kz_trajectories

from .recon import format_pair
from .sampling import SamplingOperator, tile_to_phases


def _fs_image_3d(kspace, image_shape, phases):
    """Fully sampled image ``(T, C, Z, H, W)`` from in-plane Cartesian k-space, cropped and time-tiled."""
    image = tfmr.signal.ifft(kspace, axes=[-2, -1], norm='ortho', shift=True)
    image = tfmr.resize_with_crop_or_pad(image, [-1, -1, -1, *image_shape])
    reps = int(np.ceil(phases / image.shape[0]))
    return tf.tile(image, [reps, 1, 1, 1, 1])[:phases]


def _compress(kspace, uskspace, out_coils):
    """SVD coil compression fitted on undersampled data. kspace ``(T,C,Z,H,W)``, uskspace ``(T,C,Z,R)``."""
    cc = tfmr.coils.CoilCompressorSVD(coil_axis=-3, out_coils=out_coils)
    cc.fit(tf.transpose(uskspace, [0, 2, 1, 3])[..., None])                     # (T, Z, C, R, 1)
    kspace = tf.transpose(cc.transform(tf.transpose(kspace, [0, 2, 1, 3, 4])), [0, 2, 1, 3, 4])
    usk = cc.transform(tf.transpose(uskspace, [0, 2, 1, 3])[..., None])[..., 0]
    return kspace, tf.transpose(usk, [0, 2, 1, 3])


def anchor_phase(maps, ref_coil=0):
    """Remove the unanchored Walsh eigenvector phase by referencing it to one coil."""
    ref = maps[..., ref_coil:ref_coil + 1]
    return maps * tf.math.exp(-1j * tf.cast(tf.math.angle(ref), maps.dtype))


def time_avg_tukey_maps(zfill, cfg):
    """3D coil maps ``(Z, H, W, C)`` from the time-summed zero-filled data ``(T, C, Z, H, W)``.

    Summing over all frames fills k-space far more densely than any single frame; a Tukey
    low-pass in 3D k-space then keeps the smooth, low-resolution content from which 3D Walsh
    maps are estimated.
    """
    avg = tf.reduce_sum(zfill, axis=0)                                        # (C, Z, H, W)
    k = tfmr.signal.fft(avg, axes=[-3, -2, -1], norm='ortho', shift=True)
    win = tukey_window_nd(avg.shape[1:], cfg.tukey_width, cfg.tukey_alpha)
    low = tfmr.signal.ifft(k * tf.cast(win, k.dtype)[None], axes=[-3, -2, -1], norm='ortho', shift=True)
    maps = tfmr.coils.estimate_sensitivities(tf.transpose(low, [1, 2, 3, 0]), coil_axis=-1, method='walsh')
    return anchor_phase(maps) if cfg.anchor_phase else maps


def combine_with_maps(x, maps):
    """Adaptive combination of ``x`` ``(..., Z, H, W, C)`` with maps ``(Z, H, W, C)`` -> ``(..., Z, H, W)``."""
    num = tf.reduce_sum(tf.math.conj(maps) * x, axis=-1)
    den = tf.cast(tf.reduce_sum(tf.abs(maps) ** 2, axis=-1), num.dtype)
    return tf.math.divide_no_nan(num, den)


def reconstruct_3d(kspace, traj, dcw, cfg, kz_rotation="aligned"):
    """3D+t pair from canonical k-space ``(C, T, Z, H, W)``.

    Returns zfill ``(T, Z, H, W, ch)``, image ``(T, Z, H, W)`` (abs) or ``(T, Z, H, W, 2)``.
    """
    if cfg.coil_maps != "time_avg_tukey_3d":
        raise ValueError(f"3D path supports coil_maps='time_avg_tukey_3d', got {cfg.coil_maps!r}")
    kspace = tf.transpose(kspace, [1, 0, 2, 3, 4])                             # (T, C, Z, H, W)
    image_shape = [cfg.base_resolution, cfg.base_resolution]
    z = kspace.shape[2]
    traj, dcw = tile_to_phases(traj, dcw, cfg.phases)
    kz_traj = kz_trajectories(traj.numpy(), z, kz_rotation)
    op = SamplingOperator(traj, dcw, image_shape, through_plane="cartesian_kz",
                          kz_traj=None if kz_traj is None else tf.constant(kz_traj))

    image = _fs_image_3d(kspace, image_shape, cfg.phases)
    uskspace = op.apply_dcw(op.forward(image))                                 # (T, C, Z, R)
    if cfg.input_format == 'coil_compressed':
        kspace, uskspace = _compress(kspace, uskspace, cfg.input_coils)
        image = _fs_image_3d(kspace, image_shape, cfg.phases)

    sos = tf.sqrt(tf.reduce_sum(tf.abs(image) ** 2, axis=1))                   # (T, Z, H, W)
    scale = tf.cast(tf.reduce_max(sos), image.dtype)
    image, uskspace = image / scale, uskspace / scale
    zfill = op.adjoint(uskspace)                                               # (T, C, Z, H, W)
    if cfg.normalize_input:
        zfill = tfmr.scale_by_min_max(zfill)
    maps = time_avg_tukey_maps(zfill, cfg)                                     # (Z, H, W, C)
    zfill = combine_with_maps(tf.transpose(zfill, [0, 2, 3, 4, 1]), maps)[..., None]   # (T, Z, H, W, 1)
    if cfg.target_combine == "maps":
        target = combine_with_maps(tf.transpose(image, [0, 2, 3, 4, 1]), maps)
    else:
        target = tf.cast(sos / tf.cast(scale, sos.dtype), image.dtype)
    zfill, target = format_pair(zfill, target, cfg)
    return {'zfill': zfill, 'image': target, 'uskspace': uskspace, 'traj': traj, 'dcw': dcw, 'maps': maps}
