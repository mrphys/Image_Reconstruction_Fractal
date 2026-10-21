"""Volume-consistent augmentation for 3D+t pairs (backend-free)."""

import numpy as np


def _draw_int(gen, hi):
    return int(np.asarray(gen.uniform(shape=(), minval=0, maxval=hi, dtype="int32")))


def augment_3d(zfill, image, cfg, rng):
    """zfill ``(T, Z, H, W, ch)``, image ``(T, Z, H, W[, ch])`` -> same layout, ``cfg.phases`` frames.

    One in-plane rot90, one optional z-flip and one temporal roll are drawn per sample and
    applied identically to every slice and frame of input and target.
    """
    gen = rng.augment
    if image.ndim == 4:
        image = image[..., None]
    if cfg.rotation > 0:
        k = _draw_int(gen, 4)
        zfill, image = np.rot90(zfill, k, axes=(2, 3)), np.rot90(image, k, axes=(2, 3))
    if cfg.z_flip > 0 and _draw_int(gen, 2) == 1:
        zfill, image = zfill[:, ::-1], image[:, ::-1]
    if cfg.roll > 0 and image.shape[0] > cfg.phases:
        shift = _draw_int(gen, image.shape[0] - cfg.phases)
        zfill, image = np.roll(zfill, -shift, axis=0), np.roll(image, -shift, axis=0)
    zfill, image = zfill[:cfg.phases], image[:cfg.phases]
    return (np.ascontiguousarray(np.nan_to_num(zfill, nan=0.0)),
            np.ascontiguousarray(np.nan_to_num(image, nan=0.0)))
