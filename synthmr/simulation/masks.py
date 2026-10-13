"""Object support masks (backend-free; random draws from an injected stream)."""

import numpy as np


def elliptical_mask_2d(shape, gen):
    """Random filled ellipse ``(H, W)`` uint8, drawing from ``gen`` exactly like the legacy
    ``preprocessing_natural_videos.elliptical_mask`` (including its float32 arithmetic)."""
    import cv2

    shape = np.asarray(shape)
    mask = np.zeros(np.concatenate((shape, [3])), dtype=np.int32)
    center = (int(shape[-2] // 2), int(shape[-1] // 2))
    f32 = lambda v: np.float32(np.asarray(v))
    # Legacy: numpy int * tf.float32 tensor -> float32 product, then int().
    ax0 = int(np.float32(shape[-2] // 2) * f32(gen.uniform(shape=(), minval=1.0, maxval=1.4)))
    ax1 = int(np.float32(shape[-1] // 2 * 0.8) * f32(gen.uniform(shape=(), minval=0.8, maxval=1.2)))
    angle = int(f32(gen.uniform(shape=(), minval=0, maxval=360)))
    mask = cv2.ellipse(mask, center, (ax0, ax1), angle, 0, 360, (1, 1, 1), -1)
    return mask[:, :, 0].astype(np.uint8)


def _u(gen, lo, hi, shape=()):
    return np.asarray(gen.uniform(shape=shape, minval=lo, maxval=hi), dtype=np.float64)


def ellipsoid_mask_3d(shape, gen, slice_spacing=1.0):
    """Random filled ellipsoid ``(Z, H, W)`` float32, shared by every frame and slice.

    In-plane semi-axes and angle follow the 2D ellipse distribution. The z semi-axis is
    1.1-1.6x the half stack height, so every slice holds a (shrinking) cross-section.
    """
    z, h, w = shape
    a = h / 2 * _u(gen, 1.0, 1.4)
    b = w / 2 * 0.8 * _u(gen, 0.8, 1.2)
    angle = np.deg2rad(_u(gen, 0, 360))
    c = max(z * slice_spacing / 2, 1e-6) * _u(gen, 1.1, 1.6)
    zz = (np.arange(z) - (z - 1) / 2) * slice_spacing
    yy = np.arange(h) - h // 2
    xx = np.arange(w) - w // 2
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
    # cv2 convention: angle rotates the first axis (x) towards y
    xr = X * np.cos(angle) + Y * np.sin(angle)
    yr = -X * np.sin(angle) + Y * np.cos(angle)
    return ((xr / a) ** 2 + (yr / b) ** 2 + (Z / c) ** 2 <= 1.0).astype(np.float32)
