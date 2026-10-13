"""Backend-free image filters (scipy / skimage)."""

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter


def gaussian_blur(image, sigma=1):
    return gaussian_filter(image, sigma=sigma)


def sharpen_3d(img):
    """3-D Laplacian sharpening kernel (legacy ``sharpen_image``)."""
    k = np.array([[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                  [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
                  [[0, 0, 0], [0, -1, 0], [0, 0, 0]]])
    return convolve(img, k, mode='reflect')


def central_crop_and_zoom(image, crop_fraction=0.7, order=1):
    """Centrally crop each axis by ``crop_fraction`` and resize back to the original shape."""
    from skimage.transform import resize
    original_shape = np.array(image.shape)
    crop_shape = (original_shape * crop_fraction).astype(int)
    start = ((original_shape - crop_shape) // 2).astype(int)
    cropped = image[tuple(slice(s, s + c) for s, c in zip(start, crop_shape))]
    return resize(cropped, original_shape, order=order, mode='reflect', anti_aliasing=True)


def median_blur_3d_slicewise(volume, kernel_size=3):
    """Median filter each slice along axis 0 of a ``(D, H, W)`` volume."""
    assert volume.ndim == 3 and kernel_size % 2 == 1
    out = np.empty_like(volume)
    for i in range(volume.shape[0]):
        out[i] = median_filter(volume[i], size=kernel_size)
    return out


def unsharp_mask_3d_slicewise(volume, alpha=1.0, blur_sigma=1.0):
    """Slice-wise unsharp mask along axis 0 of a ``(D, H, W)`` volume."""
    assert volume.ndim == 3
    out = np.empty_like(volume)
    for i in range(volume.shape[0]):
        out[i] = volume[i] + alpha * (volume[i] - gaussian_filter(volume[i], sigma=blur_sigma))
    return out
