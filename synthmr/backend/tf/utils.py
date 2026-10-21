"""TF helpers centralised from ``app/process_data_old.py`` (used by real-MRI workflows)."""

import numpy as np
import tensorflow as tf
import tensorflow_mri as tfmr


def convert_to_real(complex_tensor, axis=0):
    """Concatenate real and imaginary parts along ``axis`` (legacy default: axis 0)."""
    return tf.concat([tf.math.real(complex_tensor), tf.math.imag(complex_tensor)], axis=axis)


def convert_to_complex(stacked_tensor):
    """Inverse of channel-last real/imag stacking."""
    real_part, imag_part = tf.split(stacked_tensor, 2, axis=-1)
    return tf.complex(real_part, imag_part)


def normalise_max_abs(image):
    """Divide by max magnitude (legacy ``normalise``)."""
    return image / tf.cast(tf.math.reduce_max(tf.math.abs(image)), image.dtype)


def sample_noncartesian_kspace(image, trajectory):
    """Type-2 NUFFT scaled by ``1/sqrt(H*W)``."""
    kspace = tfmr.signal.nufft(image, trajectory, transform_type='type_2', fft_direction='forward')
    return kspace / tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), tf.complex64))


def compute_trajectory_and_density(image_shape, extra_shape, views, ordering):
    """Flattened radial trajectory and density for ``extra_shape[-1]`` phases."""
    trajectory = tfmr.sampling.radial_trajectory(base_resolution=image_shape[-1], views=views,
                                                 phases=extra_shape[-1], ordering=ordering)
    density = tfmr.sampling.estimate_radial_density(trajectory)
    return tfmr.sampling.flatten_trajectory(trajectory), tfmr.sampling.flatten_density(density)


class LowPassScaler:
    """Scale k-space so the low-pass, frame-averaged adjoint image has unit max."""

    def __init__(self, cutoff=0.5 * np.pi, image_shape=(256, 256)):
        self.cutoff, self.image_shape, self.scale = cutoff, list(image_shape), None

    def fit(self, kspace, trajectory, density):
        kspace = tfmr.signal.filter_kspace(kspace, trajectory=trajectory)
        image = tfmr.recon.adjoint(kspace, image_shape=self.image_shape, trajectory=trajectory, density=density)
        image = tf.math.reduce_mean(image, axis=-3)
        self.scale = tf.math.reciprocal(tf.math.reduce_max(tf.math.abs(image)))
        return self

    def transform(self, kspace):
        if self.scale is None:
            raise ValueError("LowPassScaler has not been fitted yet; call fit first.")
        return kspace * tf.cast(self.scale, kspace.dtype)

    def fit_transform(self, kspace, trajectory, density):
        return self.fit(kspace, trajectory, density).transform(kspace)
