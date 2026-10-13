"""TF implementation of the simulation stage.

The 2d path is a line-by-line port of ``preprocessing_natural_videos.preprocessing_fn``
(app version) with the module-level generators replaced by injected streams; the
order of random draws is unchanged so outputs are bit-identical.
"""

import math

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_mri as tfmri

from synthmr.simulation.masks import elliptical_mask_2d


# --- helpers (legacy) -------------------------------------------------------
def gauss_kernel(channels, kernel_size, sigma):
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-xx ** 2 / (2.0 * sigma[0] ** 2) - yy ** 2 / (2.0 * sigma[1] ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    return tf.tile(kernel[..., tf.newaxis], [1, 1, channels])


def background_phase_2d(image, gen, ngrid=6, scaling=1):
    """Random smooth phase (bicubic-upsampled ``ngrid x ngrid``), shared along axis 0."""
    siz = (ngrid,) * 2
    arr = gen.uniform(siz) * 2 * scaling
    arr = tf.expand_dims(tf.expand_dims(arr, axis=0), axis=-1)
    arr = tf.image.resize(arr, image.shape[1:], 'bicubic')
    arr = arr[..., 0]
    randphase = (arr + gen.uniform((1,)) * 2 * math.pi)
    return image * tf.exp(1j * tf.cast(randphase, tf.complex64))


def _range_or_fixed_int(gen, values):
    if len(values) == 2:
        return gen.uniform(shape=(), minval=values[0], maxval=values[1], dtype=tf.int32), values[1]
    return values[0], values[0]


def simulate_coils_2d(gen, base_resolution, sigma_coil, num_coils, coil_size=None, add_phase=0, ngrid=6):
    """Gaussian-magnitude coils with random position/intensity/phase -> ``(C, H, W)``, max SoS = 1."""
    if coil_size is None:
        coil_size = base_resolution
    smaps = []
    total_coils, nsim = _range_or_fixed_int(gen, num_coils)
    for _ in range(nsim):
        if len(sigma_coil) == 2:
            sigma_x = gen.uniform(shape=(1,), minval=sigma_coil[0], maxval=sigma_coil[1])
            sigma_y = gen.uniform(shape=(1,), minval=sigma_coil[0], maxval=sigma_coil[1])
            sigmas = [base_resolution / sigma_x, base_resolution / sigma_y]
        else:
            sigmas = [base_resolution / sigma_coil[0], base_resolution / sigma_coil[0]]
        base = gauss_kernel(2, coil_size, sigma=sigmas)
        random_phase = tf.cast(gen.uniform(shape=(1,), minval=-math.pi, maxval=math.pi), dtype=tf.complex64)
        random_intensity = tf.cast(gen.uniform(shape=(1,), minval=0.1, maxval=1), dtype=tf.complex64)
        # coils in random positions (excluding centre)
        transform = tf.cast(gen.uniform(shape=(2,), minval=base_resolution // 5, maxval=base_resolution // 2), dtype=tf.float32)
        random_locations = tf.cast(gen.uniform(shape=(2,), minval=-1, maxval=9, dtype=tf.int32), dtype=tf.float32)
        transform = transform * [(-1) ** random_locations[0], (-1) ** random_locations[1]]
        temp = tfa.image.translate(base, transform, 'bilinear')
        temp = random_intensity * tf.complex(temp[:, :, 0], temp[:, :, 1]) * tf.exp(1j * random_phase)
        temp = tfmri.resize_with_crop_or_pad(temp, [base_resolution, base_resolution])
        if add_phase > 0:
            temp = background_phase_2d(tf.expand_dims(temp, axis=0), gen, ngrid=ngrid, scaling=add_phase)[0, ...]
        smaps.append(temp)
    ridxs = tf.argsort(gen.uniform([nsim]))[:total_coils]
    smaps = tf.gather(smaps, ridxs, axis=0)
    smaps = tf.stack(smaps, axis=0)
    sos_smaps = tf.abs(tf.sqrt(tf.reduce_sum(smaps * tf.math.conj(smaps), axis=0)))
    return smaps / tf.cast(tf.reduce_max(sos_smaps), tf.complex64)


def awgn(gen, data, regsnr, cpx=False):
    """White Gaussian noise for target SNR (dB) if ``regsnr > 1`` else stddev ``regsnr``."""
    if regsnr > 1:
        sigpower = tf.reduce_mean(tf.abs(data) ** 2)
        noisepower = sigpower / (10 ** (regsnr / 10))
    else:
        noisepower = regsnr ** 2
    std = tf.math.sqrt(noisepower)
    if cpx:
        noise = tf.complex(gen.normal(shape=tf.shape(data), mean=0.0, stddev=std),
                           gen.normal(shape=tf.shape(data), mean=0.0, stddev=std))
    else:
        noise = gen.normal(shape=tf.shape(data), mean=0.0, stddev=std)
    return data + noise


def prepare_video(video, cfg, gen, spatial):
    """Crop, contrast-adjust and convert a 2-channel video to complex ``(T, *spatial)``."""
    image_series = tf.cast(video, tf.float32)
    image_series = tfmri.resize_with_crop_or_pad(image_series, shape=[cfg.phases, *spatial, -1])
    image_series = tfmri.scale_by_min_max(image_series)
    adj = tf.cast(cfg.image_adjustment, image_series.dtype)
    image_series = tf.clip_by_value(adj[0] + adj[1] * image_series, 0, 1)
    if cfg.complex_transform > 0:
        shift_cha = gen.uniform(shape=(), minval=0, maxval=tf.shape(image_series)[-1], dtype=tf.int32)
        image_series = tf.roll(image_series, shift=shift_cha, axis=-1)
        image_series = tfmri.resize_with_crop_or_pad(image_series, shape=[cfg.phases, *spatial, 2])
        image_series = tf.complex(image_series[..., 0], image_series[..., 1])
        image_series = (tf.cast(tf.math.abs(image_series), tf.complex64)
                        * tf.math.exp(1j * tf.cast(tf.math.angle(image_series) * cfg.complex_transform, tf.complex64)))
    else:
        image_series = tf.cast(tf.math.sqrt(tf.reduce_sum(image_series ** 2, axis=-1)), tf.complex64)
    return image_series


def _noise_and_normalise(coil_images, cfg, rng):
    sos_image = tf.abs(tf.sqrt(tf.reduce_sum(coil_images * tf.math.conj(coil_images), axis=0)))
    coil_images = coil_images / tf.cast(tf.reduce_max(sos_image), tf.complex64)
    regsnr = cfg.regsnr
    snr = rng.transforms.uniform(shape=(), minval=regsnr[0], maxval=regsnr[1], dtype=tf.float32) \
        if len(regsnr) == 2 else regsnr[0]
    if snr > 0:
        coil_images = awgn(rng.noise, coil_images, snr, cpx=True)
    return coil_images


# --- stage ------------------------------------------------------------------
def simulate_2d(video, cfg, rng):
    """Legacy per-frame 2D simulation. Returns canonical layouts with ``Z = 1``."""
    gen, br = rng.transforms, cfg.base_resolution
    image_series = prepare_video(video, cfg, gen, [br, br])            # (T, H, W)
    mask = None
    if cfg.masking:
        mask = tf.convert_to_tensor(elliptical_mask_2d([br, br], gen))
        image_series = image_series * tf.expand_dims(tf.cast(mask, image_series.dtype) + 10 ** -8, axis=0)
    smaps = simulate_coils_2d(gen, br, cfg.sigma_coil, cfg.num_coils, coil_size=br * cfg.coil_size_factor,
                              add_phase=cfg.add_phase, ngrid=cfg.coil_phase_ngrid)
    if cfg.masking:
        smaps = smaps * tf.expand_dims(tf.cast(mask, image_series.dtype), axis=0)
    if cfg.add_phase > 0:
        image_series = background_phase_2d(image_series, gen, ngrid=cfg.object_phase_ngrid)
    coil_images = tf.expand_dims(smaps, axis=1) * tf.expand_dims(image_series, axis=0)   # (C, T, H, W)
    coil_images = _noise_and_normalise(coil_images, cfg, rng)
    kspace = tf.cast(tfmri.signal.fft(coil_images, axes=[-2, -1], norm='ortho', shift=True), tf.complex64)
    return {
        'kspace': kspace[:, :, None],                                   # (C, T, 1, H, W)
        'smaps': smaps[:, None],                                        # (C, 1, H, W)
        'image': image_series[:, None],                                 # (T, 1, H, W)
        'mask': None if mask is None else tf.cast(mask, tf.float32)[None],
    }


def simulate(video, cfg, rng):
    if cfg.mode == "2d":
        return simulate_2d(video, cfg, rng)
    raise ValueError(f"unknown simulation mode {cfg.mode!r}")


def kspace_to_legacy(kspace):
    """Canonical ``(C, T, 1, H, W)`` -> legacy ``(1, T, C, H, W)``."""
    return tf.transpose(kspace[:, :, 0], [1, 0, 2, 3])[None]
