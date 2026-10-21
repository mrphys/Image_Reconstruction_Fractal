"""sigpy implementation of the simulation stage.

Port of ``backend/tf/simulation.py``: same math, same random draws in the same order (with
``RNG(kind="numpy")`` both backends produce the same simulation up to float rounding).
"""

import math

import numpy as np

from synthmr.simulation.masks import elliptical_mask_2d

from . import core


# --- helpers ----------------------------------------------------------------
def gauss_kernel(channels, kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    s0, s1 = (np.float32(np.asarray(s).reshape(-1)[0]) for s in sigma)
    kernel = np.exp(-xx ** 2 / (np.float32(2.0) * s0 ** 2) - yy ** 2 / (np.float32(2.0) * s1 ** 2))
    kernel = kernel / kernel.sum()
    return core.from_numpy(np.tile(kernel[..., None], [1, 1, channels]).astype(np.float32))


def background_phase_2d(image, gen, ngrid=6, scaling=1):
    """Random smooth phase (bicubic-upsampled ``ngrid x ngrid``), shared along axis 0."""
    xp = core.xp_of(image)
    arr = core.from_numpy(gen.uniform((ngrid, ngrid)) * np.float32(2 * scaling))
    arr = core.resize(arr[None, ..., None], image.shape[1:], "bicubic")[..., 0]
    randphase = arr + core.from_numpy(gen.uniform((1,)) * np.float32(2 * math.pi))
    return image * xp.exp(1j * randphase.astype(np.complex64))


def _range_or_fixed_int(gen, values):
    if len(values) == 2:
        return int(gen.uniform(shape=(), minval=values[0], maxval=values[1], dtype="int32")), values[1]
    return values[0], values[0]


def simulate_coils_2d(gen, base_resolution, sigma_coil, num_coils, coil_size=None, add_phase=0, ngrid=6):
    """Gaussian-magnitude coils with random position/intensity/phase -> ``(C, H, W)``, max SoS = 1."""
    if coil_size is None:
        coil_size = base_resolution
    xp = core.current_xp()
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
        random_phase = np.complex64(gen.uniform(shape=(1,), minval=-math.pi, maxval=math.pi)[0])
        random_intensity = np.complex64(gen.uniform(shape=(1,), minval=0.1, maxval=1)[0])
        # coils in random positions (excluding centre)
        transform = gen.uniform(shape=(2,), minval=base_resolution // 5, maxval=base_resolution // 2)
        random_locations = gen.uniform(shape=(2,), minval=-1, maxval=9, dtype="int32").astype(np.float32)
        transform = transform * np.float32([(-1) ** random_locations[0], (-1) ** random_locations[1]])
        temp = core.translate(base, transform, "bilinear")
        temp = random_intensity * (temp[:, :, 0] + 1j * temp[:, :, 1]).astype(np.complex64) \
            * np.exp(1j * random_phase)
        temp = core.crop_or_pad(temp.astype(np.complex64), [base_resolution, base_resolution])
        if add_phase > 0:
            temp = background_phase_2d(temp[None], gen, ngrid=ngrid, scaling=add_phase)[0]
        smaps.append(temp)
    ridxs = np.argsort(gen.uniform([nsim]), kind="stable")[:total_coils]
    smaps = xp.stack([smaps[i] for i in ridxs], axis=0)
    sos_smaps = xp.abs(xp.sqrt(xp.sum(smaps * xp.conj(smaps), axis=0)))
    return (smaps / sos_smaps.max()).astype(np.complex64)


def awgn(gen, data, regsnr, cpx=False):
    """White Gaussian noise for target SNR (dB) if ``regsnr > 1`` else stddev ``regsnr``."""
    xp = core.xp_of(data)
    if regsnr > 1:
        sigpower = xp.mean(xp.abs(data) ** 2)
        noisepower = sigpower / (10 ** (regsnr / 10))
    else:
        noisepower = regsnr ** 2
    std = float(np.sqrt(core.to_numpy(xp.asarray(noisepower, dtype=np.float32))))
    if cpx:
        re = gen.normal(shape=data.shape, mean=0.0, stddev=std)
        im = gen.normal(shape=data.shape, mean=0.0, stddev=std)
        noise = core.from_numpy((re + 1j * im).astype(np.complex64))
    else:
        noise = core.from_numpy(gen.normal(shape=data.shape, mean=0.0, stddev=std))
    return data + noise


def prepare_video(video, cfg, gen, spatial):
    """Crop, contrast-adjust and convert a 2-channel video to complex ``(T, *spatial)``."""
    xp = core.current_xp()
    image_series = core.from_numpy(np.asarray(core.to_numpy(video), dtype=np.float32))
    image_series = core.crop_or_pad(image_series, [cfg.phases, *spatial, -1])
    image_series = core.scale_by_min_max(image_series)
    adj = np.asarray(cfg.image_adjustment, dtype=np.float32)
    image_series = xp.clip(adj[0] + adj[1] * image_series, 0, 1)
    if cfg.complex_transform > 0:
        shift_cha = int(gen.uniform(shape=(), minval=0, maxval=image_series.shape[-1], dtype="int32"))
        image_series = xp.roll(image_series, shift=shift_cha, axis=-1)
        image_series = core.crop_or_pad(image_series, [cfg.phases, *spatial, 2])
        image_series = (image_series[..., 0] + 1j * image_series[..., 1]).astype(np.complex64)
        image_series = (xp.abs(image_series).astype(np.complex64)
                        * xp.exp(1j * (xp.angle(image_series) * np.float32(cfg.complex_transform)).astype(np.complex64)))
    else:
        image_series = xp.sqrt(xp.sum(image_series ** 2, axis=-1)).astype(np.complex64)
    return image_series.astype(np.complex64)


def _noise_and_normalise(coil_images, cfg, rng):
    xp = core.xp_of(coil_images)
    sos_image = xp.abs(xp.sqrt(xp.sum(coil_images * xp.conj(coil_images), axis=0)))
    coil_images = coil_images / sos_image.max().astype(np.complex64)
    regsnr = cfg.regsnr
    snr = float(rng.transforms.uniform(shape=(), minval=regsnr[0], maxval=regsnr[1], dtype="float32")) \
        if len(regsnr) == 2 else regsnr[0]
    if snr > 0:
        coil_images = awgn(rng.noise, coil_images, snr, cpx=True)
    return coil_images.astype(np.complex64)


# --- stage ------------------------------------------------------------------
def simulate_2d(video, cfg, rng):
    """Legacy per-frame 2D simulation. Returns canonical layouts with ``Z = 1``."""
    gen, br = rng.transforms, cfg.base_resolution
    image_series = prepare_video(video, cfg, gen, [br, br])            # (T, H, W)
    mask = None
    if cfg.masking:
        mask = core.from_numpy(elliptical_mask_2d([br, br], gen))
        image_series = image_series * (mask.astype(np.complex64) + np.float32(10 ** -8))[None]
    smaps = simulate_coils_2d(gen, br, cfg.sigma_coil, cfg.num_coils, coil_size=br * cfg.coil_size_factor,
                              add_phase=cfg.add_phase, ngrid=cfg.coil_phase_ngrid)
    if cfg.masking:
        smaps = smaps * mask.astype(np.complex64)[None]
    if cfg.add_phase > 0:
        image_series = background_phase_2d(image_series, gen, ngrid=cfg.object_phase_ngrid)
    coil_images = smaps[:, None] * image_series[None]                   # (C, T, H, W)
    coil_images = _noise_and_normalise(coil_images, cfg, rng)
    kspace = core.fft(coil_images, axes=[-2, -1]).astype(np.complex64)
    return {
        'kspace': kspace[:, :, None],                                   # (C, T, 1, H, W)
        'smaps': smaps[:, None],                                        # (C, 1, H, W)
        'image': image_series[:, None],                                 # (T, 1, H, W)
        'mask': None if mask is None else mask.astype(np.float32)[None],
    }


def simulate(video, cfg, rng):
    if cfg.mode == "2d":
        return simulate_2d(video, cfg, rng)
    raise ValueError(f"unknown simulation mode {cfg.mode!r}")


def kspace_to_legacy(kspace):
    """Canonical ``(C, T, 1, H, W)`` -> legacy ``(1, T, C, H, W)``."""
    return core.xp_of(kspace).transpose(kspace[:, :, 0], (1, 0, 2, 3))[None]
