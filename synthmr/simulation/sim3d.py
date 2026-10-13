"""Volume-consistent 3D+t simulation.

Unlike the legacy multislice approach (each slice simulated separately with the RNG re-seeded
so slices "match"), the whole ``(T, Z, H, W)`` volume is simulated at once: one 3D object
mask, one 3D background phase, one set of 3D coil sensitivities and one SNR level.
Only the video preparation and the FFT use the backend; the rest is numpy.
"""

import numpy as np

from .coils import simulate_coils_3d
from .masks import ellipsoid_mask_3d
from .phase import background_phase_3d


def awgn_np(gen, data, regsnr):
    """Complex white Gaussian noise for target SNR in dB (``regsnr > 1``) or stddev ``regsnr``."""
    noisepower = np.mean(np.abs(data) ** 2) / (10 ** (regsnr / 10)) if regsnr > 1 else regsnr ** 2
    std = float(np.sqrt(noisepower))
    re = np.asarray(gen.normal(shape=data.shape, mean=0.0, stddev=std))
    im = np.asarray(gen.normal(shape=data.shape, mean=0.0, stddev=std))
    return data + (re + 1j * im).astype(data.dtype)


def simulate_3d(video, cfg, rng, backend):
    """``video`` ``(T, Z, H, W, 2)`` -> dict(kspace ``(C, T, Z, H, W)``, smaps, image, mask)."""
    gen, br = rng.transforms, cfg.base_resolution
    z = cfg.slices
    image = backend.to_numpy(backend.simulation.prepare_video(video, cfg, gen, [z, br, br]))   # (T, Z, H, W)
    shape = (z, br, br)
    mask = None
    if cfg.masking:
        mask = ellipsoid_mask_3d(shape, gen, cfg.slice_spacing_px)
        image = image * (mask + 1e-8)[None]
    smaps = simulate_coils_3d(gen, shape, cfg.sigma_coil, cfg.num_coils, add_phase=cfg.add_phase,
                              ngrid=cfg.coil_phase_ngrid, slice_spacing=cfg.slice_spacing_px)
    if cfg.masking:
        smaps = smaps * mask[None]
    if cfg.add_phase > 0:
        image = image * np.exp(1j * background_phase_3d(shape, gen, ngrid=cfg.object_phase_ngrid,
                                                        slice_spacing=cfg.slice_spacing_px))[None]
    image = image.astype(np.complex64)
    coil_images = smaps[:, None] * image[None]                                   # (C, T, Z, H, W)
    coil_images /= np.sqrt((np.abs(coil_images) ** 2).sum(0)).max()
    regsnr = cfg.regsnr
    snr = float(np.asarray(gen.uniform(shape=(), minval=regsnr[0], maxval=regsnr[1]))) if len(regsnr) == 2 \
        else regsnr[0]
    if snr > 0:
        coil_images = awgn_np(rng.noise, coil_images, snr)
    kspace = backend.fft(backend.from_numpy(coil_images.astype(np.complex64)), axes=[-2, -1])
    return {"kspace": kspace, "smaps": smaps, "image": image, "mask": mask}
