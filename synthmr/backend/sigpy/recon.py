"""sigpy implementation of the recon stage (port of ``backend/tf/recon.py``), plus JSENSE coil maps."""

import math

import numpy as np

from . import core
from .sampling import SamplingOperator, tile_to_phases


def make_fs_rtcine_image(kspace, image_shape, gen=None, roll=0, phases=20, keep_external_signal=False):
    """Fully sampled multicoil image series ``(T, C, H, W)`` from Cartesian k-space ``(T, C, H, W)``."""
    xp = core.xp_of(kspace)
    image = core.ifft(kspace, axes=[-2, -1])
    if not keep_external_signal:
        image = core.crop_or_pad(image, image_shape)
    input_phases = image.shape[-4]
    if roll > 0:
        image = xp.roll(image, shift=int(gen.uniform((), minval=0, maxval=input_phases, dtype="int32")), axis=-4)
    while image.shape[-4] < phases:
        image = xp.concatenate([image, image], axis=-4)
    image = image[:phases]
    return core.crop_or_pad(image, image_shape)


def gaussian_blur(img, kernel_size=11, sigma=0.5):
    """Normalised Gaussian blur over the last two axes (zero 'SAME' padding)."""
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    kernel = core.from_numpy(kernel / kernel.sum()).reshape((1,) * (img.ndim - 2) + kernel.shape)
    return core.ndimage_of(img).convolve(img, kernel, mode="constant")


def format_pair(zfill, image, cfg):
    """Legacy output formats: zfill ``(..., C)`` complex -> abs or ``[re_1..re_C, im_1..im_C]`` channels;
    target abs or 2 channels. C = 1 for a coil-combined zfill."""
    xp = core.xp_of(zfill)
    if cfg.input_format == 'abs':
        zfill = xp.abs(zfill)
    else:
        zfill = xp.concatenate((zfill.real, zfill.imag), axis=-1)
    if cfg.output_format in ('abs', 'abspre'):
        image = xp.abs(image)
    else:
        image = xp.stack((image.real, image.imag), axis=-1)
    return zfill, image


def jsense_maps(uskspace, coord, dcw, img_shape, cfg):
    """JSENSE coil maps (``sigpy.mri.app.JsenseRecon``, as in extreme_mri ``jsense_recon.py``) from all frames.

    ``uskspace`` ``(C, *M)`` is density-compensated k-space, ``coord`` ``(*M, ndim)`` in pixel units,
    ``dcw`` broadcastable to ``M``. Returns ``(C, *img_shape)`` maps with unit root-sum-of-squares.
    """
    from sigpy.mri.app import JsenseRecon

    xp = core.xp_of(uskspace)
    dcw = xp.broadcast_to(dcw, uskspace.shape[1:]).astype(np.float32)
    raw = core.divide_no_nan(uskspace, dcw.astype(np.complex64))          # undo the dcw: JSENSE weights itself
    # Host arrays in, as in extreme_mri: JsenseRecon selects the calibration region with numpy, then moves
    # the data to ``device`` itself.
    raw, coord, dcw = (core.to_numpy(a) for a in (raw, coord.astype(np.float32), dcw))
    mps = JsenseRecon(raw, coord=coord, weights=dcw, img_shape=list(img_shape),
                      mps_ker_width=cfg.jsense_ker_width, ksp_calib_width=cfg.jsense_calib_width,
                      lamda=cfg.jsense_lamda, max_iter=cfg.jsense_max_iter,
                      max_inner_iter=cfg.jsense_max_inner_iter, device=core.current_device(),
                      show_pbar=False).run()
    mps = core.from_numpy(mps).astype(np.complex64)
    return core.divide_no_nan(mps, xp.sqrt(xp.sum(xp.abs(mps) ** 2, axis=0, keepdims=True)).astype(np.complex64))


def reconstruct_2d(kspace, traj, dcw, cfg):
    """2D+t: canonical ``(C, T, 1, H, W)`` -> legacy-layout pair (zfill ``(H, W, T, ch)``, image ``(H, W, T)``)."""
    if cfg.roll > 0:
        raise NotImplementedError("roll>0 in recon is unused by the fractal pipelines")
    if cfg.coil_maps not in ("walsh_xyt_legacy", "jsense"):
        raise ValueError(f"2D path supports coil_maps='walsh_xyt_legacy' or 'jsense', got {cfg.coil_maps!r}")
    xp = core.xp_of(kspace)
    kspace = xp.transpose(kspace[:, :, 0], (1, 0, 2, 3))                 # (T, C, H, W)
    image_shape = [cfg.base_resolution, cfg.base_resolution]
    traj, dcw = tile_to_phases(traj, dcw, cfg.phases)
    op = SamplingOperator(traj, dcw, image_shape)

    image = make_fs_rtcine_image(kspace, image_shape, phases=cfg.phases)
    uskspace = op.apply_dcw(op.forward(image))                            # (T, C, R)
    if cfg.input_format == 'coil_compressed':
        cc = core.coil_compressor_svd(coil_axis=-3, out_coils=cfg.input_coils)
        cc.fit(uskspace[..., None])
        kspace = cc.transform(kspace)
        uskspace = cc.transform(uskspace[..., None])[..., 0]
    image = make_fs_rtcine_image(kspace, image_shape, phases=cfg.phases)

    scalingfactor = xp.abs(core.combine_coils(image, coil_axis=-3)).max().astype(image.dtype)
    image = image / scalingfactor
    uskspace = uskspace / scalingfactor
    zfill = op.adjoint(uskspace)                                          # (T, C, H, W)
    image = core.combine_coils(image, coil_axis=-3)                       # (T, H, W) SoS
    if cfg.normalize_input:
        zfill = core.scale_by_min_max(zfill)
    zfill = xp.transpose(zfill, (2, 3, 0, 1))                             # (H, W, T, C)
    if cfg.coil_maps == "jsense":
        coord = traj * np.float32(cfg.base_resolution / (2 * math.pi))    # (T, R, 2) pixels
        maps = jsense_maps(xp.transpose(uskspace, (1, 0, 2)), coord, dcw, image_shape, cfg)   # (C, H, W)
        maps = xp.transpose(maps, (1, 2, 0))[:, :, None]                  # (H, W, 1, C)
    else:
        maps = core.estimate_sensitivities(zfill, coil_axis=-1)          # time acts as a 3rd spatial axis
    if cfg.input_combine == "maps":
        zfill = core.combine_coils(zfill, maps=maps, coil_axis=-1, keepdims=True)   # (H, W, T, 1)
    elif cfg.input_combine != "none":
        raise ValueError(f"unknown input_combine {cfg.input_combine!r}; use 'maps' or 'none'")
    if cfg.gfilt is not None:
        # TF applies this to frame 0 only and fails on its 3D conv input; here every frame is blurred.
        image = (gaussian_blur(image.real, sigma=cfg.gfilt) + 1j * gaussian_blur(image.imag, sigma=cfg.gfilt))
    image = xp.transpose(image, (1, 2, 0))                                # (H, W, T)
    zfill, image = format_pair(zfill, image, cfg)
    return {'zfill': zfill, 'image': image, 'uskspace': uskspace, 'traj': traj, 'dcw': dcw, 'maps': maps}


def reconstruct(kspace, traj, dcw, cfg, through_plane="none", kz_rotation="aligned"):
    if through_plane == "none":
        if kspace.shape[2] != 1:
            raise ValueError("through_plane='none' needs Z=1; use 'cartesian_kz' for 3D+t")
        return reconstruct_2d(kspace, traj, dcw, cfg)
    if through_plane == "cartesian_kz":
        from .recon3d import reconstruct_3d
        return reconstruct_3d(kspace, traj, dcw, cfg, kz_rotation)
    raise ValueError(f"unknown through_plane {through_plane!r}")


def augment_2d(zfill, image, cfg, rng):
    """Legacy ``rolling_fn``: zfill ``(H, W, T[, ch])``, image ``(H, W, T[, ch])`` -> time-first, rotated, rolled."""
    gen = rng.augment
    xp = core.xp_of(zfill)
    if zfill.ndim == 3:
        zfill = zfill[..., None]
    zfill = xp.transpose(zfill, (2, 0, 1, 3))
    if image.ndim == 3:
        image = image[..., None]
    image = xp.transpose(image, (2, 0, 1, 3))
    if cfg.rotation > 0:
        rot_im = int(gen.uniform(shape=(), minval=0, maxval=4, dtype="int32"))
        image = xp.rot90(image, k=rot_im, axes=(1, 2))
        zfill = xp.rot90(zfill, k=rot_im, axes=(1, 2))
    if cfg.roll > 0:
        shift_im = int(gen.uniform(shape=(), minval=0, maxval=image.shape[0] - cfg.phases, dtype="int32"))
        image = xp.roll(image, shift=-shift_im, axis=0)
        zfill = xp.roll(zfill, shift=-shift_im, axis=0)
    image = image[:cfg.phases]
    zfill = zfill[:cfg.phases]
    image = xp.ascontiguousarray(xp.where(xp.isnan(image), 0, image))
    zfill = xp.ascontiguousarray(xp.where(xp.isnan(zfill), 0, zfill))
    return zfill, image


def augment(zfill, image, cfg, rng):
    return augment_2d(zfill, image, cfg, rng)
