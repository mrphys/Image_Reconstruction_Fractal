"""TF implementation of the recon stage.

The 2D path is a line-by-line port of the app version of
``preprocessing_multicoil_noselect.preprocessing_fn`` / ``rolling_fn``.
"""

import tensorflow as tf
import tensorflow_mri as tfmr

from .sampling import SamplingOperator, tile_to_phases


def make_fs_rtcine_image(kspace, image_shape, gen=None, roll=0, phases=20, keep_external_signal=False):
    """Fully sampled multicoil image series ``(T, C, H, W)`` from Cartesian k-space ``(T, C, H, W)``."""
    kspace = tf.ensure_shape(kspace, (None,) * 4)
    image = tfmr.signal.ifft(kspace, axes=[-2, -1], norm='ortho', shift=True)
    if not keep_external_signal:
        image = tfmr.resize_with_crop_or_pad(image, image_shape)
    input_phases = tf.shape(image)[-4]
    if roll > 0:
        image = tf.roll(image, shift=gen.uniform((), minval=0, maxval=input_phases, dtype=tf.int32), axis=-4)
    _cond = lambda x: tf.math.less(tf.shape(x)[-4], phases)
    _body = lambda x: [tf.concat([x, x], axis=-4)]
    image = tf.while_loop(_cond, _body, [image], shape_invariants=[tf.TensorShape([None, None, None, None])])[0]
    image = image[:phases, ...]
    return tfmr.resize_with_crop_or_pad(image, image_shape)


def gaussian_blur(img, kernel_size=11, sigma=0.5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        return tf.tile(kernel[..., tf.newaxis], [1, 1, channels])

    k = gauss_kernel(tf.shape(tf.expand_dims(img, axis=-1))[-1], kernel_size, sigma)[..., tf.newaxis]
    out = tf.nn.depthwise_conv2d(tf.expand_dims(img, axis=-1), k, [1, 1, 1, 1], padding='SAME', data_format='NHWC')
    return out[..., 0]


def format_pair(zfill, image, cfg):
    """Legacy output formats: zfill ``(..., 1)`` complex -> abs or real/imag channels; target abs or 2 channels."""
    if cfg.input_format == 'abs':
        zfill = tf.math.abs(zfill)
    else:
        zfill = tf.concat((tf.math.real(zfill), tf.math.imag(zfill)), axis=-1)
    if cfg.output_format in ('abs', 'abspre'):
        image = tf.math.abs(image)
    else:
        image = tf.concat((tf.expand_dims(tf.math.real(image), axis=-1),
                           tf.expand_dims(tf.math.imag(image), axis=-1)), axis=-1)
    return zfill, image


def reconstruct_2d(kspace, traj, dcw, cfg):
    """2D+t: canonical ``(C, T, 1, H, W)`` -> legacy-layout pair (zfill ``(H, W, T, ch)``, image ``(H, W, T)``)."""
    if cfg.roll > 0:
        raise NotImplementedError("roll>0 in recon is unused by the fractal pipelines")
    kspace = tf.transpose(kspace[:, :, 0], [1, 0, 2, 3])                  # (T, C, H, W)
    image_shape = [cfg.base_resolution, cfg.base_resolution]
    traj, dcw = tile_to_phases(traj, dcw, cfg.phases)
    op = SamplingOperator(traj, dcw, image_shape)

    image = make_fs_rtcine_image(kspace, image_shape, phases=cfg.phases)
    uskspace = op.apply_dcw(op.forward(image))                            # (T, C, R)
    if cfg.input_format == 'coil_compressed':
        cc = tfmr.coils.CoilCompressorSVD(coil_axis=-3, out_coils=cfg.input_coils)
        cc.fit(tf.expand_dims(uskspace, axis=-1))
        kspace = cc.transform(kspace)
        uskspace = cc.transform(tf.expand_dims(uskspace, axis=-1))[..., 0]
    image = make_fs_rtcine_image(kspace, image_shape, phases=cfg.phases)

    scalingfactor = tf.cast(tf.reduce_max(tf.abs(tfmr.coils.combine_coils(image, coil_axis=-3))), image.dtype)
    image = image / scalingfactor
    uskspace = uskspace / scalingfactor
    zfill = op.adjoint(uskspace)                                          # (T, C, H, W)
    image = tfmr.coils.combine_coils(image, coil_axis=-3)                 # (T, H, W) SoS
    if cfg.normalize_input:
        zfill = tfmr.scale_by_min_max(zfill)
    zfill = tf.transpose(zfill, [2, 3, 0, 1])                             # (H, W, T, C)
    if cfg.coil_maps != "walsh_xyt_legacy":
        raise ValueError(f"2D path supports coil_maps='walsh_xyt_legacy', got {cfg.coil_maps!r}")
    maps = tfmr.coils.estimate_sensitivities(zfill, method='walsh')      # time acts as a 3rd spatial axis
    zfill = tfmr.coils.combine_coils(zfill, maps=maps, coil_axis=-1, keepdims=True)   # (H, W, T, 1)
    if cfg.gfilt is not None:
        image = tf.expand_dims(tf.complex(gaussian_blur(tf.math.real(image[0, ...]), sigma=cfg.gfilt),
                                          gaussian_blur(tf.math.imag(image[0, ...]), sigma=cfg.gfilt)), axis=0)
    image = tf.transpose(image, [1, 2, 0])                                # (H, W, T)
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
    if tf.rank(zfill) == 3:
        zfill = tf.expand_dims(zfill, axis=-1)
    zfill = tf.transpose(zfill, perm=[2, 0, 1, 3])
    if tf.rank(image) == 3:
        image = tf.expand_dims(image, axis=-1)
    image = tf.transpose(image, perm=[2, 0, 1, 3])
    if cfg.rotation > 0:
        rot_im = gen.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=rot_im)
        zfill = tf.image.rot90(zfill, k=rot_im)
    if cfg.roll > 0:
        shift_im = gen.uniform(shape=(), minval=0, maxval=tf.shape(image)[0] - cfg.phases, dtype=tf.int32)
        image = tf.roll(image, shift=-shift_im, axis=0)
        zfill = tf.roll(zfill, shift=-shift_im, axis=0)
    image = image[:cfg.phases, ...]
    zfill = zfill[:cfg.phases, ...]
    image = tf.where(tf.math.is_nan(image), 0., image)
    zfill = tf.where(tf.math.is_nan(zfill), 0., zfill)
    return zfill, image


def augment(zfill, image, cfg, rng):
    return augment_2d(zfill, image, cfg, rng)
