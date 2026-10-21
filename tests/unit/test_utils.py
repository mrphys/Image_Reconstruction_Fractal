import numpy as np

from synthmr.phantoms import complexify
from synthmr.utils import array, filters


def test_crop_or_circular_pad_matches_cyclic_repeat():
    img = np.arange(5)[:, None]
    np.testing.assert_array_equal(array.crop_or_circular_pad(img, 12)[:, 0], np.tile(np.arange(5), 3)[:12])
    np.testing.assert_array_equal(array.crop_or_circular_pad(img, 3)[:, 0], [0, 1, 2])


def test_circular_pad_and_scale_frames():
    img = np.arange(10)[:, None]
    assert array.circular_pad(img).shape[0] == 16
    assert array.scale_frames(img).shape[0] == 20


def test_uint8_roundtrip():
    x = np.linspace(-1, 1, 11)
    np.testing.assert_allclose(array.int255_to_float(array.float_to_uint8(x, 1, -1), 1, -1), x, atol=1 / 127)


def test_complex_channels_roundtrip():
    z = np.random.default_rng(0).standard_normal((3, 4)) + 1j
    np.testing.assert_array_equal(array.complex_from_channels(array.channels_from_complex(z)), z)


def test_filters_shapes():
    v = np.random.default_rng(1).random((4, 16, 16))
    for f in (filters.median_blur_3d_slicewise, filters.unsharp_mask_3d_slicewise,
              filters.central_crop_and_zoom, filters.sharpen_3d):
        assert f(v).shape == v.shape


def test_complexify_variants_shapes():
    esc = np.random.default_rng(2).integers(0, 50, (4, 16, 16))
    for name, fn in complexify.METHODS.items():
        out = fn(esc, rng=np.random.default_rng(0))
        assert out.shape == (4, 16, 16, 2), name
