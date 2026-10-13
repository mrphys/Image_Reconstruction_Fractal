"""Backend contract tests (docs/BACKEND.md). Parametrised so a torch backend can reuse them."""

import numpy as np
import pytest

from synthmr.backend import set_backend

BACKENDS = ["tf"]


@pytest.fixture(params=BACKENDS)
def B(request):
    return set_backend(request.param)


def _cplx(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def test_fft_matches_centred_numpy(B):
    x = _cplx(np.random.default_rng(0), (3, 16, 12))
    ref = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"), axes=(-2, -1))
    out = B.to_numpy(B.fft(B.from_numpy(x), axes=[-2, -1]))
    np.testing.assert_allclose(out, ref, atol=1e-5)
    back = B.to_numpy(B.ifft(B.from_numpy(out), axes=[-2, -1]))
    np.testing.assert_allclose(back, x, atol=1e-5)


def test_nufft_adjoint_inner_product(B):
    rng = np.random.default_rng(1)
    x = _cplx(rng, (32, 32))
    traj = rng.uniform(-np.pi, np.pi, (500, 2)).astype(np.float32)
    y = _cplx(rng, (500,))
    Ax = B.to_numpy(B.nufft_forward(B.from_numpy(x), B.from_numpy(traj)))
    AHy = B.to_numpy(B.nufft_adjoint(B.from_numpy(y), B.from_numpy(traj), grid_shape=[32, 32]))
    lhs, rhs = np.vdot(y, Ax), np.vdot(AHy, x)
    assert abs(lhs - rhs) / abs(lhs) < 1e-3


def test_generator_reset_and_independence(B):
    g = B.Generator(7)
    a = B.to_numpy(g.uniform((5,)))
    b = B.to_numpy(g.uniform((5,)))
    g.reset()
    np.testing.assert_array_equal(B.to_numpy(g.uniform((5,))), a)
    assert not np.array_equal(a, b)
    ints = B.to_numpy(g.uniform((), minval=0, maxval=4, dtype="int32"))
    assert ints.dtype == np.int32 and 0 <= ints < 4


def test_crop_or_pad_centred(B):
    x = np.arange(36, dtype=np.float32).reshape(6, 6)
    out = B.to_numpy(B.crop_or_pad(B.from_numpy(x), [4, 8]))
    assert out.shape == (4, 8)
    np.testing.assert_array_equal(out[:, 1:7], x[1:5])


def test_combine_coils_sos(B):
    x = _cplx(np.random.default_rng(2), (8, 8, 4))
    out = B.to_numpy(B.combine_coils(B.from_numpy(x), coil_axis=-1))
    np.testing.assert_allclose(np.abs(out), np.sqrt((np.abs(x) ** 2).sum(-1)), rtol=1e-5)


def test_walsh_supports_3d_spatial(B):
    """Needed by the 3D+t coil-map strategy (Phase 3)."""
    x = _cplx(np.random.default_rng(3), (16, 16, 6, 4))
    maps = B.to_numpy(B.estimate_sensitivities(B.from_numpy(x), coil_axis=-1))
    assert maps.shape == x.shape
    np.testing.assert_allclose(np.sqrt((np.abs(maps) ** 2).sum(-1)), 1.0, atol=1e-3)


def test_rng_streams_follow_legacy_seeds():
    from synthmr.rng import LEGACY_SEEDS, RNG
    rng = RNG()
    assert rng.seeds() == LEGACY_SEEDS
    rng.reset(transforms=11, augment=11)
    assert rng.seeds()["transforms"] == 11 and rng.seeds()["noise"] == 2
