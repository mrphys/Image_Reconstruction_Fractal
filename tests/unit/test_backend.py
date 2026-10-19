"""Backend contract tests (docs/BACKEND.md), parametrised over every backend importable here.

References are backend-free (numpy DFTs, synthetic coils, ``tests/data/radial_ref_tfmri.npz`` captured
from tfmri), so the TF-free sigpy environment checks against the same ground truth as the TF one.
"""

from pathlib import Path

import numpy as np
import pytest

from synthmr.backend import set_backend
from tests.backends import available_backends

BACKENDS = available_backends()
RADIAL_REF = Path(__file__).resolve().parents[1] / "data/radial_ref_tfmri.npz"


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


def test_nufft_forward_matches_dft(B):
    """Pins sign, scaling (unscaled DFT on a centred grid) and that ``traj[..., i]`` maps to image axis i."""
    rng = np.random.default_rng(4)
    h, w = 10, 8
    x = _cplx(rng, (2, h, w))
    traj = rng.uniform(-np.pi, np.pi, (2, 40, 2)).astype(np.float32)     # one trajectory per batch item
    iy, ix = np.arange(h) - h // 2, np.arange(w) - w // 2
    phase = np.exp(-1j * (traj[..., 0, None, None] * iy[:, None] + traj[..., 1, None, None] * ix[None, :]))
    ref = (x[:, None] * phase).sum(axis=(-2, -1))
    out = B.to_numpy(B.nufft_forward(B.from_numpy(x), B.from_numpy(traj)))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-3 * np.abs(ref).max())


@pytest.mark.parametrize("ordering", ["linear", "golden", "golden_half", "sorted", "sorted_half", "tiny",
                                      "tiny_half"])
def test_radial_trajectory_and_density_match_tfmri(B, ordering):
    ref = np.load(RADIAL_REF)
    traj = B.radial_trajectory(32, 13, 3, ordering)
    np.testing.assert_allclose(B.to_numpy(traj), ref[f"traj_{ordering}"], atol=1e-5)
    dens = B.estimate_radial_density(B.flatten_trajectory(traj))
    np.testing.assert_allclose(B.to_numpy(dens), ref[f"dens_{ordering}"], rtol=1e-5)


def test_generator_reset_and_independence(B):
    g = B.Generator(7)
    a = B.to_numpy(g.uniform((5,)))
    b = B.to_numpy(g.uniform((5,)))
    g.reset()
    np.testing.assert_array_equal(B.to_numpy(g.uniform((5,))), a)
    assert not np.array_equal(a, b)
    ints = B.to_numpy(g.uniform((), minval=0, maxval=4, dtype="int32"))
    assert ints.dtype == np.int32 and 0 <= ints < 4


def test_numpy_rng_is_backend_independent(B):
    """``RNG(kind="numpy")`` gives every backend the same draws (cross-backend regression relies on it)."""
    from synthmr.rng import RNG
    a, b = RNG(kind="numpy", backend=B), RNG(kind="numpy")
    for _ in range(3):
        np.testing.assert_array_equal(np.asarray(a.transforms.uniform((4,), minval=-1, maxval=3)),
                                      np.asarray(b.transforms.uniform((4,), minval=-1, maxval=3)))
    np.testing.assert_array_equal(np.asarray(a.noise.normal((3, 2), stddev=0.5)), np.asarray(b.noise.normal((3, 2), stddev=0.5)))
    assert np.asarray(a.augment.uniform((), minval=0, maxval=4, dtype="int32")).dtype == np.int32


def test_crop_or_pad_centred(B):
    x = np.arange(36, dtype=np.float32).reshape(6, 6)
    out = B.to_numpy(B.crop_or_pad(B.from_numpy(x), [4, 8]))
    assert out.shape == (4, 8)
    np.testing.assert_array_equal(out[:, 1:7], x[1:5])
    odd = B.to_numpy(B.crop_or_pad(B.from_numpy(np.ones((3, 5, 2), np.float32)), [6, 2, -1]))
    assert odd.shape == (6, 2, 2) and odd[1:4].all() and not odd[0].any() and not odd[4:].any()


def test_scale_by_min_max_keeps_phase(B):
    z = _cplx(np.random.default_rng(5), (6, 7))
    out = B.to_numpy(B.scale_by_min_max(B.from_numpy(z)))
    a = np.abs(z)
    np.testing.assert_allclose(np.abs(out), (a - a.min()) / (a.max() - a.min()), atol=1e-6)
    np.testing.assert_allclose(np.angle(out[np.abs(out) > 1e-3]), np.angle(z[np.abs(out) > 1e-3]), atol=1e-5)


def test_combine_coils_sos(B):
    x = _cplx(np.random.default_rng(2), (8, 8, 4))
    out = B.to_numpy(B.combine_coils(B.from_numpy(x), coil_axis=-1))
    np.testing.assert_allclose(np.abs(out), np.sqrt((np.abs(x) ** 2).sum(-1)), rtol=1e-5)


def test_coil_compressor_svd_keeps_energy_subspace(B):
    """Compressing to the signal rank keeps (nearly) all energy and the coil-combined SoS."""
    rng = np.random.default_rng(6)
    mix = _cplx(rng, (3, 8))                                             # 8 coils spanning a rank-3 space
    k = (_cplx(rng, (5, 200, 3)) @ mix).transpose(0, 2, 1)[..., None]   # (5, 8, 200, 1), coil axis -3
    cc = B.coil_compressor_svd(coil_axis=-3, out_coils=3).fit(B.from_numpy(k))
    out = B.to_numpy(cc.transform(B.from_numpy(k)))
    assert out.shape == (5, 3, 200, 1)
    np.testing.assert_allclose(np.sqrt((np.abs(out) ** 2).sum(1)), np.sqrt((np.abs(k) ** 2).sum(1)), rtol=1e-3)


def test_walsh_supports_3d_spatial(B):
    """Needed by the 3D+t coil-map strategy (Phase 3)."""
    x =_cplx(np.random.default_rng(3), (16, 16, 6, 4))
    maps = B.to_numpy(B.estimate_sensitivities(B.from_numpy(x), coil_axis=-1))
    assert maps.shape == x.shape
    np.testing.assert_allclose(np.sqrt((np.abs(maps) ** 2).sum(-1)), 1.0, atol=1e-3)


def test_walsh_recovers_smooth_coils(B):
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:32, :32] / 32.0
    centres = [(0, 0, 0.3), (1, 0, 1.0), (0, 1, 2.0), (1, 1, -1.0)]
    true = np.stack([np.exp(-((yy - a) ** 2 + (xx - b) ** 2) / 0.2) * np.exp(1j * p) for a, b, p in centres], -1)
    data = (true * (1 + 0.3 * np.cos(6 * xx))[..., None] + 0.01 * _cplx(rng, true.shape)).astype(np.complex64)
    maps = B.to_numpy(B.estimate_sensitivities(B.from_numpy(data), coil_axis=-1))
    corr = np.abs((maps * np.conj(true)).sum(-1)) / np.linalg.norm(true, axis=-1)
    assert corr.mean() > 0.99


def test_rng_streams_follow_legacy_seeds():
    from synthmr.rng import LEGACY_SEEDS, RNG
    rng = RNG()
    assert rng.seeds() == LEGACY_SEEDS
    rng.reset(transforms=11, augment=11)
    assert rng.seeds()["transforms"] == 11 and rng.seeds()["noise"] == 2
