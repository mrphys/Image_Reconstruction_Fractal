"""Phase 3 property tests: volume-consistent simulation, stack-of-stars operator, 3D coil maps."""

import numpy as np
import pytest
import tensorflow as tf

from synthmr.backend.tf.recon3d import reconstruct_3d
from synthmr.backend.tf.sampling import SamplingOperator
from synthmr.recon import AugmentConfig, ReconConfig
from synthmr.recon.augment3d import augment_3d
from synthmr.recon.coil_maps import tukey_window_nd
from synthmr.rng import RNG
from synthmr.sampling import TrajectoryConfig, make_trajectory
from synthmr.sampling.encoders import kz_trajectories
from synthmr.simulation import SimulationConfig, simulate
from synthmr.simulation.coils import simulate_coils_3d
from synthmr.simulation.masks import ellipsoid_mask_3d
from synthmr.simulation.phase import background_phase_3d


def _corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_ellipsoid_every_slice_has_object_and_tapers():
    m = ellipsoid_mask_3d((10, 64, 64), RNG().transforms, slice_spacing=10)
    area = m.sum(axis=(1, 2))
    assert m.shape == (10, 64, 64) and (area > 0).all()
    assert area[5] >= area[0] and area[4] >= area[9]


def test_background_phase_smooth_across_slices():
    # realistic geometry: 10 slices x 5 px spanning 50 px of a 128 px field of view
    ph = background_phase_3d((10, 128, 128), RNG().transforms, ngrid=6, scaling=1, slice_spacing=5)
    adjacent = np.abs(np.diff(ph, axis=0)).mean()
    inplane = np.abs(np.diff(ph, axis=1)).mean() * 5      # per 5 px in-plane, for comparison
    assert ph.shape == (10, 128, 128) and 0 < adjacent < 0.5 and adjacent < 2 * inplane


def test_coils_3d_normalised_and_smooth_through_z():
    maps = simulate_coils_3d(RNG().transforms, (10, 128, 128), (2, 6), (8,), add_phase=2, ngrid=2,
                             slice_spacing=5)
    assert maps.shape == (8, 10, 128, 128) and maps.dtype == np.complex64
    np.testing.assert_allclose(np.sqrt((np.abs(maps) ** 2).sum(0)).max(), 1.0, rtol=1e-5)
    corr = np.mean([_corr(maps[c, z], maps[c, z + 1]) for c in range(8) for z in range(9)])
    assert corr > 0.95, corr


def test_tukey_window_support():
    w = tukey_window_nd((4, 32, 32), (1.0, 0.25, 0.25), alpha=0.5)
    assert w.shape == (4, 32, 32) and np.isclose(w.max(), 1.0)
    assert w[:, :8].max() == 0 and w[:, :, 24:].max() == 0 and w[:, 16, 16].min() > 0


def test_kz_golden_rotation_preserves_radius():
    traj, _ = make_trajectory(TrajectoryConfig(radial_spokes=5, base_resolution=32, phases=2))
    rot = kz_trajectories(traj.numpy(), 4, "golden")
    assert rot.shape == (2, 4, 5 * 64, 2) and kz_trajectories(traj.numpy(), 4, "aligned") is None
    radius = np.broadcast_to(np.linalg.norm(traj.numpy(), axis=-1)[:, None], rot.shape[:-1])
    np.testing.assert_allclose(np.linalg.norm(rot, axis=-1), radius, atol=1e-5)
    assert not np.allclose(rot[:, 0], rot[:, 1])


def _cplx(shape, seed=0):
    r = np.random.default_rng(seed)
    return (r.standard_normal(shape) + 1j * r.standard_normal(shape)).astype(np.complex64)


@pytest.mark.parametrize("kz", ["aligned", "golden"])
def test_stack_of_stars_adjoint(kz):
    traj, dcw = make_trajectory(TrajectoryConfig(radial_spokes=7, base_resolution=32, phases=2))
    kzt = kz_trajectories(traj.numpy(), 4, kz)
    op = SamplingOperator(traj, dcw, [32, 32], "cartesian_kz", None if kzt is None else tf.constant(kzt))
    x = _cplx((2, 3, 4, 32, 32))
    Ax = op.forward(x).numpy()
    y = _cplx(Ax.shape, 1)
    AHy = op.adjoint(tf.constant(y)).numpy()
    lhs, rhs = np.vdot(y, Ax), np.vdot(AHy, x)
    assert abs(lhs - rhs) / abs(lhs) < 1e-3


def test_stack_of_stars_fully_sampled_reconstructs():
    n = 32
    traj, dcw = make_trajectory(TrajectoryConfig(radial_spokes=int(np.pi / 2 * n * 2), base_resolution=n,
                                                 phases=1, ordering="linear"))
    op = SamplingOperator(traj, dcw, [n, n], "cartesian_kz")
    yy, xx = np.mgrid[:n, :n] - n / 2
    disk = ((yy ** 2 + xx ** 2) < (n / 3) ** 2).astype(np.complex64)
    x = np.stack([disk * (z + 1) for z in range(4)])[None, None]               # (1, 1, 4, n, n)
    rec = op.adjoint(op.apply_dcw(op.forward(x))).numpy()
    scale = np.vdot(rec, x).real / np.vdot(rec, rec).real
    nrmse = np.linalg.norm(scale * rec - x) / np.linalg.norm(x)
    assert nrmse < 0.15, nrmse
    # slices stay separated (Cartesian kz is exact): slice ratios preserved
    means = np.abs(rec[0, 0]).mean(axis=(1, 2))
    np.testing.assert_allclose(means / means[0], [1, 2, 3, 4], rtol=0.05)


def test_cartesian_kz_with_one_slice_equals_2d_operator():
    traj, dcw = make_trajectory(TrajectoryConfig(radial_spokes=7, base_resolution=32, phases=2))
    x = _cplx((2, 3, 32, 32))
    k2 = SamplingOperator(traj, dcw, [32, 32]).forward(x).numpy()
    k3 = SamplingOperator(traj, dcw, [32, 32], "cartesian_kz").forward(x[:, :, None]).numpy()
    np.testing.assert_allclose(k3[:, :, 0], k2, atol=1e-5)


def _small_3d_case(seed=0, z=4, t=8, n=64, spokes=21):
    sim_cfg = SimulationConfig(mode="3d", base_resolution=n, phases=t, slices=z, slice_spacing_px=8.0,
                               num_coils=(6,), regsnr=(30,))
    rng = RNG({"transforms": seed + 1})
    r = np.random.default_rng(seed)
    video = r.uniform(60, 200, (t, z, n, n, 2))
    sim = simulate(video, sim_cfg, rng)
    traj, dcw = make_trajectory(TrajectoryConfig(radial_spokes=spokes, base_resolution=n, phases=t,
                                                 ordering="golden_half"))
    return sim, traj, dcw


def test_simulate_3d_shapes_and_shared_mask():
    sim, _, _ = _small_3d_case()
    k = sim["kspace"].numpy()
    assert k.shape == (6, 8, 4, 64, 64)
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
    outside = sim["mask"] == 0
    sos = np.sqrt((np.abs(img) ** 2).sum(0))
    assert sos[:, outside].mean() < 0.05 * sos[:, ~outside].mean()   # same support in every frame


def test_time_avg_tukey_maps_match_true_coils():
    sim, traj, dcw = _small_3d_case()
    cfg = ReconConfig(base_resolution=64, phases=8, input_format="complex", coil_maps="time_avg_tukey_3d",
                      tukey_width=(1.0, 0.5, 0.5))
    out = reconstruct_3d(sim["kspace"], traj, dcw, cfg)
    est = np.moveaxis(out["maps"].numpy(), -1, 0)                              # (C, Z, H, W)
    true = sim["smaps"]
    inside = sim["mask"] > 0
    # voxelwise phase-invariant correlation across coils
    num = np.abs((np.conj(est) * true).sum(0))
    den = np.linalg.norm(est, axis=0) * np.linalg.norm(true, axis=0) + 1e-12
    corr = (num / den)[inside]
    assert np.median(corr) > 0.9, np.median(corr)
    assert out["zfill"].shape == (8, 4, 64, 64, 2) and out["image"].shape == (8, 4, 64, 64)
    # anchored maps: phase of coil 0 is zero
    assert np.abs(np.angle(out["maps"].numpy()[..., 0][inside])).max() < 1e-3


def test_augment_3d_same_transform_for_all_slices():
    base = np.random.default_rng(0).random((32, 1, 16, 16, 2)).astype(np.float32)
    zfill = np.repeat(base, 5, axis=1)
    image = zfill[..., :1].copy()
    zr, ir = augment_3d(zfill, image, AugmentConfig(phases=24, roll=1, rotation=1, z_flip=1), RNG())
    assert zr.shape == (24, 5, 16, 16, 2) and ir.shape == (24, 5, 16, 16, 1)
    for s in range(1, 5):
        np.testing.assert_array_equal(zr[:, s], zr[:, 0])
    np.testing.assert_array_equal(ir[..., 0], zr[..., 0])
