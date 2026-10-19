"""sigpy backend: sampling operator properties and JSENSE coil maps (no TensorFlow needed)."""

import numpy as np
import pytest

pytest.importorskip("sigpy")

from synthmr.backend import load_backend  # noqa: E402
from synthmr.pipeline import Pipeline, load_config  # noqa: E402
from synthmr.sampling.encoders import kz_trajectories  # noqa: E402

B = load_backend("sigpy")
from synthmr.backend.sigpy.sampling import SamplingOperator  # noqa: E402


def _cplx(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _star(n, spokes, phases, ordering="golden_half"):
    traj = B.flatten_trajectory(B.radial_trajectory(n, spokes, phases, ordering))
    dens = B.estimate_radial_density(traj)
    return traj, 1.0 / dens


@pytest.mark.parametrize("through_plane,kz_rotation", [("none", "aligned"), ("cartesian_kz", "aligned"),
                                                       ("cartesian_kz", "golden")])
def test_sampling_operator_adjoint(through_plane, kz_rotation):
    rng = np.random.default_rng(0)
    traj, dcw = _star(16, 5, 3)
    shape = (3, 2, 16, 16) if through_plane == "none" else (3, 2, 4, 16, 16)
    kz = kz_trajectories(B.to_numpy(traj), 4, kz_rotation) if through_plane != "none" else None
    op = SamplingOperator(traj, dcw, [16, 16], through_plane=through_plane,
                          kz_traj=None if kz is None else B.from_numpy(kz))
    x = B.from_numpy(_cplx(rng, shape))
    ax = op.forward(x)
    y = B.from_numpy(_cplx(rng, ax.shape))
    lhs, rhs = np.vdot(B.to_numpy(y), B.to_numpy(ax)), np.vdot(B.to_numpy(op.adjoint(y)), B.to_numpy(x))
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_fully_sampled_star_reconstructs_image():
    """Dense radial sampling + density compensation ~ identity (up to a global scale) in the image centre."""
    n = 32
    yy, xx = np.mgrid[:n, :n] - n // 2
    img = np.exp(-(yy ** 2 + xx ** 2) / 40.0).astype(np.complex64) * np.exp(1j * 0.02 * xx)
    traj, dcw = _star(n, 64, 1, "linear")
    op = SamplingOperator(traj, dcw, [n, n])
    rec = B.to_numpy(op.adjoint(op.apply_dcw(op.forward(B.from_numpy(img[None, None])))))[0, 0]
    c = slice(n // 4, 3 * n // 4)
    scale = np.vdot(rec[c, c], img[c, c]) / np.vdot(rec[c, c], rec[c, c])
    assert np.abs(scale * rec[c, c] - img[c, c]).max() < 0.05 * np.abs(img).max()


def _map_corr(maps, smaps, mask):
    s = np.moveaxis(smaps, 0, -1)
    num = np.abs((maps * np.conj(s)).sum(-1))
    den = np.linalg.norm(maps, axis=-1) * np.linalg.norm(s, axis=-1)
    return float((num / np.maximum(den, 1e-12))[mask > 0].mean())


def _run(preset, shape, n, extra):
    small = {"backend": "sigpy", "rng": "numpy",
             "simulation": {"base_resolution": n, "phases": shape[0], "regsnr": [30], "slices": 4},
             "trajectory": {"base_resolution": n, "phases": shape[0], "spokes": 21},
             "recon": {"base_resolution": n, "phases": shape[0], "coil_maps": "jsense"}, "augment": {"phases": 4}}
    cfg = load_config(preset, overrides=small).updated(extra)
    v = np.random.default_rng(2).standard_normal(shape) * (1 + 0j)
    return Pipeline(cfg).run((v / np.abs(v).max()).astype(np.complex64), return_stages=True)


def test_jsense_2d_maps_match_true_coils():
    s = _run("fractal_new_2dt_nocc", (6, 48, 48), 48, {})
    st = s.stages
    assert st["maps"].shape == (48, 48, 1, 10)
    assert _map_corr(st["maps"][:, :, 0][None], st["sim_smaps"], st["sim_mask"]) > 0.99
    assert s.input.shape == (4, 48, 48, 2) and np.isfinite(s.input).all()


def test_jsense_3d_maps_match_true_coils():
    """Tiny problem (4 partitions, 32^2, 4 frames): JSENSE must beat the Tukey-Walsh maps (~0.79 here)."""
    s = _run("fractal_3dt_sos", (4, 4, 32, 32), 32,
             {"recon": {"jsense_max_iter": 5, "jsense_ker_width": 8, "jsense_calib_width": 16}})
    st = s.stages
    assert st["maps"].shape == (4, 32, 32, 10)
    walsh = _run("fractal_3dt_sos", (4, 4, 32, 32), 32, {"recon": {"coil_maps": "time_avg_tukey_3d"}}).stages
    jsense_corr = _map_corr(st["maps"], st["sim_smaps"], st["sim_mask"])
    assert jsense_corr > 0.85 and jsense_corr > _map_corr(walsh["maps"], walsh["sim_smaps"], walsh["sim_mask"])
    assert s.input.shape == (4, 4, 32, 32, 2) and np.isfinite(s.input).all()


def test_spiral_generation_points_to_file_trajectories():
    from synthmr.sampling import SpiralTrajectoryConfig, make_trajectory
    with pytest.raises(NotImplementedError, match="kind='file'"):
        make_trajectory(SpiralTrajectoryConfig(), backend=B)
