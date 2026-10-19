"""Trajectory + density loaded from .npy files (``TrajectoryConfig(kind="file")``)."""

import numpy as np
import pytest

from synthmr.backend import set_backend
from synthmr.pipeline import Pipeline, load_config
from synthmr.sampling import FileTrajectoryConfig, TrajectoryConfig
from synthmr.sampling.files import load_trajectory_files
from synthmr.utils.array import complex_from_channels
from tests.backends import available_backends

N = 32


def _radial(t=3, v=5, s=2 * N):
    ang = np.arange(t * v).reshape(t, v, 1) * 0.7
    r = np.linspace(-np.pi, np.pi, s, endpoint=False)
    return np.stack([np.cos(ang) * r, np.sin(ang) * r], -1).astype(np.float32)        # (T, V, S, 2)


def _save(tmp_path, traj, dens):
    np.save(tmp_path / "traj.npy", traj)
    np.save(tmp_path / "dens.npy", dens)
    return dict(traj_path=str(tmp_path / "traj.npy"), density_path=str(tmp_path / "dens.npy"), base_resolution=N)


def test_shapes_units_and_density_kind(tmp_path):
    traj = _radial()
    dens = 1.0 + np.abs(traj).sum(-1)
    paths = _save(tmp_path, traj, dens)
    t, w = load_trajectory_files(FileTrajectoryConfig(**paths))
    assert t.shape == (3, 5 * 2 * N, 2) and w.shape == (3, 5 * 2 * N)
    np.testing.assert_allclose(t, traj.reshape(3, -1, 2))
    np.testing.assert_allclose(w, 1.0 / dens.reshape(3, -1), rtol=1e-6)

    _save(tmp_path, traj * N / (2 * np.pi), dens)                       # sigpy pixel units, dcw given directly
    t2, w2 = load_trajectory_files(FileTrajectoryConfig(**paths, traj_units="pixels", density_kind="dcw"))
    np.testing.assert_allclose(t2, t, atol=1e-6)
    np.testing.assert_allclose(w2, dens.reshape(3, -1))

    _save(tmp_path, traj[0, 0], dens[0, 0])                              # (R, 2): one frame
    assert load_trajectory_files(FileTrajectoryConfig(**paths))[0].shape == (1, 2 * N, 2)
    _save(tmp_path, traj[0], dens[0])                                    # (V, S, 2) with views_axis
    assert load_trajectory_files(FileTrajectoryConfig(**paths, views_axis=True))[0].shape == (1, 5 * 2 * N, 2)
    t3 = load_trajectory_files(FileTrajectoryConfig(**paths, views_axis=True, reverse=True))[0]
    np.testing.assert_allclose(t3[..., ::-1], traj[0].reshape(1, -1, 2))


def test_rejects_bad_inputs(tmp_path):
    traj = _radial()
    paths = _save(tmp_path, traj * 2, np.ones(traj.shape[:-1]))
    with pytest.raises(ValueError, match="exceeds"):
        load_trajectory_files(FileTrajectoryConfig(**paths))
    paths = _save(tmp_path, traj, np.ones(traj.shape[:-2]))
    with pytest.raises(ValueError, match="density shape"):
        load_trajectory_files(FileTrajectoryConfig(**paths))


def test_from_dict_and_kind_switch_override():
    cfg = TrajectoryConfig.from_dict({"kind": "file", "traj_path": "a.npy", "density_path": "b.npy"})
    assert isinstance(cfg, FileTrajectoryConfig) and cfg.reverse is False
    p = load_config("fractal_new_2dt", overrides={"trajectory": {"kind": "file", "traj_path": "a.npy"}})
    assert p.trajectory.kind == "file" and p.trajectory.reverse is False and p.trajectory.ordering == "golden_half"


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize("preset,shape", [("fractal_new_2dt", (6, N, N)), ("fractal_3dt_sos", (6, 3, N, N))])
def test_file_trajectory_reproduces_generated_radial(tmp_path, backend, preset, shape):
    """Saving the generated radial trajectory/density and loading it back gives the same pipeline output."""
    B = set_backend(backend)
    small = {"backend": backend, "rng": "numpy", "simulation": {"base_resolution": N, "phases": 6, "slices": 3},
             "trajectory": {"base_resolution": N, "phases": 6, "spokes": 8},
             "recon": {"base_resolution": N, "phases": 6}, "augment": {"phases": 4}}
    cfg = load_config(preset, overrides=small)
    gen = Pipeline(cfg)
    traj, dcw = gen.trajectory()
    np.save(tmp_path / "traj.npy", B.to_numpy(traj))
    np.save(tmp_path / "dcw.npy", B.to_numpy(dcw))
    fcfg = cfg.updated({"trajectory": {"kind": "file", "traj_path": str(tmp_path / "traj.npy"),
                                       "density_path": str(tmp_path / "dcw.npy"), "density_kind": "dcw"}})
    vol = np.random.default_rng(0).uniform(-1, 1, shape).astype(np.float32) * (1 + 0j)
    a, b = gen.run(vol), Pipeline(fcfg).run(vol)
    # Walsh-combined input: the eigenvector phase is unanchored (see tolerances.json), compare magnitudes
    np.testing.assert_allclose(np.abs(complex_from_channels(b.input)), np.abs(complex_from_channels(a.input)),
                               atol=1e-5)
    np.testing.assert_allclose(b.target, a.target, atol=1e-5)
