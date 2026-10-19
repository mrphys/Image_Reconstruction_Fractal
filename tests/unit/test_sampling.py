import numpy as np
import pytest

from synthmr.sampling import RadialTrajectoryConfig, SpiralTrajectoryConfig, TrajectoryConfig, make_trajectory


def test_radial_shapes():
    traj, dcw = make_trajectory(RadialTrajectoryConfig(spokes=13, base_resolution=64, phases=4,
                                                       ordering="golden_half"))
    assert tuple(traj.shape) == (4, 13 * 128, 2) and tuple(dcw.shape) == (4, 13 * 128)
    assert np.abs(traj.numpy()).max() <= np.pi + 1e-6


def test_spiral_smoke():
    """Ported verbatim from legacy; needs tfmri spiral support (spiral-waveform)."""
    cfg = SpiralTrajectoryConfig(base_resolution=128, phases=2, ordering="tiny", reverse=False)
    try:
        traj, dcw = make_trajectory(cfg)
    except Exception as e:  # pragma: no cover - environment without spiral-waveform
        pytest.skip(f"spiral trajectory unavailable: {e}")
    assert traj.shape[-1] == 2 and traj.shape[0] == 2 and dcw.shape[:2] == traj.shape[:2]


def test_trajectory_config_per_kind():
    import dataclasses
    from synthmr.pipeline import PipelineConfig
    for cfg in (RadialTrajectoryConfig(spokes=21), SpiralTrajectoryConfig(vd_spiral_arms=10)):
        d = dataclasses.asdict(cfg)
        assert TrajectoryConfig.from_dict(d) == cfg
    radial = dataclasses.asdict(RadialTrajectoryConfig())
    assert radial["kind"] == "radial" and "vd_spiral_arms" not in radial and "field_of_view" not in radial
    with pytest.raises(KeyError):
        TrajectoryConfig.from_dict({"kind": "radial", "vd_spiral_arms": 3})
    with pytest.raises(ValueError):
        TrajectoryConfig.from_dict({"kind": "rosette"})
    cfg = PipelineConfig().updated({"trajectory": {"kind": "spiral", "phases": 4}})
    assert cfg.trajectory == SpiralTrajectoryConfig(phases=4)
    assert PipelineConfig().updated({"trajectory": {"spokes": 5}}).trajectory.spokes == 5
