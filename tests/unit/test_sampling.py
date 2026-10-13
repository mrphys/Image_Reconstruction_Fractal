import numpy as np
import pytest

from synthmr.sampling import TrajectoryConfig, make_trajectory


def test_radial_shapes():
    traj, dcw = make_trajectory(TrajectoryConfig(radial_spokes=13, base_resolution=64, phases=4,
                                                 ordering="golden_half"))
    assert tuple(traj.shape) == (4, 13 * 128, 2) and tuple(dcw.shape) == (4, 13 * 128)
    assert np.abs(traj.numpy()).max() <= np.pi + 1e-6


def test_spiral_smoke():
    """Ported verbatim from legacy; needs tfmri spiral support (spiral-waveform)."""
    cfg = TrajectoryConfig(kind="spiral", base_resolution=128, phases=2, ordering="tiny", reverse=False)
    try:
        traj, dcw = make_trajectory(cfg)
    except Exception as e:  # pragma: no cover - environment without spiral-waveform
        pytest.skip(f"spiral trajectory unavailable: {e}")
    assert traj.shape[-1] == 2 and traj.shape[0] == 2 and dcw.shape[:2] == traj.shape[:2]
