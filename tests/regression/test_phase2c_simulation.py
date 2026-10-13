"""2c: synthmr.simulation reproduces the legacy simulated k-space exactly (R4)."""

import numpy as np
import pytest
import tensorflow as tf

from legacy.app.inter4k_utils import preprocessing_multicoil_noselect as pmc
from legacy.app.inter4k_utils import preprocessing_natural_videos as pnv
from legacy.app.inter4k_utils import preprocessing_trajectory_gen as ptg
from synthmr.backend.tf.simulation import kspace_to_legacy
from synthmr.rng import RNG
from synthmr.simulation import SimulationConfig, simulate
from tests.regression import fixtures as fx
from tests.regression.compare import check_stage

CASES = [("A", "A", None), ("B_cc", "B_cc", None),
         ("B_grouped", "B_grouped_slice_1", "julia_00"), ("B_grouped", "B_grouped_slice_2", "julia_00")]


@pytest.mark.parametrize("fixture,case,group", CASES)
def test_r4_kspace_matches_golden(fixture, case, group, golden_dir):
    config_ni, _, _ = fx.legacy_configs(fixture, pnv, pmc, ptg)
    rng = RNG()
    if group is not None:
        # Grouped legacy run: transforms re-seeded per slice, noise stream continues across slices.
        rng.reset(transforms=fx.group_seed(group))
        if case.endswith("_2"):
            # replay slice 1 to advance the noise stream exactly as in the legacy loop
            simulate(np.load(golden_dir / "B_grouped_slice_1/video_uint8.npy"),
                     SimulationConfig.from_legacy(config_ni), rng)
            rng.reset(transforms=fx.group_seed(group))
    video = tf.convert_to_tensor(np.load(golden_dir / f"{case}/video_uint8.npy"))
    out = simulate(video, SimulationConfig.from_legacy(config_ni), rng)
    ok, msg = check_stage("kspace", np.load(golden_dir / f"{case}/kspace.npy"),
                          kspace_to_legacy(out["kspace"]).numpy())
    assert ok, msg
    assert out["smaps"].shape[:2] == (config_ni["num_coils"][0], 1)
