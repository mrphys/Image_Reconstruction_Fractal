"""Run the refactored (synthmr) code on the regression fixtures, returning the same stages as golden."""

import numpy as np

from legacy.app.inter4k_utils import preprocessing_multicoil_noselect as pmc
from legacy.app.inter4k_utils import preprocessing_natural_videos as pnv
from legacy.app.inter4k_utils import preprocessing_trajectory_gen as ptg
from synthmr.backend.tf.simulation import kspace_to_legacy
from synthmr.phantoms import complexify
from synthmr.phantoms.fractal import JuliaConfig, candidate_c, evaluate_field
from synthmr.phantoms.recipes import random_recipe, render_recipe
from synthmr.pipeline import Pipeline, get_preset
from synthmr.recon import AugmentConfig, ReconConfig
from synthmr.sampling import RadialTrajectoryConfig
from synthmr.simulation import SimulationConfig
from tests.regression import fixtures as fx
from tests.regression.test_phase2b_fractal import fixture_a_grid, fixture_b_grid


PRESET_FOR = {"A": "fractal_2dt_legacy", "B_cc": "fractal_new_2dt", "B_grouped": "fractal_new_2dt_nocc"}


def configs(fixture):
    """synthmr configs built from the legacy config dicts of ``fixture`` (used to validate presets)."""
    ni, pp, traj = fx.legacy_configs(fixture, pnv, pmc, ptg)
    return (SimulationConfig.from_legacy(ni), RadialTrajectoryConfig.from_legacy(traj),
            ReconConfig.from_legacy(pp), AugmentConfig(phases=fx.ROLL_KWARGS["phases"], rotation=fx.ROLL_KWARGS["rotation"]))


def run_stages(volume, pipeline):
    """Run ``pipeline`` on one phantom volume and map its stages to golden stage names."""
    sample = pipeline.run(volume, return_stages=True)
    st = sample.stages
    return {"video_uint8": st["video_uint8"], "kspace": kspace_to_legacy(st["sim_kspace"]).numpy(),
            "traj": st["traj"], "dcw": st["dcw"], "uskspace": st["uskspace"], "zfill": st["zfill"],
            "image": st["image"], "zfill_rolled": sample.input, "image_rolled": sample.target}


def run_fixture(name):
    pipeline = Pipeline(get_preset(PRESET_FOR[name]))
    if name == "A":
        cfg = JuliaConfig(c=fx.A_C, max_iter=fx.A_MAX_ITER, bailout=4.0)
        escape = evaluate_field(fixture_a_grid().coordinates(), "escape", {"config": cfg}).astype(np.int32)
        np.random.seed(fx.A_NP_SEED)                       # preset draws from the legacy global np.random
        stages = run_stages(escape, pipeline)
        np.random.seed(fx.A_NP_SEED)
        video = complexify.sine_wave(escape, rng=np.random)
        return {"A": {"fractal_escape": escape, "fractal_video": video, **stages}}

    r = np.random.default_rng(fx.B_RECIPE_SEED)
    cands = candidate_c(resolution=40, extent=1.5, max_iter=50, min_iter=10, max_escape=30)
    recipe = random_recipe(r, cands[r.choice(len(cands))])
    block = render_recipe(recipe, fixture_b_grid(), max_iter=30, norm_axes=(-2, -1))   # (T, Z, H, W)
    if name == "B_cc":
        vol = block[:, 0]
        return {"B_cc": {"fractal_volume": vol, **run_stages(vol, pipeline)}}
    out, seed = {}, fx.group_seed("julia_00")
    for s in range(block.shape[1]):
        pipeline.rng.reset(transforms=seed, augment=seed)
        vol = block[:, s]
        out[f"B_grouped_slice_{s + 1}"] = {"fractal_volume": vol, **run_stages(vol, pipeline)}
    return out
