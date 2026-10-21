"""2b: synthmr.phantoms reproduces the legacy fractal stages exactly.

R1  escape field on a synthmr Grid == base-repo julia_escape (fixture A)
R2  render_recipe on a 4-D (T, Z, Y, X) grid == julia_generator_new_grasp.render_volume per y (fixture B)
R3  sine_wave / complex_to_video == golden videos
"""

import numpy as np
import pytest

from legacy.app import julia_generator_new_grasp as gen
from synthmr.phantoms import complexify
from synthmr.phantoms.fractal import Axis, Grid, JuliaConfig, evaluate_field
from synthmr.phantoms.fractal import legacy as fl
from synthmr.phantoms.recipes import random_recipe, render_recipe
from tests.regression import fixtures as fx
from tests.regression.compare import check_stage


def fixture_a_grid():
    """Legacy build_grid([1,1,0,0.2], [256,256,1,32]) as a (T, Y, X) Grid."""
    return Grid(axes=[Axis.linspace("t", 3, -0.2, 0.2, 32),
                      Axis.linspace("y", 0, -1, 1, 256),
                      Axis.linspace("x", 1, -1, 1, 256)])


def fixture_b_grid(n_slices=fx.B_N_Y_SLICES):
    """julia_generator_new_grasp domain: rows = x (comp 1), cols = w (comp 0), z -> T, y -> Z."""
    (r0, r1), (c0, c1) = gen.BOUNDS
    return Grid(axes=[Axis("t", 3, tuple(gen.Z_SLICES.tolist())),
                      Axis("z", 2, tuple(gen.Y_SLICES[:n_slices].tolist())),
                      Axis.linspace("y", 1, r0, r1, gen.SHAPE[0]),
                      Axis.linspace("x", 0, c0, c1, gen.SHAPE[1])])


def assert_stage(stage, ref, new):
    ok, msg = check_stage(stage, ref, new)
    assert ok, f"{stage}: {msg}"


def test_r1_escape_field_matches_julia_escape(golden_np_dir):
    ref = np.load(golden_np_dir / "A/fractal_escape.npy")
    cfg = JuliaConfig(c=fx.A_C, max_iter=fx.A_MAX_ITER, bailout=4.0)
    new = evaluate_field(fixture_a_grid().coordinates(), "escape", {"config": cfg})
    assert_stage("fractal_escape", ref, new.astype(np.int32))


def test_r1_legacy_module_matches_golden(golden_np_dir):
    ref = np.load(golden_np_dir / "A/fractal_escape.npy")
    grid = fl.build_grid(fx.A_GRID_LIMITS, fx.A_GRID_SIZES)
    escape = np.squeeze(fl.julia_escape(grid, np.asarray(fx.A_C), max_iter=fx.A_MAX_ITER))
    assert fl.test_complexity(escape)
    assert_stage("fractal_escape", ref, np.transpose(escape, [2, 0, 1]))


def test_r2_recipe_block_matches_render_volume(golden_np_dir):
    rng = np.random.default_rng(fx.B_RECIPE_SEED)
    from synthmr.phantoms.fractal import candidate_c
    cands = candidate_c(resolution=40, extent=1.5, max_iter=50, min_iter=10, max_escape=30)
    recipe = random_recipe(rng, cands[rng.choice(len(cands))])
    block = render_recipe(recipe, fixture_b_grid(), max_iter=gen.MAX_ITER, norm_axes=(-2, -1))
    assert block.shape == (32, fx.B_N_Y_SLICES, 256, 256)
    assert_stage("fractal_volume", np.load(golden_np_dir / "B_cc/fractal_volume.npy"), block[:, 0])
    for s in range(fx.B_N_Y_SLICES):
        ref = np.load(golden_np_dir / f"B_grouped_slice_{s + 1}/fractal_volume.npy")
        assert_stage("fractal_volume", ref, block[:, s])


def test_r3_sine_wave_matches_golden_video(golden_np_dir):
    escape = np.load(golden_np_dir / "A/fractal_escape.npy")
    np.random.seed(fx.A_NP_SEED)
    video = complexify.sine_wave(escape, rng=np.random)
    assert_stage("fractal_video", np.load(golden_np_dir / "A/fractal_video.npy"), video)


@pytest.mark.parametrize("case", ["B_cc", "B_grouped_slice_2"])
def test_r3_complex_to_video_matches_golden(golden_np_dir, case):
    volume = np.load(golden_np_dir / f"{case}/fractal_volume.npy")
    video = np.clip(complexify.complex_to_video(volume), 0, 255).astype(np.uint8)
    assert_stage("video_uint8", np.load(golden_np_dir / f"{case}/video_uint8.npy"), video)
