"""Deterministic fixtures shared by the golden capture and the regression tests.

Each fixture fully specifies the input fractal and the pipeline configuration of
one legacy pipeline, so the refactored code can be run on exactly the same input
and compared stage-by-stage against the golden outputs of the old code.

Fixtures
--------
A          Base-repo 2D+t path (``train_network.ipynb``): escape-time Julia set via
           ``fractal_utils.julia_escape`` -> ``make_complex_fractal`` -> radial
           pipeline with coil compression.
B_cc       New fractal framework -> ``process_fractals_new_grasp`` (coil compressed).
B_grouped  New fractal framework, 2 y-slices -> ``process_fractals_new_grasp_grouped``
           (no coil compression, RNG re-seeded per group so slices share transforms).
"""

import zlib

import numpy as np

# Seeds of the module-level generators at import time in the legacy utils.
# Resetting to these reproduces "this fixture is the first file of a fresh run".
LEGACY_RG_SEED = 1
LEGACY_NOISE_RG_SEED = 2

# --- Fixture A ---------------------------------------------------------------
A_C = (-0.02, 0.38, -0.53, -0.47)
A_GRID_LIMITS = [1, 1, 0, 0.2]
A_GRID_SIZES = [256, 256, 1, 32]
A_MAX_ITER = 50
A_NP_SEED = 1234  # global np.random seed before make_complex_fractal

# --- Fixture B ---------------------------------------------------------------
B_RECIPE_SEED = 0
B_N_Y_SLICES = 2  # rendered y-slices (subset of the generator's 10)


def legacy_configs(fixture, pnv, pmc, ptg):
    """Return ``(config_natural_images, config_preproc, config_traj)`` exactly as
    the legacy entry point for ``fixture`` builds them.

    ``pnv``/``pmc``/``ptg`` are the legacy preprocessing_natural_videos,
    preprocessing_multicoil_noselect and preprocessing_trajectory_gen modules.
    """
    config_preproc = pmc.config_base_preproc()
    config_preproc['normalize_input'] = True
    if fixture == 'B_grouped':
        config_preproc['input_format'] = 'blah'
    config_ni = pnv.config_default(config_preproc['base_resolution'], config_preproc['phases'])
    config_ni['complex_transform'] = 1
    config_traj = ptg.config_radial_traj()
    config_traj['radial_spokes'] = 13
    if fixture == 'A':
        config_ni['num_coils'] = [15]
        config_traj['base_resolution'] = 256
        config_traj['ordering'] = 'sorted_half'
    else:
        config_ni['num_coils'] = [10]
        config_traj['ordering'] = 'golden_half'
    return config_ni, config_preproc, config_traj


ROLL_KWARGS = dict(phases=24, rotation=1)


def group_seed(group, seed=0):
    """Per-group seed used by ``process_fractals_new_grasp_grouped``."""
    return seed + zlib.crc32(group.encode())


def fixture_a_escape(fr):
    """Escape-time volume ``(T, H, W)`` for fixture A using base-repo ``fractal_utils``."""
    grid = fr.build_grid(limits=A_GRID_LIMITS, sizes=A_GRID_SIZES)
    escape = np.squeeze(fr.julia_escape(grid, np.asarray(A_C), max_iter=A_MAX_ITER))
    assert fr.test_complexity(escape), "fixture A fractal failed the complexity test"
    return np.transpose(escape, [2, 0, 1])


def fixture_b_recipe(gen, ff):
    """Seeded ``(c, recipe)`` drawn the same way as ``julia_generator_new_grasp.main``."""
    rng = np.random.default_rng(B_RECIPE_SEED)
    candidates = ff.candidate_c(resolution=40, extent=1.5, max_iter=50,
                                min_iter=10, max_escape=30)
    c = candidates[rng.choice(len(candidates))]
    recipe = {
        "c": [float(v) for v in c],
        "real": gen.random_channel(rng),
        "imag": gen.random_channel(rng),
    }
    return recipe


def fixture_b_volumes(gen, ff):
    """List of complex volumes ``(T, H, W)``, one per y-slice, plus the recipe."""
    recipe = fixture_b_recipe(gen, ff)
    renderer = ff.QuaternionJulia(ff.JuliaConfig(c=tuple(recipe["c"]), max_iter=gen.MAX_ITER))
    ys = gen.Y_SLICES[:B_N_Y_SLICES]
    return [gen.render_volume(renderer, recipe, y) for y in ys], recipe


def new_fractal_to_video(volume):
    """uint8-range video from a complex volume, as in ``process_fractals_new_grasp``."""
    return np.stack([volume.real, volume.imag], axis=-1) * 127.5 + 127.5
