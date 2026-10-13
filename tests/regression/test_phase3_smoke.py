"""Phase 3 end-to-end: fractal_3dt_sos preset on a full-size 4-D fractal; writes GIFs for visual sign-off.

No golden exists for 3D+t (behaviour intentionally differs from the per-slice legacy hack).
"""

import numpy as np

from synthmr.phantoms.generate import RecipeDomain, default_candidates
from synthmr.phantoms.recipes import random_recipe, render_recipe
from synthmr.pipeline import Pipeline, get_preset, load_sample, save_sample
from synthmr.viz.gifs import save_ortho_gif, save_panel_gif, save_slice_mosaic_gif
from tests.regression import fixtures as fx


def test_fractal_3dt_sos_end_to_end(golden_np_dir, artifacts_dir, tmp_path):
    out = artifacts_dir / "phase3/smoke"
    r = np.random.default_rng(fx.B_RECIPE_SEED)
    cands = default_candidates()
    recipe = random_recipe(r, cands[r.choice(len(cands))])           # same recipe as fixture B
    volume = render_recipe(recipe, RecipeDomain(slices=10).grid(), max_iter=30, norm_axes=None)
    assert volume.shape == (32, 10, 256, 256)

    cfg = get_preset("fractal_3dt_sos")
    sample = Pipeline(cfg).run(volume, return_stages=True)
    assert sample.input.shape == (24, 10, 256, 256, 2) and sample.target.shape == (24, 10, 256, 256, 1)
    assert np.isfinite(sample.input).all() and np.isfinite(sample.target).all()
    st = sample.stages

    save_sample(sample, tmp_path, "julia_000", fmt="h5")
    inp, tgt = load_sample(tmp_path, "julia_000", fmt="h5")
    np.testing.assert_array_equal(inp, sample.input)

    # consistency across slices: the simulated object mask is one ellipsoid, coil maps vary smoothly in z
    mask = st["sim_mask"]
    assert (mask.sum(axis=(1, 2)) > 0).all()
    est = np.moveaxis(st["maps"], -1, 0)
    true = st["sim_smaps"]
    inside = mask > 0
    corr = np.abs((np.conj(est) * true).sum(0)) / (np.linalg.norm(est, axis=0) * np.linalg.norm(true, axis=0) + 1e-12)
    print("median voxelwise |corr(estimated, true maps)| inside object:", np.median(corr[inside]))
    assert np.median(corr[inside]) > 0.8

    zin = np.abs(sample.input[..., 0] + 1j * sample.input[..., 1])
    save_slice_mosaic_gif(volume, out / "00_fractal_mosaic.gif", "fractal |.| (T x 10 slices)")
    save_slice_mosaic_gif(zin, out / "01_input_mosaic.gif", "zero-filled input |.|")
    save_slice_mosaic_gif(sample.target[..., 0], out / "02_target_mosaic.gif", "target")
    save_ortho_gif(sample.target[..., 0], out / "03_target_ortho.gif", "target")
    save_ortho_gif(zin, out / "04_input_ortho.gif", "input")
    # true vs estimated coil maps (|.|), coils as frames, middle slice
    zmid = mask.shape[0] // 2
    save_panel_gif([np.abs(true[:, zmid]), np.abs(est[:, zmid])], ["true |S_c| (z mid)", "estimated |S_c|"],
                   out / "05_coil_maps.gif", fps=2)
    save_panel_gif([np.abs(true[:, :, 128, :]).repeat(10, axis=1), np.abs(est[:, :, 128, :]).repeat(10, axis=1)],
                   ["true |S_c| XZ", "estimated |S_c| XZ"], out / "06_coil_maps_xz.gif", fps=2)
