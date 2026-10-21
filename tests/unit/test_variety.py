import json

import numpy as np
import pytest

from synthmr.phantoms.generate import FractalDomain, iter_recipes_3dt
from synthmr.phantoms.recipes import render_recipe
from synthmr.phantoms.variety import Variety, apply_framing, random_rotation

CANDS = np.array([[-0.02, 0.38, -0.53, -0.47], [0.1, 0.6, -0.2, 0.1], [-0.4, 0.5, 0.1, 0.2]])
DOM = FractalDomain(shape=(24, 24), frames=4, slices=3)


def _run(steps, n=3):
    return list(iter_recipes_3dt(DOM, n, 0, norm="volume", candidates=CANDS, variety=Variety.from_steps(steps)))


def test_identity_framing_matches_plain_render():
    (_, recipe, block), = _run("none", 1)
    assert "framing" not in recipe
    framing = {"matrix": np.eye(4).tolist(), "origin": [0.0] * 4, "centre": [0.0] * 4}
    np.testing.assert_array_equal(apply_framing(DOM.grid().coordinates(), framing), DOM.grid().coordinates())
    np.testing.assert_array_equal(render_recipe({**recipe, "framing": framing}, DOM.grid(), norm_axes=None), block)


def test_steps_keep_recipes_and_rerender_from_json():
    plain = _run("none")
    varied = _run("frame,background,warp,rotate")
    for (_, r0, _), (_, r1, block) in zip(plain, varied):
        assert r1["c"] == r0["c"]
        assert [f["kind"] for f in r1["real"]["fields"][:len(r0["real"]["fields"])]] == \
            [f["kind"] for f in r0["real"]["fields"]]
        assert any(f["kind"] in ("distance", "orbit_trap") for f in r1["real"]["fields"])
        assert np.isclose(sum(r1["imag"]["weights"]), 1)
        assert block.shape == DOM.grid().shape and np.isfinite(block).all()
        again = render_recipe(json.loads(json.dumps(r1, default=list)), DOM.grid(), norm_axes=None)
        np.testing.assert_allclose(again, block, atol=1e-6)


def test_reject_reaches_min_detail():
    for _, recipe, _ in _run("frame,reject"):
        assert recipe["framing"]["detail"] >= 0.6


def test_random_rotation_is_special_orthogonal():
    rng = np.random.default_rng(1)
    for _ in range(5):
        r = random_rotation(rng)
        np.testing.assert_allclose(r @ r.T, np.eye(4), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1)


def test_unknown_step():
    with pytest.raises(ValueError):
        Variety.from_steps("frame,zoomies")


def test_contrast_keeps_other_draws_and_rerenders_from_json():
    base = _run("frame,warp")
    dyn = _run("frame,warp,contrast")
    for (_, r0, b0), (_, r1, b1) in zip(base, dyn):
        assert {k: v for k, v in r1.items() if k != "contrast"} == r0
        assert len(r1["contrast"]["curves"]) == len(r1["contrast"]["levels"]) + 2
        assert b1.shape == b0.shape and np.isfinite(b1).all()
        assert np.abs(b1.real).max() <= 1 + 1e-6 and np.abs(b1.imag).max() <= 1 + 1e-6
        again = render_recipe(json.loads(json.dumps(r1, default=list)), DOM.grid(), norm_axes=None)
        np.testing.assert_allclose(again, b1, atol=1e-6)


def test_contrast_curves_and_memberships():
    from synthmr.phantoms.contrast import CURVES, curve, memberships
    u = np.linspace(0, 1, 2001)
    rng = np.random.default_rng(0)
    for kind, draw in CURVES.items():
        onset, tau_in, tau_out, scale = draw(rng)
        c = {"onset": onset, "tau_in": tau_in, "tau_out": tau_out, "peak": 1.5 * scale}
        e = curve(c, u)
        assert np.all(e[u <= onset] == 0) and e.min() >= 0
        if kind == "static":
            assert not e.any()
        elif tau_out is not None:
            assert np.isclose(e.max(), c["peak"], rtol=1e-3)
    labels = np.random.default_rng(1).integers(0, 3, size=DOM.grid().shape)
    m = memberships(labels, 3, DOM.grid(), 1.5)
    np.testing.assert_allclose(m.sum(0), 1, atol=1e-5)
    assert m.min() >= 0


def test_contrast_scales_tissues_by_their_curves():
    from synthmr.phantoms.contrast import curve, signal_scale, time_axis, tissue_labels
    from synthmr.phantoms.fractal import JuliaConfig
    (_, recipe, _), = _run("frame,contrast", 1)
    grid = DOM.grid()
    cfg = JuliaConfig(c=tuple(recipe["c"]), max_iter=DOM.max_iter)
    coords = apply_framing(grid.coordinates(), recipe["framing"])
    con = dict(recipe["contrast"], softness=0.0)     # hard labels: every voxel follows its tissue's curve
    s = signal_scale(coords, grid, con, cfg)
    labels = tissue_labels(coords, con, cfg)
    u = np.broadcast_to(time_axis(grid), s.shape)
    expected = np.choose(labels, [1 + curve(c, u) for c in con["curves"]])
    np.testing.assert_allclose(s, expected, atol=1e-5)
    assert len(np.unique(labels)) > 1
