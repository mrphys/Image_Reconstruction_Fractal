import numpy as np

from synthmr.phantoms.generate import EscapeDomain, RecipeDomain, generate
from tests.regression.test_phase2b_fractal import fixture_a_grid, fixture_b_grid


def test_domains_match_legacy_grids():
    np.testing.assert_array_equal(EscapeDomain().grid().coordinates(), fixture_a_grid().coordinates())
    np.testing.assert_array_equal(RecipeDomain(slices=10).grid().coordinates()[:, :2],
                                  fixture_b_grid().coordinates())


def test_generate_modes_small(tmp_path):
    cands = np.array([[-0.02, 0.38, -0.53, -0.47], [0.1, 0.6, -0.2, 0.1]])
    p3 = generate("recipe_3dt", 1, tmp_path / "a", domain=RecipeDomain(shape=(16, 16), frames=3, slices=2),
                  norm="volume", candidates=cands)
    v = np.load(p3[0])
    assert v.shape == (3, 2, 16, 16) and np.iscomplexobj(v)
    assert np.isclose(v.real.min(), -1) and np.isclose(v.real.max(), 1, atol=1e-6)
    p2 = generate("recipe_2dt", 1, tmp_path / "b", domain=RecipeDomain(shape=(16, 16), frames=3, slices=2),
                  candidates=cands)
    assert [p.name for p in p2] == ["julia_00_slice_1.npy", "julia_00_slice_2.npy"]
    pe = generate("escape", 1, tmp_path / "c", domain=EscapeDomain(shape=(64, 64), frames=4), candidates=cands[:1])
    assert np.load(pe[0]).dtype == np.int32 and (tmp_path / "c").glob("*.json")
