import numpy as np
import pytest

from synthmr.phantoms.fractal import Axis, Grid, Slice
from synthmr.phantoms.fractal.channels import normalise


def test_grid_matches_slice_coordinates():
    s = Slice(axes=(1, 0), fixed=(0.0, 0.0, 0.3, -0.1), bounds=((-1.2, 1.2), (-1.6, 1.6)))
    g = Grid(axes=[Axis.linspace("y", 1, -1.2, 1.2, 20), Axis.linspace("x", 0, -1.6, 1.6, 30)],
             fixed=(0.0, 0.0, 0.3, -0.1))
    np.testing.assert_array_equal(g.coordinates(), s.coordinates((20, 30)))


def test_grid_4d_shape_and_axis_order():
    g = Grid(axes=[Axis.centred("t", 3, 4, 0.1), Axis.centred("z", 2, 3, 0.2),
                   Axis.linspace("y", 1, -1, 1, 5), Axis.linspace("x", 0, -1, 1, 6)])
    q = g.coordinates()
    assert q.shape == (4, 3, 5, 6, 4) and g.names == ("t", "z", "y", "x")
    np.testing.assert_allclose(q[:, 0, 0, 0, 3], [-0.15, -0.05, 0.05, 0.15])
    np.testing.assert_allclose(q[0, :, 0, 0, 2], [-0.2, 0.0, 0.2])


def test_grid_rejects_duplicate_components():
    with pytest.raises(ValueError):
        Grid(axes=[Axis.linspace("a", 1, 0, 1, 2), Axis.linspace("b", 1, 0, 1, 2)])


def test_normalise_axes_matches_scalar_per_frame():
    x = np.random.default_rng(0).standard_normal((5, 7, 9)).astype(np.float32) * 3 + 1
    per_frame = np.stack([normalise(f) for f in x])
    np.testing.assert_array_equal(normalise(x, axes=(-2, -1)), per_frame)
