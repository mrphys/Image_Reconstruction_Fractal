"""N-dimensional sampling grids in quaternion space.

A :class:`Grid` maps each output array dimension to one quaternion component
``(w, x, y, z) = (0, 1, 2, 3)`` and a 1-D array of coordinate values; the remaining
components are held at ``fixed``. ``Grid.coordinates()`` returns ``(*shape, 4)``
which any registered field evaluates in a single (numba-parallel) call.

This generalises :class:`~synthmr.phantoms.fractal.core.Slice` (2-D) to e.g.
``(T, Z, Y, X)`` blocks for 3D+time phantoms.
"""

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Axis:
    """One output dimension: ``name``, quaternion ``component`` and its coordinate ``values``."""

    name: str
    component: int
    values: Tuple[float, ...]

    @classmethod
    def linspace(cls, name, component, low, high, n):
        return cls(name, component, tuple(np.linspace(low, high, n).tolist()))

    @classmethod
    def centred(cls, name, component, n, step, centre=0.0):
        """``n`` values spaced by ``step`` and centred on ``centre``."""
        return cls(name, component, tuple(((np.arange(n) - (n - 1) / 2) * step + centre).tolist()))


@dataclass(frozen=True)
class Grid:
    """Rectilinear grid over any subset of quaternion components."""

    axes: Sequence[Axis]
    fixed: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    _check: bool = field(default=True, repr=False)

    def __post_init__(self):
        comps = [a.component for a in self.axes]
        if len(set(comps)) != len(comps) or any(c < 0 or c > 3 for c in comps):
            raise ValueError(f"axes must use distinct components in [0, 3], got {comps}")

    @property
    def shape(self):
        return tuple(len(a.values) for a in self.axes)

    @property
    def names(self):
        return tuple(a.name for a in self.axes)

    def coordinates(self):
        q = np.empty((*self.shape, 4), dtype=np.float64)
        q[...] = np.asarray(self.fixed, dtype=np.float64)
        for d, a in enumerate(self.axes):
            idx = [None] * len(self.axes)
            idx[d] = slice(None)
            q[..., a.component] = np.asarray(a.values, dtype=np.float64)[tuple(idx)]
        return q
