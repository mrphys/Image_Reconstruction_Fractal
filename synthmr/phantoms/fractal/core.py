
"""Core quaternion-Julia rendering.

The notebook uses quaternion coordinates in ``(w, x, y, z)`` order in its
later implementations. This module standardises that convention throughout.

A rendered image is sampled on a 2-D plane in quaternion space. The two image
axes are mapped to any two quaternion components, while the remaining
components are fixed. A third "phase" parameter can be used to generate a
stack of related images.
"""

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple, Optional, Union
import numpy as np

from .fields import evaluate_field


@dataclass(frozen=True)
class Slice:
    """Describe a 2-D slice through quaternion space.

    Parameters
    ----------
    axes:
        Two component indices in ``(w, x, y, z)`` used for image coordinates.
    fixed:
        Four fixed quaternion coordinates. Entries corresponding to ``axes``
        are ignored and replaced by the image coordinates.
    bounds:
        ``((min_axis0, max_axis0), (min_axis1, max_axis1))``.
    """

    axes: Tuple[int, int] = (1, 2)
    fixed: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0))

    def coordinates(self, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        if self.axes[0] == self.axes[1] or any(a < 0 or a > 3 for a in self.axes):
            raise ValueError("axes must contain two distinct indices in [0, 3].")
        q = np.empty((h, w, 4), dtype=np.float64)
        q[...] = np.asarray(self.fixed, dtype=np.float64)
        a = np.linspace(*self.bounds[0], h)
        b = np.linspace(*self.bounds[1], w)
        q[..., self.axes[0]] = a[:, None]
        q[..., self.axes[1]] = b[None, :]
        return q


@dataclass(frozen=True)
class JuliaConfig:
    """Parameters shared by quaternion Julia fields."""

    c: Tuple[float, float, float, float] = (-0.11, 0.83, -0.11, 0.32)
    max_iter: int = 50
    bailout: float = 4.0
    power: int = 2
    epsilon: float = 0.01
    trap: str = "plane"
    trap_axes: Tuple[int, ...] = (0,)
    derivative_floor: float = 1e-12


@dataclass(frozen=True)
class FieldSpec:
    """A named scalar field to render.

    ``kind`` is one of the registered field names, e.g. ``"smooth_escape"``,
    ``"distance"``, ``"orbit_trap"``, ``"power"`` or ``"perturbed"``.
    """

    name: str
    kind: str
    params: dict = field(default_factory=dict)


class QuaternionJulia:
    """Render scalar fields from a quaternion Julia iteration."""

    def __init__(self, config: Optional[JuliaConfig] = None):
        self.config = config or JuliaConfig()

    def render(
        self,
        shape: Tuple[int, int] = (256, 256),
        slice_: Optional[Slice] = None,
        field: str = "smooth_escape",
        **field_params,
    ) -> np.ndarray:
        """Render one scalar field."""
        slice_ = slice_ or Slice()
        coords = slice_.coordinates(shape)
        params = {**field_params, **{"config": self.config}}
        return evaluate_field(coords, field, params)

    def render_stack(
        self,
        fields: Sequence[Union[str,FieldSpec]],
        shape: Tuple[int, int] = (256, 256),
        slice_: Optional[Slice] = None,
    ) -> np.ndarray:
        """Render several fields and return ``(n_fields, H, W)``."""
        slice_ = slice_ or Slice()
        coords = slice_.coordinates(shape)
        out = []
        for spec in fields:
            if isinstance(spec, str):
                spec = FieldSpec(spec, spec)
            params = {**spec.params, "config": self.config}
            out.append(evaluate_field(coords, spec.kind, params))
        return np.stack(out, axis=0)


def render_fields(
    config: JuliaConfig,
    fields: Iterable[Union[FieldSpec,str]],
    shape=(256, 256),
    slice_: Optional[Slice] =  None,
) -> np.ndarray:
    """Convenience wrapper returning ``(channels, H, W)``."""
    return QuaternionJulia(config).render_stack(fields, shape, slice_)
