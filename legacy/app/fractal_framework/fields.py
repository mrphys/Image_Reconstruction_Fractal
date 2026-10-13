
"""Scalar fields derived from quaternion Julia iteration.

The notebook experimented with several definitions of a "fractal value".
They are deliberately exposed as independent fields so they can be mixed
later into real/imaginary channels.
"""


from typing import Callable, Dict
import numpy as np

from . import _kernels as _k

FIELD_REGISTRY: Dict[str, Callable] = {}


def register_field(name: str):
    """Decorator for adding a custom scalar field to the registry."""
    def decorator(fn):
        FIELD_REGISTRY[name] = fn
        return fn
    return decorator


def _c(config):
    return np.asarray(config.c, dtype=np.float64)


def _escape(flat, config, power=None, perturb=False, epsilon=None,
            smooth=False, interior=False):
    p = config.power if power is None else power
    eps = config.epsilon if epsilon is None else epsilon
    return _k.escape_grid(flat, _c(config), config.max_iter, config.bailout,
                          int(p), perturb, float(eps), smooth, interior)


def _trap_args(config, trap, axes):
    trap = trap or config.trap
    axes = tuple(config.trap_axes if axes is None else axes)
    if trap == "point":
        code = _k.TRAP_POINT
    elif trap == "line":
        code = _k.TRAP_LINE
        axes = axes or (1, 2, 3)
    elif trap == "plane":
        code = _k.TRAP_PLANE
        axes = axes or (0,)
    else:
        raise ValueError(f"Unknown trap {trap!r}")
    return code, np.asarray(axes or (0,), dtype=np.int64)


# Grid implementations of the built-ins: ``grid(flat (N, 4), config, **params) -> (N,)``.
_GRIDS = {
    "escape": lambda flat, cfg: _escape(flat, cfg),
    "smooth_escape": lambda flat, cfg: _escape(flat, cfg, smooth=True),
    "distance": lambda flat, cfg: _k.distance_grid(
        flat, _c(cfg), cfg.max_iter, cfg.bailout, cfg.derivative_floor),
    "orbit_trap": lambda flat, cfg, trap=None, axes=None: _k.orbit_trap_grid(
        flat, _c(cfg), cfg.max_iter, cfg.bailout, *_trap_args(cfg, trap, axes)),
    "power": lambda flat, cfg, power=None: _escape(flat, cfg, power=power),
    "perturbed": lambda flat, cfg, epsilon=None: _escape(
        flat, cfg, perturb=True, epsilon=epsilon),
    "internal_detail": lambda flat, cfg: _escape(flat, cfg, smooth=True, interior=True),
}


def _point(kind, q0, config, **params):
    flat = np.ascontiguousarray(q0, dtype=np.float64).reshape(1, 4)
    return float(_GRIDS[kind](flat, config, **params)[0])


@register_field("escape")
def escape(q0, config):
    """Integer escape iteration count."""
    return _point("escape", q0, config)


@register_field("smooth_escape")
def smooth_escape(q0, config):
    """Continuous/smoothed escape iteration count."""
    return _point("smooth_escape", q0, config)


@register_field("distance")
def distance(q0, config):
    """Distance-estimator field following the notebook's derivative form."""
    return _point("distance", q0, config)


@register_field("orbit_trap")
def orbit_trap(q0, config, trap=None, axes=None):
    """Minimum distance to a point, line, or plane during the orbit."""
    return _point("orbit_trap", q0, config, trap=trap, axes=axes)


@register_field("power")
def power(q0, config, power=None):
    """Julia iteration with integer power ``z -> z**power + c``."""
    return _point("power", q0, config, power=power)


@register_field("perturbed")
def perturbed(q0, config, epsilon=None):
    """Julia iteration with an element-wise ``epsilon * sin(z)`` perturbation."""
    return _point("perturbed", q0, config, epsilon=epsilon)


@register_field("internal_detail")
def internal_detail(q0, config):
    """Smooth escape outside the set and the notebook's internal-detail term inside."""
    return _point("internal_detail", q0, config)


FIELD_REGISTRY_BUILTINS = dict(FIELD_REGISTRY)


def evaluate_field(coords: np.ndarray, kind: str, params: dict) -> np.ndarray:
    """Evaluate a registered field over ``(..., 4)`` quaternion coordinates."""
    if kind not in FIELD_REGISTRY:
        raise KeyError(f"Unknown field {kind!r}. Available: {sorted(FIELD_REGISTRY)}")
    fn = FIELD_REGISTRY[kind]
    cfg = params.get("config")
    if cfg is None:
        from .core import JuliaConfig
        cfg = JuliaConfig()
    extra = {k: v for k, v in params.items() if k != "config"}
    flat = np.ascontiguousarray(coords.reshape(-1, 4), dtype=np.float64)
    if kind in _GRIDS and fn is FIELD_REGISTRY_BUILTINS.get(kind):
        # Built-in field: evaluate the whole grid in one compiled, parallel call.
        values = _GRIDS[kind](flat, cfg, **extra).astype(np.float32)
        return values.reshape(coords.shape[:-1])
    # Custom Python field: per-point loop.
    values = np.fromiter((fn(q, cfg, **extra) for q in flat), dtype=np.float32, count=len(flat))
    return values.reshape(coords.shape[:-1])
