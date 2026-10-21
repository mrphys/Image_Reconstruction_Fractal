"""Per-phantom variety for the recipe generator: how a recipe's Julia set is framed and sampled.

Steps (all off in ``Variety()``, so the plain recipe protocol and its fixtures are unchanged):

``frame``       zoom into a random window (scale drawn from ``zoom``) centred on a boundary point of the set. The
                scale applies to all four components, so slice spacing and the time sweep shrink with the in-plane
                field of view (voxel aspect and motion relative to the FOV are kept).
``reject``      reframe (up to ``tries``) and then redraw ``c`` (up to ``redraws``) until the boundary band
                ``2 < escape < max_iter`` covers at least ``min_detail`` of a coarse preview.
``background``  every channel gets a field that varies outside the set (``distance`` / ``orbit_trap``), and every
                field is gamma-balanced so its median maps to 0.5 (escape-time fields are heavily skewed).
``warp``        smooth fBm domain warp of the coordinates; strength and noise scale are relative to the window.
``permute``     random assignment of quaternion components to the array axes (which one is time, slices, ...).
``rotate``      Haar-random 4-D rotation of the sampling grid (oblique sections; subsumes ``permute``).

A framing is stored in the recipe (``recipe["framing"]``) so :func:`~synthmr.phantoms.recipes.render_recipe`
re-renders it from the metadata JSON.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import zoom as _resize

from .fractal import Axis, Grid, JuliaConfig, domain_warp, evaluate_field

STEPS = ("frame", "reject", "background", "warp", "permute", "rotate")


@dataclass
class Variety:
    zoom: Optional[Tuple[float, float]] = None   # window scale range relative to the domain (frame)
    min_detail: Optional[float] = None           # min boundary-band fraction of the preview (reject)
    tries: int = 8                               # framings per recipe before ``c`` is redrawn
    redraws: int = 10                            # recipe redraws before the best framing is accepted
    background: bool = False
    warp: Optional[float] = None                 # warp strength relative to the window half-width
    permute: bool = False
    rotate: bool = False
    preview: int = 16                            # preview points per axis

    @classmethod
    def from_steps(cls, steps):
        """``"frame,reject,..."``, a sequence of step names, ``"all"`` or ``"none"``/``None``."""
        if not steps or steps == "none":
            return cls()
        steps = STEPS if steps == "all" else (steps.split(",") if isinstance(steps, str) else steps)
        unknown = set(steps) - set(STEPS)
        if unknown:
            raise ValueError(f"unknown variety steps {sorted(unknown)}; available: {STEPS}")
        return cls(zoom=(0.4, 0.75) if "frame" in steps else None,
                   min_detail=0.6 if "reject" in steps else None,
                   background="background" in steps, warp=0.15 if "warp" in steps else None,
                   permute="permute" in steps, rotate="rotate" in steps)

    @property
    def active(self):
        return any((self.zoom, self.min_detail is not None, self.background, self.warp, self.permute, self.rotate))


def random_rotation(rng):
    """Haar-random ``SO(4)`` matrix."""
    q, r = np.linalg.qr(rng.standard_normal((4, 4)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def apply_framing(coords, framing):
    """``(..., 4)`` grid coordinates -> framed coordinates ``centre + (q - origin) @ matrix.T`` (then warped)."""
    m = np.asarray(framing["matrix"], dtype=np.float64)
    out = (coords - np.asarray(framing["origin"])) @ m.T + np.asarray(framing["centre"])
    w = framing.get("warp")
    if w:
        out = _warp(out, w)
    return out


def _warp(coords, w, samples=2):
    """``domain_warp`` of a grid's coordinates. Large grids get the (smooth) displacement from a coarser grid with
    ``samples`` points per finest noise cell, linearly interpolated (the unwarped coordinates are affine in the
    index, so the coarse grid is exact)."""
    def warp(q):
        return domain_warp(q, strength=w["strength"], scale=w["scale"], octaves=w["octaves"], seed=w["seed"])

    shape = coords.shape[:-1]
    if len(shape) < 2 or np.prod(shape) < 2 ** 20:
        return warp(coords)
    cell = 1.0 / (w["scale"] * 2.0 ** (w["octaves"] - 1))     # finest octave's cell length (lacunarity 2)
    small = []
    origin = (0,) * len(shape)
    for d, n in enumerate(shape):
        nxt = origin[:d] + (min(1, n - 1),) + origin[d + 1:]
        step = np.linalg.norm(coords[nxt] - coords[origin])                   # coordinate step along axis d
        stride = max(1, int(cell / (samples * step))) if step > 0 else n
        small.append(min(n, (n - 1) // stride + 2))
    def resize(x, out):
        return np.stack([_resize(x[..., i], [o / n for o, n in zip(out, x.shape[:-1])], order=1, mode="nearest")
                         for i in range(4)], -1)

    sub = resize(coords, small)
    return coords + resize(warp(sub) - sub, shape)


def _preview_grid(grid, n):
    axes = [Axis(a.name, a.component, tuple(np.asarray(a.values)[np.unique(np.linspace(0, len(a.values) - 1,
                                                                                         n).round().astype(int))]))
            for a in grid.axes]
    return Grid(axes=axes, fixed=grid.fixed).coordinates()


def _band(coords, config):
    esc = evaluate_field(coords, "escape", {"config": config})
    return (esc > 2) & (esc < config.max_iter)


def draw_framing(grid, c, max_iter, variety, rngs):
    """Draw up to ``variety.tries`` framings for Julia parameter ``c``; returns ``(framing, detail)`` of the first
    reaching ``min_detail`` (or the best one). ``rngs`` holds one generator per step (``orient``, ``frame``,
    ``warp``), so enabling a step leaves the other steps' draws unchanged."""
    config = JuliaConfig(c=tuple(c), max_iter=max_iter)
    pre = _preview_grid(grid, variety.preview)
    flat = pre.reshape(-1, 4)
    origin = (flat.min(0) + flat.max(0)) / 2
    half = float(max((a.values[-1] - a.values[0]) / 2 for a in grid.axes))   # widest in-window half extent
    best = None
    for _ in range(variety.tries if variety.min_detail is not None else 1):
        base = np.eye(4)
        if variety.permute:
            base = base[rngs["orient"].permutation(4)]
        if variety.rotate:
            base = random_rotation(rngs["orient"]) @ base
        framing = {"matrix": base, "origin": origin, "centre": origin, "zoom": 1.0}
        if variety.zoom:
            full = apply_framing(pre, framing)
            pts = full[_band(full, config)]
            framing["centre"] = pts[rngs["frame"].integers(len(pts))] if len(pts) else origin
            framing["zoom"] = float(rngs["frame"].uniform(*variety.zoom))
            framing["matrix"] = framing["zoom"] * base
        if variety.warp:
            window = framing["zoom"] * half
            framing["warp"] = {"strength": variety.warp * window, "scale": 1.0 / window, "octaves": 4,
                               "seed": int(rngs["warp"].integers(2 ** 31))}
        detail = float(_band(apply_framing(pre, framing), config).mean())
        framing = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in framing.items()}
        if best is None or detail > best[1]:
            best = (framing, detail)
        if variety.min_detail is None or detail >= variety.min_detail:
            break
    return best


def enrich_channel(channel, rng):
    """Add a field that varies outside the set (unless one is present) and gamma-balance every field."""
    fields = [dict(f, balance=True) for f in channel["fields"]]
    weights = list(channel["weights"])
    if not any(f["kind"] in ("distance", "orbit_trap") for f in fields):
        if rng.random() < 0.5:
            fields.append({"kind": "distance", "params": {}, "balance": True})
        else:
            fields.append({"kind": "orbit_trap", "params": {"trap": "point", "axes": (0,)}, "balance": True})
        weights.append(float(rng.uniform(0.2, 0.5)) * sum(weights))
    total = sum(weights)
    return {"fields": fields, "weights": [w / total for w in weights]}


def vary_recipe(recipe, grid, max_iter, variety, seed, redraw):
    """Recipe with ``framing`` (and ``background`` enrichment) per ``variety``. ``seed`` (a ``SeedSequence``) is
    split into one stream per step; ``redraw(rng)`` gives a new recipe when no framing of the current ``c`` reaches
    ``min_detail``."""
    rngs = dict(zip(("orient", "frame", "warp", "background", "redraw"), map(np.random.default_rng, seed.spawn(5))))
    best = None
    for attempt in range(variety.redraws + 1 if variety.min_detail is not None else 1):
        framing, detail = draw_framing(grid, recipe["c"], max_iter, variety, rngs)
        if best is None or detail > best[2]:
            best = (recipe, framing, detail)
        if variety.min_detail is None or detail >= variety.min_detail:
            break
        recipe = redraw(rngs["redraw"])
    recipe, framing, detail = best
    out = dict(recipe, framing={**framing, "detail": detail, "redraws": attempt})
    if variety.background:
        out["real"] = enrich_channel(recipe["real"], rngs["background"])
        out["imag"] = enrich_channel(recipe["imag"], rngs["background"])
    return out
