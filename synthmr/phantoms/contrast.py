"""Tissue-specific contrast dynamics for fractal phantoms (DCE-like signal changes over time).

The phantom is split into ``K`` "tissues" from one fractal field rendered on the same (framed, warped) coordinates
as the image, so a tissue follows its structure through time and across slices:

* tissue 0 is the interior of the set (``escape == max_iter``);
* tissues ``1..K-1`` are quantile bands of ``smooth_escape`` over the exterior (nested shells around the set).

Each tissue gets a wash-in / wash-out curve ``e_k(t)`` and the image is scaled by ``s(x, t) = sum_k m_k(x) (1 + e_k(t))``
where ``m_k`` is the one-hot label map blurred by ``softness`` in-plane pixels (soft partial-volume boundaries; the
memberships still sum to 1). Scaling is real and non-negative, so it changes magnitude only and keeps the phase.

The draw is stored in ``recipe["contrast"]`` (quantile levels, not values), so
:func:`~synthmr.phantoms.recipes.render_recipe` re-renders it from the metadata JSON.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .fractal import evaluate_field

# Curve kinds: probability and samplers for (onset, tau_in, tau_out, peak scale), in units of the time sweep [0, 1].
# ``tau_out=None`` is a persistent curve (no wash-out).
CURVES = {
    "static": lambda rng: (0.0, 1.0, None, 0.0),
    "persistent": lambda rng: (rng.uniform(0.05, 0.4), rng.uniform(0.3, 0.8), None, 1.0),
    "plateau": lambda rng: (rng.uniform(0.05, 0.3), rng.uniform(0.04, 0.12), None, 1.0),
    "washout": lambda rng: (rng.uniform(0.05, 0.3), rng.uniform(0.03, 0.1), rng.uniform(0.3, 1.0), 1.0),
    "arterial": lambda rng: (rng.uniform(0.02, 0.15), rng.uniform(0.01, 0.04), rng.uniform(0.05, 0.15), 1.5),
}


@dataclass
class Contrast:
    tissues: Tuple[int, int] = (3, 5)            # number of tissues (inclusive range, interior included)
    softness: float = 1.5                        # label blur, in-plane pixels
    peak: Tuple[float, float] = (0.3, 2.0)       # relative enhancement at the curve maximum (times the kind's scale)
    kinds: Dict[str, float] = field(default_factory=lambda: {
        "static": 0.2, "persistent": 0.2, "plateau": 0.2, "washout": 0.25, "arterial": 0.15})


def draw_contrast(rng, spec):
    """Random contrast dict for a recipe: tissue quantile levels, softness and one curve per tissue."""
    k = int(rng.integers(spec.tissues[0], spec.tissues[1] + 1))
    names, p = list(spec.kinds), np.asarray(list(spec.kinds.values()), dtype=np.float64)
    curves = []
    for kind in rng.choice(names, size=k, p=p / p.sum()):
        onset, tau_in, tau_out, scale = CURVES[str(kind)](rng)
        curves.append({"kind": str(kind), "onset": float(onset), "tau_in": float(tau_in),
                       "tau_out": None if tau_out is None else float(tau_out),
                       "peak": float(scale * rng.uniform(*spec.peak))})
    levels = np.sort(rng.uniform(0.1, 0.9, size=k - 2)) if k > 2 else np.zeros(0)
    return {"field": "smooth_escape", "levels": levels.tolist(), "softness": float(spec.softness), "curves": curves}


def curve(c, u):
    """Enhancement ``e(u)`` of one curve dict at normalised times ``u`` in [0, 1]; its maximum over ``u >= onset`` is
    ``peak`` (or approaches it, for persistent curves)."""
    tau = np.maximum(np.asarray(u, dtype=np.float64) - c["onset"], 0.0)
    rise = 1 - np.exp(-tau / c["tau_in"])
    if c["tau_out"] is None:
        return c["peak"] * rise
    t_in, t_out = c["tau_in"], c["tau_out"]
    t_max = t_in * np.log1p(t_out / t_in)                    # argmax of (1 - e^{-t/t_in}) e^{-t/t_out}
    g_max = (1 - np.exp(-t_max / t_in)) * np.exp(-t_max / t_out)
    return c["peak"] * rise * np.exp(-tau / t_out) / g_max


def tissue_labels(coords, contrast, config):
    """Integer tissue map of ``coords.shape[:-1]``: 0 = interior, then exterior ``smooth_escape`` quantile bands."""
    esc = evaluate_field(coords, contrast["field"], {"config": config})
    interior = evaluate_field(coords, "escape", {"config": config}) >= config.max_iter
    labels = np.zeros(esc.shape, dtype=np.int32)
    ext = esc[~interior]
    if ext.size:
        edges = np.quantile(ext, contrast["levels"]) if contrast["levels"] else np.zeros(0)
        labels[~interior] = 1 + np.searchsorted(edges, ext, side="right")
    return labels


def memberships(labels, n, grid, softness):
    """Soft memberships ``(n, *labels.shape)`` summing to 1: one-hot labels blurred by ``softness`` in-plane pixels
    (scaled by each axis' coordinate step, so thick slices are barely blurred across; time is never blurred)."""
    steps = np.array([abs(a.values[1] - a.values[0]) if len(a.values) > 1 else np.inf for a in grid.axes])
    sigma = [0.0 if a.name == "t" else softness * steps[-1] / s for a, s in zip(grid.axes, steps)]
    m = np.stack([gaussian_filter((labels == k).astype(np.float32), sigma, mode="nearest") for k in range(n)])
    return m / np.maximum(m.sum(0), 1e-8)


def time_axis(grid):
    """Normalised time ``u`` broadcastable against ``grid.shape`` (zeros when the grid has no ``t`` axis)."""
    u = np.zeros([1] * len(grid.axes))
    if "t" in grid.names:
        d = grid.names.index("t")
        n = grid.shape[d]
        u = np.linspace(0, 1, n).reshape([n if i == d else 1 for i in range(len(grid.axes))])
    return u


def signal_scale(coords, grid, contrast, config):
    """``s(x, t) = sum_k m_k(x) (1 + e_k(t))`` over ``grid.shape``."""
    curves = contrast["curves"]
    m = memberships(tissue_labels(coords, contrast, config), len(curves), grid, contrast["softness"])
    u = time_axis(grid)
    return sum(m[k] * (1 + curve(c, u)) for k, c in enumerate(curves)).astype(np.float32)


def apply_contrast(block, scale):
    """Scale a recipe block (complex, re/im in [-1, 1]) by ``scale`` and renormalise over the whole block.

    The pipeline maps re/im linearly to [0, 255] (``complex_to_video``) and the simulation builds its image from
    those channels, so the scaling acts on the channels shifted to [0, 1] (where 0 is black). One normalisation over
    the whole block keeps the enhancement relative across time.
    """
    pos = (block + (1 + 1j)) / 2 * scale
    pos /= max(float(pos.real.max()), float(pos.imag.max()), 1e-8)
    return (2 * pos - (1 + 1j)).astype(np.complex128)
