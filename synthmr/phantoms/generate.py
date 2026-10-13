"""Fractal phantom generation (replaces ``julia_generator_new_grasp.py`` and the notebook loop).

Modes
-----
``escape``      base-repo escape-time volume ``(T, H, W)`` int32, entropy-filtered (paper method;
                complexified later by the pipeline's ``sine_wave`` step).
``recipe_2dt``  new framework, complex ``(T, H, W)`` per slice, saved one file per slice
                (``julia_XX_slice_N.npy``) exactly like the legacy generator.
``recipe_3dt``  new framework, complex ``(T, Z, H, W)`` volume, one file per fractal.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .fractal import Axis, Grid, JuliaConfig, candidate_c, evaluate_field
from .fractal import legacy as fl
from .recipes import random_recipe, render_recipe


@dataclass
class RecipeDomain:
    """Quaternion domain of the new generator: rows = x (comp 1), cols = w (comp 0),
    time = z sweep (comp 3), slices = y (comp 2) centred on 0 with ``slice_step`` spacing."""

    shape: Tuple[int, int] = (256, 256)
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.2, 1.2), (-1.6, 1.6))
    time_range: Tuple[float, float] = (-0.3, 0.3)
    frames: int = 32
    slices: int = 10
    slice_step: float = 0.1
    max_iter: int = 30

    def grid(self):
        (r0, r1), (c0, c1) = self.bounds
        return Grid(axes=[Axis.linspace("t", 3, *self.time_range, self.frames),
                          Axis.centred("z", 2, self.slices, self.slice_step),
                          Axis.linspace("y", 1, r0, r1, self.shape[0]),
                          Axis.linspace("x", 0, c0, c1, self.shape[1])])


@dataclass
class EscapeDomain:
    """Base-repo grid ``build_grid([1, 1, 0, 0.2], [256, 256, 1, 32])`` as ``(T, Y, X)``."""

    shape: Tuple[int, int] = (256, 256)
    extent: Tuple[float, float] = (1.0, 1.0)
    time_extent: float = 0.2
    frames: int = 32
    max_iter: int = 50
    complexity_threshold: float = 1.7

    def grid(self):
        return Grid(axes=[Axis.linspace("t", 3, -self.time_extent, self.time_extent, self.frames),
                          Axis.linspace("y", 0, -self.extent[0], self.extent[0], self.shape[0]),
                          Axis.linspace("x", 1, -self.extent[1], self.extent[1], self.shape[1])])


def default_candidates():
    return candidate_c(resolution=40, extent=1.5, max_iter=50, min_iter=10, max_escape=30)


def generate(mode, n, out_dir, seed=0, domain=None, norm="frame", candidates=None, max_tries=100):
    """Generate ``n`` phantoms into ``out_dir``; returns the list of written paths.

    ``norm``: ``frame`` (legacy, per 2-D frame) or ``volume`` (whole block; recommended for 3D+t).
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = default_candidates() if candidates is None else candidates
    norm_axes = (-2, -1) if norm == "frame" else None
    written = []
    if mode == "escape":
        domain = domain or EscapeDomain()
        coords = domain.grid().coordinates()
        made, tries = 0, 0
        while made < n and tries < n * max_tries:
            tries += 1
            c = candidates[rng.integers(len(candidates))]
            esc = evaluate_field(coords, "escape", {"config": JuliaConfig(c=tuple(c), max_iter=domain.max_iter)})
            esc = esc.astype(np.int32)
            if not fl.test_complexity(esc, domain.complexity_threshold):
                continue
            path = out_dir / ("_".join(f"{v:.2f}" for v in c) + f"_{made:04d}.npy")
            np.save(path, esc)
            _write_meta(path, {"mode": mode, "c": c.tolist(), "domain": asdict(domain)})
            written.append(path)
            made += 1
        return written

    domain = domain or RecipeDomain()
    grid = domain.grid()
    for k in range(n):
        recipe = random_recipe(rng, candidates[rng.integers(len(candidates))])
        block = render_recipe(recipe, grid, max_iter=domain.max_iter, norm_axes=norm_axes)   # (T, Z, H, W)
        meta = {"mode": mode, "recipe": recipe, "domain": asdict(domain), "norm": norm}
        if mode == "recipe_2dt":
            for s in range(block.shape[1]):
                path = out_dir / f"julia_{k:02d}_slice_{s + 1}.npy"
                np.save(path, block[:, s])
                _write_meta(path, {**meta, "slice": s})
                written.append(path)
        elif mode == "recipe_3dt":
            path = out_dir / f"julia_{k:03d}.npy"
            np.save(path, block)
            _write_meta(path, meta)
            written.append(path)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return written


def _write_meta(path, meta):
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=lambda o: getattr(o, "tolist", str)(o)))
