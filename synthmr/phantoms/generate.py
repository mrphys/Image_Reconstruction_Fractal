"""Fractal phantom generation (replaces ``julia_generator_new_grasp.py`` and the notebook loop).

Modes
-----
``legacy``  base-repo escape-time volume ``(T, H, W)`` int32, entropy-filtered (paper method;
            complexified later by the pipeline's ``sine_wave`` step).
``2dt``     new framework, complex ``(T, H, W)`` per slice, saved one file per slice
            (``julia_XX_slice_N.npy``) exactly like the legacy generator.
``3dt``     new framework, complex ``(T, Z, H, W)`` volume, one file per fractal.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .fractal import Axis, Grid, JuliaConfig, candidate_c, evaluate_field
from .fractal import legacy as fl
from .recipes import random_recipe, render_recipe
from .variety import Variety, vary_recipe


@dataclass
class FractalDomain:
    """Quaternion domain of the new generator: rows = x (comp 1), cols = w (comp 0),
    time = z sweep (comp 3), slices = y (comp 2) centred on 0 with ``slice_step`` spacing."""

    shape: Tuple[int, int] = (256, 256)
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.2, 1.2), (-1.6, 1.6))
    time_range: Tuple[float, float] = (-0.3, 0.3)
    frames: int = 32
    slices: int = 10
    slice_step: float = 0.1
    max_iter: int = 30

    @classmethod
    def isotropic(cls, n=64, extent=2.4, frames=32, time_range=(-0.3, 0.3), max_iter=30):
        """``n^3`` domain with voxel ``v = extent / n`` in all three spatial components (cone pairs)."""
        v = extent / n
        half = (n - 1) / 2 * v
        return cls(shape=(n, n), slices=n, slice_step=v, bounds=((-half, half),) * 2, frames=frames,
                   time_range=time_range, max_iter=max_iter)

    def grid(self):
        (r0, r1), (c0, c1) = self.bounds
        return Grid(axes=[Axis.linspace("t", 3, *self.time_range, self.frames),
                          Axis.centred("z", 2, self.slices, self.slice_step),
                          Axis.linspace("y", 1, r0, r1, self.shape[0]),
                          Axis.linspace("x", 0, c0, c1, self.shape[1])])


@dataclass
class LegacyDomain:
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
    """
    Default candidates for ``c`` in the quaternion Julia set generation, sampled near the Mandelbrot boundary.
    """
    return candidate_c(resolution=40, extent=1.5, max_iter=50, min_iter=10, max_escape=30)


def generate(mode, n, out_dir, seed=0, domain=None, norm="frame", candidates=None, max_tries=100, variety=None):
    """
    Generate ``n`` phantoms into ``out_dir``; returns the list of written paths.

    Args:
        mode: str, one of ``legacy``, ``2dt``, ``3dt``. ``legacy`` uses the base-repo escape-time generator, while ``2dt``/``3dt`` use the new recipe-based generator.
        n: int, number of phantoms to generate.
        out_dir: str or Path, output directory.
        seed: int, random seed for reproducibility.
        domain: FractalDomain or LegacyDomain, optional; if None, defaults are used.
        norm: str, normalization method; "frame" or "volume".
        candidates: list of complex numbers for Julia parameter c; if None, defaults are used.
        max_tries: int, maximum attempts to generate valid escape-time volumes in "legacy" mode.
        variety: Variety, optional; framing / rejection / warp / rotation steps for ``2dt``/``3dt`` (default: none).
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = default_candidates() if candidates is None else candidates # candidates based on mandelbrot
    written = []
    if mode == "legacy":
        domain = domain or LegacyDomain() # base-repo escape-time domain
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

    domain = domain or FractalDomain()
    for k, recipe, block in iter_recipes_3dt(domain, n, rng, norm, candidates, variety):
        meta = {"mode": mode, "recipe": recipe, "domain": asdict(domain), "norm": norm}
        if mode == "2dt":
            for s in range(block.shape[1]):
                path = out_dir / f"julia_{k:02d}_slice_{s + 1}.npy"
                np.save(path, block[:, s])
                _write_meta(path, {**meta, "slice": s})
                written.append(path)
        elif mode == "3dt":
            path = out_dir / f"julia_{k:03d}.npy"
            np.save(path, block)
            _write_meta(path, meta)
            written.append(path)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return written


def iter_recipes_3dt(domain, n, seed=0, norm="frame", candidates=None, variety=None):
    """
    Draw and render ``n`` random recipes on ``domain``; yields ``(k, recipe, block)`` with block ``(T, Z, H, W)``.

    Args:
        domain: FractalDomain.
        n: int, number of recipes.
        seed: int or ``np.random.Generator`` (shared with the caller, as in :func:`generate`).
        norm: str, "frame" or "volume".
        candidates: list of complex numbers for Julia parameter c; if None, defaults are used.
        variety: Variety, optional. Its draws come from children of the generator's seed sequence, so enabling a
            step leaves the recipes unchanged (except those that ``reject`` redraws, from the variety streams).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    candidates = default_candidates() if candidates is None else candidates
    norm_axes = (-2, -1) if norm == "frame" else None
    grid = domain.grid()
    variety = variety or Variety()
    children = rng.bit_generator._seed_seq.spawn(n) if variety.active else None

    def draw(r):
        return random_recipe(r, candidates[r.integers(len(candidates))])

    for k in range(n):
        recipe = draw(rng)
        if variety.active:
            recipe = vary_recipe(recipe, grid, domain.max_iter, variety, children[k], draw)
        yield k, recipe, render_recipe(recipe, grid, max_iter=domain.max_iter, norm_axes=norm_axes)


def _write_meta(path, meta):
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=lambda o: getattr(o, "tolist", str)(o)))
