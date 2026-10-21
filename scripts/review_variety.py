"""Review GIFs for the phantom variety steps (``synthmr/phantoms/variety.py``), applied cumulatively.

For each domain (anisotropic ``FractalDomain()`` 256^2 x 10 slices, isotropic cone ``64^3``) and phantom, one GIF
animates time with one panel per cumulative step: top = central slice, bottom = edge slice (anisotropic) or central
coronal plane (isotropic). Panel titles give the preview boundary-band fraction ``2 < escape < max_iter``.

Example::

    python scripts/review_variety.py --n 3 --out tests/artifacts/variety
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmr.phantoms.generate import FractalDomain, iter_recipes_3dt  # noqa: E402
from synthmr.phantoms.variety import STEPS, Variety  # noqa: E402
from synthmr.viz.gifs import save_panel_gif  # noqa: E402


def planes(block, isotropic):
    """``(T, Z, Y, X)`` complex -> ``(T, 2H, W)`` magnitude: central slice over edge slice / central coronal."""
    v = np.abs(block).astype(np.float32)
    z = v.shape[1] // 2
    bottom = v[:, :, v.shape[2] // 2] if isotropic else v[:, 0]
    return np.concatenate([v[:, z], bottom], axis=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tests/artifacts/variety")
    p.add_argument("--domains", default="aniso,iso")
    a = p.parse_args(argv)
    domains = {"aniso": (FractalDomain(), False), "iso": (FractalDomain.isotropic(64, 2.4, 32), True)}
    stages = [("none", ())] + [(s, STEPS[:i + 1]) for i, s in enumerate(STEPS)]
    for name in a.domains.split(","):
        domain, iso = domains[name]
        panels, titles = [[] for _ in range(a.n)], [[] for _ in range(a.n)]
        for label, steps in stages:
            for k, recipe, block in iter_recipes_3dt(domain, a.n, a.seed, norm="volume",
                                                     variety=Variety.from_steps(list(steps))):
                fr = recipe.get("framing", {})
                panels[k].append(planes(block, iso))
                titles[k].append(f"+{label}" + (f"\nzoom {fr['zoom']:.2f} detail {fr['detail']:.2f}"
                                                f" redraws {fr['redraws']}" if fr else ""))
                print(f"{name} {label:10s} julia_{k:03d} c={np.round(recipe['c'], 2).tolist()} "
                      f"{titles[k][-1].splitlines()[-1]}", flush=True)
        bottom = "coronal (y mid)" if iso else "edge slice z=0"
        for k in range(a.n):
            path = Path(a.out) / f"{name}_julia_{k:03d}.gif"
            save_panel_gif(panels[k], titles[k], path, fps=6, dpi=90,
                           suptitle=f"{name} julia_{k:03d}: top central slice, bottom {bottom}")
            print(f"-> {path}")


if __name__ == "__main__":
    main()
