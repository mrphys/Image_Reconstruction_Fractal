"""Generate fractal phantoms for make_dataset.py.

Examples::

    python scripts/generate_fractals.py --mode 3dt --n 100 --out data/julia4d --norm volume
    python scripts/generate_fractals.py --mode 2dt --n 500 --out data/julia --slices 10
    python scripts/generate_fractals.py --mode legacy --n 692 --out data/legacy
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmr.phantoms.generate import FractalDomain, LegacyDomain, generate  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["legacy", "2dt", "3dt"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--slices", type=int, default=10, help="2dt/3dt modes: number of y slices (Z)")
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--norm", choices=["frame", "volume"], default="frame",
                   help="frame = legacy per-2D-frame normalisation; volume = consistent across T and Z")
    p.add_argument("--gifs", type=int, default=0, help="write GIFs for the first N outputs")
    a = p.parse_args(argv)
    if a.mode == "legacy":
        dom = LegacyDomain(shape=tuple(a.size), frames=a.frames, **({"max_iter": a.max_iter} if a.max_iter else {}))
    else:
        dom = FractalDomain(shape=tuple(a.size), frames=a.frames, slices=a.slices,
                            **({"max_iter": a.max_iter} if a.max_iter else {}))
    paths = generate(a.mode, a.n, a.out, seed=a.seed, domain=dom, norm=a.norm)
    if a.gifs:
        import numpy as np
        from synthmr.viz.gifs import save_gif, save_ortho_gif, save_slice_mosaic_gif
        for path in paths[: a.gifs]:
            v = np.load(path)
            if v.ndim == 4:
                save_slice_mosaic_gif(v, path.with_suffix(".mosaic.gif"), path.stem)
                save_ortho_gif(v, path.with_suffix(".ortho.gif"), path.stem)
            else:
                save_gif(v, path.with_suffix(".gif"), path.stem)
    print(f"wrote {len(paths)} files to {a.out}")


if __name__ == "__main__":
    main()
