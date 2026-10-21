"""Review GIFs for the ``contrast`` variety step (``synthmr/phantoms/contrast.py``).

For each phantom one GIF animates time with four panels: the phantom without contrast, with contrast (same recipe
and framing; both shown as the magnitude the simulation sees, ``|re + i im|`` of the channels shifted to [0, 1]),
the tissue map (tissues follow their structure through time) and each tissue's designed curve ``1 + e_k(t)``
(solid) next to its measured mean intensity relative to frame 0 (dashed), with a time cursor.

Example::

    python scripts/review_contrast.py --n 4 --out tests/artifacts/contrast
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthmr.phantoms.contrast import curve, tissue_labels  # noqa: E402
from synthmr.phantoms.fractal import JuliaConfig  # noqa: E402
from synthmr.phantoms.generate import FractalDomain, iter_recipes_3dt  # noqa: E402
from synthmr.phantoms.recipes import render_recipe  # noqa: E402
from synthmr.phantoms.variety import Variety, apply_framing  # noqa: E402

COLORS = plt.get_cmap("tab10").colors


def magnitude(block):
    return np.abs(block + (1 + 1j)).astype(np.float32) / 2


def review(domain, recipe, block, plane, path, title, fps=6):
    grid = domain.grid()
    con = recipe["contrast"]
    k = len(con["curves"])
    plain = magnitude(render_recipe({key: v for key, v in recipe.items() if key != "contrast"}, grid,
                                    max_iter=domain.max_iter, norm_axes=None))[plane]
    dyn = magnitude(block)[plane]
    coords = apply_framing(grid.coordinates(), recipe["framing"])
    labels = tissue_labels(coords, con, JuliaConfig(c=tuple(recipe["c"]), max_iter=domain.max_iter))[plane]
    t = np.linspace(0, 1, dyn.shape[0])
    measured = [np.array([dyn[i][labels[i] == j].mean() if (labels[i] == j).any() else np.nan
                          for i in range(len(t))]) for j in range(k)]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    ims = [axes[0].imshow(plain[0], cmap="gray", vmin=0, vmax=float(plain.max())),
           axes[1].imshow(dyn[0], cmap="gray", vmin=0, vmax=float(dyn.max())),
           axes[2].imshow(labels[0], cmap=ListedColormap(COLORS[:k]), vmin=-0.5, vmax=k - 0.5,
                          interpolation="nearest")]
    for ax, name in zip(axes, ["no contrast", "with contrast", "tissues (0 = interior)"]):
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    ax = axes[3]
    fine = np.linspace(0, 1, 200)
    for j, c in enumerate(con["curves"]):
        ax.plot(fine, 1 + curve(c, fine), color=COLORS[j], label=f"{j}: {c['kind']}")
        if np.isfinite(measured[j][0]) and measured[j][0] > 0:
            ax.plot(t, measured[j] / measured[j][0], "--", color=COLORS[j], lw=1)
    ax.set_xlabel("time (fraction of sweep)", fontsize=8)
    ax.set_ylabel("relative signal", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("designed (solid) / measured tissue mean (dashed)", fontsize=8)
    cursor = ax.axvline(0, color="k", lw=0.8)
    header = fig.suptitle("", fontsize=9)
    fig.tight_layout()

    def update(i):
        for im, arr in zip(ims, (plain, dyn, labels)):
            im.set_data(arr[i])
        cursor.set_xdata([t[i], t[i]])
        header.set_text(f"{title}  frame {i + 1}/{len(t)}")
        return [*ims, cursor, header]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(t), interval=1000 / fps).save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tests/artifacts/contrast")
    p.add_argument("--domains", default="aniso,iso")
    p.add_argument("--steps", default="frame,reject,background,warp,contrast",
                   help="variety steps (must include contrast)")
    a = p.parse_args(argv)
    domains = {"aniso": FractalDomain(), "iso": FractalDomain.isotropic(64, 2.4, 32)}
    for name in a.domains.split(","):
        domain = domains[name]
        for k, recipe, block in iter_recipes_3dt(domain, a.n, a.seed, norm="volume",
                                                 variety=Variety.from_steps(a.steps)):
            plane = (slice(None), block.shape[1] // 2)                       # central slice, all frames
            kinds = ", ".join(c["kind"] for c in recipe["contrast"]["curves"])
            path = Path(a.out) / f"{name}_julia_{k:03d}.gif"
            review(domain, recipe, block, plane, path, f"{name} julia_{k:03d} central slice")
            print(f"{name} julia_{k:03d} tissues [{kinds}] -> {path}", flush=True)


if __name__ == "__main__":
    main()
