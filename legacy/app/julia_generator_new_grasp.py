"""Generate a batch of complex 3-D quaternion Julia fractals.

1. Define the rendering domain (image grid, slice bounds, z sweep).
2. Sample candidate ``c`` values near the quaternion Mandelbrot boundary.
3. Choose 10 of them.
4. For each ``c``, build a complex volume ``(n_z, H, W)`` whose real and
   imaginary channels are random weighted mixes of randomly chosen fields.
   Repeat this for ``N_Y_SLICES`` contiguous y values (same ``c`` and recipe)
   to get a 4-D dataset of stacked 3-D submanifolds.
5. Save each volume (plus its recipe) to ``julia_XX_slice_N.npy`` / ``.json``.
6. Render a GIF of each volume swept through z, as in ``example.py``.
"""

import json
from pathlib import Path

import matplotlib


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .fractal_framework import (
    JuliaConfig, Slice, QuaternionJulia, candidate_c, compose_complex,
    complex_channels,
)

SEED = 0
N_FRACTALS = 500
OUT_DIR = Path("resources/new_julia_grasp")

# --- 1. Domain --------------------------------------------------------------
# Rows = x (imaginary), columns = w (real); the third spatial axis is z, which
# we sweep to build a 3-D volume. The fourth axis is y: each y value gives one
# 3-D submanifold, and consecutive y values are spaced like z so the stacked
# slices form a contiguous 4-D grid centred on y = 0.
SHAPE = (256,256)
BOUNDS = ((-1.2, 1.2), (-1.6, 1.6))
Z_SLICES = np.linspace(-0.3, 0.3, 32)
N_Y_SLICES = 10
Y_STEP = 0.1
Y_SLICES = (np.arange(N_Y_SLICES) - (N_Y_SLICES - 1) / 2) * Y_STEP
MAX_ITER = 30

# Field recipes: (kind, param sampler). Samplers draw random extra params.
FIELD_CHOICES = {
    "escape": lambda rng: {},
    "smooth_escape": lambda rng: {},
    "distance": lambda rng: {},
    "orbit_trap": lambda rng: {
        "trap": str(rng.choice(["point", "line", "plane"])),
        "axes": tuple(int(a) for a in sorted(
            rng.choice(4, size=rng.integers(1, 4), replace=False))),
    },
    "power": lambda rng: {"power": int(rng.integers(3, 6))},
    "perturbed": lambda rng: {"epsilon": float(rng.uniform(0.005, 0.05))},
    "internal_detail": lambda rng: {},
}


def random_channel(rng):
    """Pick 1-2 distinct fields with random params and weights."""
    n = int(rng.integers(1, 3))
    kinds = rng.choice(list(FIELD_CHOICES), size=n, replace=False)
    specs = [{"kind": str(k), "params": FIELD_CHOICES[k](rng)} for k in kinds]
    # Floor at 0.2 so no chosen field is weighted into irrelevance.
    weights = rng.uniform(0.2, 1.0, size=n)
    weights = (weights / weights.sum()).tolist()
    return {"fields": specs, "weights": weights}


def render_volume(renderer, recipe, y):
    """Render the complex volume ``(n_z, H, W)`` for one recipe at fixed ``y``."""
    volume = np.empty((len(Z_SLICES), *SHAPE), dtype=np.complex128)
    for i, z in enumerate(Z_SLICES):
        slice_ = Slice(axes=(1, 0), fixed=(0.0, 0.0, y, z), bounds=BOUNDS)
        parts = {}
        for part in ("real", "imag"):
            parts[part] = [
                renderer.render(SHAPE, slice_, f["kind"], **f["params"])
                for f in recipe[part]["fields"]
            ]
        field = compose_complex(
            parts["real"], parts["imag"],
            recipe["real"]["weights"], recipe["imag"]["weights"],
        )
        # Recentre from [0, 1] to [-1, 1] so the phase spans all quadrants.
        volume[i] = 2 * field - (1 + 1j)
    return volume


def describe(channel):
    return " + ".join(
        f"{w:.2f}·{f['kind']}" for f, w in zip(channel["fields"], channel["weights"])
    )


# def save_gif(volume, recipe, path):
#     """Animate the four complex channels through the z sweep."""
#     frames = [complex_channels(v) for v in volume]
#     panels = [
#         ("real", "magma", (-1, 1)),
#         ("imag", "magma", (-1, 1)),
#         ("magnitude", "inferno", (0, np.sqrt(2))),
#         ("phase", "twilight", (-np.pi, np.pi)),
#     ]
#     fig, axes = plt.subplots(2, 2, figsize=(12, 7))
#     images = []
#     for ax, image, (title, cmap, (vmin, vmax)) in zip(axes.flat, frames[0], panels):
#         im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
#         ax.set_title(title)
#         ax.axis("off")
#         fig.colorbar(im, ax=ax, shrink=0.8)
#         images.append(im)
#     c = recipe["c"]
#     header = (f"c = ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f})\n"
#               f"re: {describe(recipe['real'])}   im: {describe(recipe['imag'])}")
#     suptitle = fig.suptitle("", fontsize=9)
#     plt.tight_layout()

#     def update(i):
#         for im, image in zip(images, frames[i]):
#             im.set_data(image)
#         suptitle.set_text(f"{header}\nz = {Z_SLICES[i]:.2f}")
#         return [*images, suptitle]

#     anim = FuncAnimation(fig, update, frames=len(frames), interval=100)
#     anim.save(path, writer=PillowWriter(fps=10), dpi=80)
#     plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    # --- 2. Candidate c values from the quaternion Mandelbrot set -----------
    candidates = candidate_c(resolution=40, extent=1.5, max_iter=50,
                             min_iter=10, max_escape=30)
    print(f"{len(candidates)} Mandelbrot-boundary candidates")

    # --- 3. Choose 10 --------------------------------------------------------
    idx = rng.choice(len(candidates), size=N_FRACTALS,
                     replace=N_FRACTALS > len(candidates))
    chosen = candidates[idx]

    print(chosen)

    for k, c in enumerate(chosen):
        print(f"[{k + 1}/{N_FRACTALS}] c = {c}")
        # --- 4. Randomised channel recipe per c ------------------------------
        recipe = {
            "c": [float(v) for v in c],
            "real": random_channel(rng),
            "imag": random_channel(rng),
        }
        renderer = QuaternionJulia(JuliaConfig(c=tuple(recipe["c"]), max_iter=MAX_ITER))

        # Same c and complex recipe for every contiguous y submanifold.
        for s, y in enumerate(Y_SLICES, start=1):
            volume = render_volume(renderer, recipe, y)

            # --- 5. Save ------------------------------------------------------
            stem = OUT_DIR / f"julia_{k:02d}_slice_{s}"
            np.save(stem.with_suffix(".npy"), volume)
            # recipe["z_slices"] = Z_SLICES.tolist()
            # recipe["y"] = float(y)
            # recipe["bounds"] = BOUNDS
            # stem.with_suffix(".json").write_text(json.dumps(recipe, indent=2))

        # # --- 6. GIF -----------------------------------------------------------
        # save_gif(volume, recipe, stem.with_suffix(".gif"))
        # print(f"[{k + 1}/{N_FRACTALS}] {stem}  re: {describe(recipe['real'])}  "
        #       f"im: {describe(recipe['imag'])}")


if __name__ == "__main__":
    main()
