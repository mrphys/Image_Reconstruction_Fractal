"""Random fractal recipes: a Julia ``c`` plus weighted field mixes for real/imag parts.

Moved from ``julia_generator_new_grasp.py``; draws are made in the same order so a
seeded ``np.random.Generator`` gives the same recipes as the original script.
"""


from .complexify import recipe_complex
from .fractal import JuliaConfig, QuaternionJulia, evaluate_field

# Field recipes: kind -> sampler of extra params.
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


def random_channel(rng, choices=None):
    """Pick 1-2 distinct fields with random params and weights (floored at 0.2, normalised)."""
    choices = FIELD_CHOICES if choices is None else choices
    n = int(rng.integers(1, 3))
    kinds = rng.choice(list(choices), size=n, replace=False)
    specs = [{"kind": str(k), "params": choices[k](rng)} for k in kinds]
    weights = rng.uniform(0.2, 1.0, size=n)
    weights = (weights / weights.sum()).tolist()
    return {"fields": specs, "weights": weights}


def random_recipe(rng, c):
    """Recipe dict ``{"c", "real", "imag"}`` for Julia parameter ``c``."""
    return {"c": [float(v) for v in c], "real": random_channel(rng), "imag": random_channel(rng)}


def describe(channel):
    return " + ".join(f"{w:.2f}·{f['kind']}" for f, w in zip(channel["fields"], channel["weights"]))


def render_recipe(recipe, grid, max_iter=30, norm_axes=(-2, -1)):
    """Render a recipe over ``grid`` -> complex array of ``grid.shape`` in [-1, 1]."""
    config = JuliaConfig(c=tuple(recipe["c"]), max_iter=max_iter)
    coords = grid.coordinates()
    parts = {
        part: [evaluate_field(coords, f["kind"], {**f["params"], "config": config})
               for f in recipe[part]["fields"]]
        for part in ("real", "imag")
    }
    return recipe_complex(parts["real"], parts["imag"], recipe["real"]["weights"],
                          recipe["imag"]["weights"], norm_axes=norm_axes)


__all__ = ["FIELD_CHOICES", "random_channel", "random_recipe", "describe", "render_recipe",
           "QuaternionJulia"]
