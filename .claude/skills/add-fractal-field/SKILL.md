---
name: add-fractal-field
description: Add a new scalar fractal field or channel recipe to synthmr.phantoms.fractal.
---
1. Fast path: add a numba grid kernel in `synthmr/phantoms/fractal/_kernels.py` and an entry in `_GRIDS`
   (`fields.py`), then a `@register_field` point function. Slow path: just `@register_field` (per-point Python loop).
2. Coordinates are `(..., 4)` in `(w, x, y, z)` order; N-D grids come from `grid.py`.
3. To make it sampleable in random recipes, add a param sampler to `FIELD_CHOICES` in `synthmr/phantoms/recipes.py`.
4. Test: deterministic output for a fixed config; render a small `(T, H, W)` block and save a GIF.
5. Existing fields must stay bit-identical (regression R1/R2).
