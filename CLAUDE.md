# Image_Reconstruction_Fractal: refactor branch `refactor/modular-3dt`

- Package: `synthmr/` (see docs/ARCHITECTURE.md). Frozen old code: `legacy/` (never edit; only used for golden capture).
- Axis/format conventions: docs/DATA_CONVENTIONS.md. Backend contract: docs/BACKEND.md.
- Only `synthmr/backend/tf/` may import tensorflow / tensorflow_mri / tensorflow_nufft / tensorflow_addons. It holds the core
  MRI ops (`core.py`) plus the TF implementation of each stage; stage modules elsewhere stay backend-free.
- Tests: `python -m pytest tests -q -p no:warnings` (GPU, ~1 min).
- No module-level RNGs: random draws go through the injected `synthmr.rng.RNG` streams (transforms / noise / augment). Draw order in the 2D path must match legacy.
- Every change to the 2D path must pass the golden regression (skill: regression-check). Golden is GPU-only.
- Run from the repo root inside the devcontainer (TF 2.10, GPU).
