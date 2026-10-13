# Image_Reconstruction_Fractal (`synthmr`)

Synthetic training data for deep-learning MRI reconstruction, made from **quaternion Julia fractals**
instead of patient data.

Read the paper! https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.70521

The `synthmr` package builds a random fractal "movie", simulates a multicoil MRI acquisition of it,
undersamples it on a radial trajectory and writes out matched pairs:

| | What it is | Use it as |
|---|---|---|
| **input** | zero-filled, coil-combined reconstruction of the undersampled data (aliased, noisy) | network input |
| **target** | fully sampled, coil-combined image | ground truth |

Two kinds of data are supported:

- **2D+t**: one slice over time, `(T, H, W)`. This reproduces the original paper pipeline exactly.
- **3D+t**: a stack of slices over time, `(T, Z, H, W)`, acquired as a stack-of-stars (radial in-plane,
  Cartesian along z), with simulation that stays consistent across the whole volume.

## Quick start

All commands run from the repository root, inside the devcontainer (TensorFlow 2.10, **GPU required**;
see [Requirements](docs/USER_GUIDE.md#1-requirements)).

```bash
# 1. Generate 100 fractal volumes (T=32, Z=10, 256x256), normalised over the whole volume
python scripts/generate_fractals.py --mode recipe_3dt --n 100 --out data/julia4d --norm volume

# 2. Turn them into (input, target) training pairs with the 3D+t stack-of-stars preset
python scripts/make_dataset.py --preset fractal_3dt_sos --input data/julia4d --output data/pairs3d --gifs 3
```

For 2D+t data:

```bash
python scripts/generate_fractals.py --mode recipe_2dt --n 50 --out data/julia
python scripts/make_dataset.py --preset fractal_new_2dt --input data/julia --output data/pairs
```

From Python:

```python
import numpy as np
from synthmr.pipeline import Pipeline, load_config

cfg = load_config("fractal_3dt_sos", overrides={"trajectory": {"radial_spokes": 21}})
sample = Pipeline(cfg).run(np.load("data/julia4d/julia_000.npy"))
sample.input.shape, sample.target.shape     # (24, 10, 256, 256, 2), (24, 10, 256, 256, 1)
```

## Documentation

| Document | For |
|---|---|
| [**User guide**](docs/USER_GUIDE.md) | Everything a user needs: both CLIs, all presets, every configuration option, output formats, the Python API and troubleshooting |
| [Architecture](docs/ARCHITECTURE.md) | Package layout, the processing stages, how 2D and 3D data flow differ |
| [Data conventions](docs/DATA_CONVENTIONS.md) | Array layouts, axis order, normalisation details |
| [Backend contract](docs/BACKEND.md) | The TensorFlow backend and how to port it (e.g. to PyTorch) |
| [Regression testing](docs/REGRESSION.md) | How the refactor is checked against the original code |
| [Legacy map](docs/LEGACY_MAP.md) | Where each function from the old scripts now lives |

## Repository layout

```
synthmr/            the package (phantoms, simulation, sampling, recon, pipeline, viz, backend)
scripts/            command-line tools: generate_fractals.py, make_dataset.py
train_network.ipynb example: build a 2D+t dataset and train a 3D UNet on it
trained_models/     pretrained network weights (.h5)
tests/              unit tests and golden regression tests
legacy/             frozen copy of the original code (reference only; do not edit)
docs/               documentation
```
