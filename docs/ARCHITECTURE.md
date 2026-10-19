# Architecture

Goal: generate paired (undersampled input, fully-sampled target) MRI training data from synthetic
fractals, in 2D+t and 3D+t, with every MRI operation behind a swappable backend (TF now, torch later).

## Stages
```
phantom (fractal)  ->  simulation (image -> multicoil k-space)  ->  sampling (trajectory + operator)
                   ->  recon (undersample, coil maps, combine, format)  ->  augment  ->  io
```
| Package | Responsibility | Replaces (legacy) |
|---|---|---|
| `synthmr.phantoms` | quaternion Julia fields, N-D grids, complexify (real/imag channels) | `app/fractal_framework`, `utils/fractal_utils.py`, `make_complex_fractal` |
| `synthmr.simulation` | contrast, mask, background phase, coil sensitivities, noise, FFT | `preprocessing_natural_videos.py` |
| `synthmr.sampling` | `Trajectory` = in-plane encoder (radial, spiral) x through-plane encoder (none, cartesian kz); `SamplingOperator` | `preprocessing_trajectory_gen.py` |
| `synthmr.recon` | NUFFT forward/adjoint, coil compression, coil-map strategies, combine, output formats, rolling/rotation | `preprocessing_multicoil_noselect.py` |
| `synthmr.pipeline` | config dataclasses, presets, `Pipeline`, I/O | the `process_fractals*` functions |
| `synthmr.backend` | backend packages (`tf/`): core MRI ops + TF stage implementations; only place importing TF-family libs | - |
| `synthmr.rng` | injected random streams ("transforms", "noise"), no module globals | module-level `rg`, `noise_rg` |
| `synthmr.viz` | GIFs for human checks | `display_gifs`, `plot_data.py` |

## Usage
```
python scripts/generate_fractals.py --mode 3dt --n 10 --out data/julia4d --norm volume --gifs 2
python scripts/make_dataset.py --preset fractal_3dt_sos --input data/julia4d --output data/pairs3d --gifs 2
python scripts/make_dataset.py --list-presets            # --print-config shows the resolved config
```
```python
from synthmr.pipeline import Pipeline, load_config
cfg = load_config("fractal_new_2dt", overrides={"trajectory": {"spokes": 21}})
sample = Pipeline(cfg).run(volume)      # sample.input, sample.target
```

`legacy/` is a frozen snapshot of the old code, used only to capture golden outputs (see REGRESSION.md).

## Status
- [x] Phase 1: legacy snapshot, golden capture (GPU), docs/skills skeleton
- [ ] Phase 2: refactor with stage-by-stage regression
  - [x] 2a backend + RNG streams (legacy code driven by `synthmr.rng` reproduces golden)
  - [x] 2b fractal framework + N-D grids + complexify (R1-R3 exact)
  - [x] 2c simulation (R4 k-space exact)
  - [x] 2d sampling + recon + augment (R5-R7: whole chain matches golden)
  - [x] 2e utils centralised, Pipeline/presets/config, CLIs, notebook updated, old `utils/` removed
  - [x] 2f lint clean, no TF outside `backend/tf`, no module-level RNG or hard-coded device
- [x] Phase 3: 3D+t (preset `fractal_3dt_sos`)
  - [x] 3a volume-consistent simulation: one 3D ellipsoid mask, one 3D background phase, shared SNR; in-plane rot90 / z-flip / temporal roll applied identically to all slices (`synthmr/simulation/sim3d.py`, `synthmr/recon/augment3d.py`)
  - [x] 3b 3D coil sensitivities: anisotropic 3D Gaussians + 3D background phase (`synthmr/simulation/coils.py`)
  - [x] 3c stack-of-stars: `SamplingOperator(through_plane="cartesian_kz")`, aligned or golden-rotated partitions (`synthmr/sampling/encoders.py`)
  - [x] 3d 3D coil maps: sum over time -> 3D Tukey low-pass in k-space -> 3D Walsh, phase anchored to coil 0 (`backend/tf/recon3d.py`)
  - [x] 3e preset, h5 output, `generate_fractals.py --mode 3dt`, slice-mosaic / ortho / coil-map GIFs

## 2D vs 3D+t data flow
| Step | 2D+t (legacy presets) | 3D+t (`fractal_3dt_sos`) |
|---|---|---|
| phantom | `(T, H, W)` per slice file | `(T, Z, H, W)` one file, normalised over the volume (`--norm volume`) |
| mask / phase | 2D ellipse (cv2), 2D bicubic phase | 3D ellipsoid, tricubic 3D phase (numpy) |
| coils | 2D Gaussians (TF, translate) | 3D Gaussians (numpy), z-extent from `slice_spacing_px` |
| sampling | radial per frame | radial per kz partition + Cartesian kz FFT |
| coil maps | Walsh on `(H, W, T, C)` (legacy, unanchored phase) | Walsh on the time-averaged, Tukey low-passed 3D volume, anchored phase |
| augmentation | rot90 + roll (TF) | rot90 + optional z-flip + roll shared by all slices (numpy) |
| output | `*_input.npy (T,H,W,2)`, `*_output.npy (T,H,W,1)` | `.h5` `input (T,Z,H,W,2)`, `output (T,Z,H,W,1)` |
