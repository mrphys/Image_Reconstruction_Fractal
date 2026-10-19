# Backend contract

A backend is a package selected with `synthmr.backend.set_backend(name)` / `get_backend()` (default:
`$SYNTHMR_BACKEND`, else `tf`). `load_backend(name)` returns one without activating it; `Pipeline` uses it for
`PipelineConfig.backend`, so building a pipeline never changes the global default.

```
synthmr/backend/<name>/
  core.py         # the ops in the table below
  <stage>.py      # implementation of stage logic that is tensor math (simulation, sampling, recon, ...)
```
Two backends exist:

- **`tf`** (`synthmr/backend/tf/`): thin wrappers over the exact legacy TF calls (tfmri, tfft, tfa). The only
  package allowed to import TensorFlow-family libraries. Reproduces the legacy golden.
- **`sigpy`** (`synthmr/backend/sigpy/`): numpy on CPU, cupy on GPU, NUFFTs from `sigpy`, following
  [mikgroup/extreme_mri](https://github.com/mikgroup/extreme_mri) (`sp.Device`/`xp`, pixel-unit `coord`,
  per-frame NUFFT loops, `JsenseRecon`). The only package allowed to import sigpy / cupy. Imports and runs
  without TensorFlow (`.devcontainer/sigpy`, `requirements-sigpy.txt`).

A new backend (registered with `register_backend`) must implement the same functions with the same semantics
and pass `tests/unit/test_backend.py` (parametrised over the importable backends) plus
`tests/regression/test_xbackend.py`-style comparison against captured TF outputs.

Stage modules: `simulation.py` (2D legacy simulation, `prepare_video`), `sampling.py` (trajectories,
`SamplingOperator` with `none` / `cartesian_kz` through-plane encoders), `recon.py` (2D recon + augmentation),
`recon3d.py` (3D recon, coil maps), `utils.py` (real-MRI helpers). The 3D simulation and 3D augmentation are
numpy and need only the core ops (`fft`, `Generator`, `to_numpy`/`from_numpy`) plus `simulation.prepare_video`.

Elementwise tensor math inside stage implementations is deliberately *not* abstracted: wrapping every op
would add indirection without making a port easier. Porting a stage means rewriting its `<stage>.py`
(the sigpy stages are line-by-line ports of the TF ones, keeping the order of random draws).

| Op | TF call | Semantics to preserve | sigpy backend |
|---|---|---|---|
| `fft` / `ifft` | `tfmri.signal.fft/ifft(axes, norm='ortho', shift=True)` | centred, orthonormal | `sp.fft/ifft(center=True, norm='ortho')` |
| `nufft_forward` | `tfft.nufft(type_2, forward)` (callers divide by `sqrt(npix)`) | unscaled DFT on a centred grid, traj in [-pi, pi] `(..., R, ndim)`, `traj[..., i]` <-> image axis `-ndim+i`, batch axes broadcast | `sp.nufft(x, traj*N/2pi) * sqrt(N)`, `oversamp=2, width=6` (~5e-5 rel.), loop over trajectory batch axes |
| `nufft_adjoint` | `tfft.nufft(type_1, backward, grid_shape)` | adjoint of the above | `sp.nufft_adjoint(...) * sqrt(N)` |
| `radial_trajectory` | `tfmri.sampling.radial_trajectory` + `flatten_trajectory` | orderings linear, golden, golden_half, tiny, tiny_half, sorted, sorted_half | numpy port of tfmri (float32), matches to 2e-7 |
| `estimate_radial_density` | `tfmri.sampling.estimate_radial_density`, dcw = 1/dens | applied to the flattened trajectory (legacy) | numpy port, matches to 6e-8 |
| `coil_compressor_svd` | `tfmri.coils.CoilCompressorSVD(coil_axis=-3).fit/transform` | fit on undersampled k-space | `xp.linalg.svd`; virtual coils equal TF's up to one phase each |
| `estimate_sensitivities` | `tfmri.coils.estimate_sensitivities(method='walsh')` | spatial rank 2 or 3, coil axis last, 5-point zero-padded box filter | same algorithm (box filter + batched `eigh`, slab-wise), matches to 4e-7 up to the unanchored eigenvector phase |
| `combine_coils` | `tfmri.coils.combine_coils(maps=None -> SoS)` | | same formula |
| `crop_or_pad` | `tfmri.resize_with_crop_or_pad` | centred, tfmri offsets | same offsets, exact |
| `scale_by_min_max` | `tfmri.scale_by_min_max` | complex: on magnitude | exact |
| `resize` | `tf.image.resize(bicubic)` | background phase upsampling | TF ResizeBicubic (Keys a=-0.5, half-pixel, 1024-entry table), matches to 4e-7 |
| `translate` | `tfa.image.translate(bilinear)` | coil placement, zero fill | `ndimage.shift(order=1, mode='grid-constant')`, matches to 1e-6 |
| `Generator(seed)` | `tf.random.Generator.from_seed(seed, "philox")`: `uniform`, `normal`, `reset` | draw order is part of the contract for exact regression | `synthmr.rng.NumpyGenerator` (host numpy philox) |
| `device(spec)` | `tf.device` | legacy hard-codes `/gpu:0`; now chosen by config | `sp.Device` (`None`: GPU 0 if cupy finds one, else CPU) |

## Random streams across backends

`synthmr.rng.RNG` holds the streams transforms (seed 1), noise (2) and augment (1). With `kind="backend"`
(default, `PipelineConfig.rng = "backend"`) each stream is the backend's `Generator`: TF philox, which the
golden needs. With `kind="numpy"` every backend uses `NumpyGenerator`, so TF and sigpy draw the same numbers in
the same order and whole pipelines can be compared (`tests/regression/test_xbackend.py`).

## sigpy-only / TF-only features

- `ReconConfig.coil_maps = "jsense"` (sigpy): `sigpy.mri.app.JsenseRecon` on the density-compensated
  non-Cartesian data of all frames (weights = dcw). 2D: maps `(H, W, 1, C)`; 3D+t: coordinates `[kz, ky, kx]`
  with partition `p` at kz = `p - Z//2`, maps `(Z, H, W, C)`, phase-anchored like the Tukey maps. Settings:
  `jsense_ker_width`, `jsense_calib_width`, `jsense_lamda`, `jsense_max_iter`, `jsense_max_inner_iter`.
  On CPU the 3D version is slow (single-threaded numba 3D gridding); use a GPU.
- Generated spiral trajectories (TF): need tfmri's spiral-waveform library. With sigpy, load a precomputed
  trajectory with `TrajectoryConfig(kind="file")` (below).
- `utils.LowPassScaler` (TF): depends on `tfmri.signal.filter_kspace`.

## Trajectories from files (all backends)

`FileTrajectoryConfig` (`kind="file"`) loads `traj_path` / `density_path` `.npy` files in
`synthmr.sampling.make_trajectory`, independent of the backend. Trajectory shapes `(R, 2)`, `(V, S, 2)`
(with `views_axis=True`), `(T, R, 2)` or `(T, V, S, 2)`; density has the same shape without the last axis.
`traj_units`: `radians` ([-pi, pi]) or `pixels` (sigpy's [-N/2, N/2], N = `base_resolution`).
`density_kind`: `density` (dcw = 1/density) or `dcw`. Fewer frames than `phases` repeat cyclically.
