# Legacy -> new location map

Paths are relative to `legacy/` (frozen copies of the base repo `utils/` and of `app/`).

## Entry points
| Legacy | New | Notes |
|---|---|---|
| `train_network.ipynb` dataset loop (base repo) | preset `fractal_2dt_legacy` + `Pipeline` (notebook updated) | golden fixture A |
| `app/process_data_old.py::process_fractals_new_grasp` | preset `fractal_new_2dt` / `scripts/make_dataset.py` | golden fixture B_cc |
| `app/process_data_old.py::process_fractals_new_grasp_grouped` | preset `fractal_new_2dt_nocc`; grouping superseded by `fractal_3dt_sos` | per-group re-seeding = `pipeline.rng.reset(transforms=s, augment=s)`; it also deleted inputs (not reproduced) |
| `app/julia_generator_new_grasp.py` | `scripts/generate_fractals.py --mode 2dt` (`synthmr/phantoms/generate.py`) | `--mode 3dt` writes whole `(T, Z, H, W)` volumes |
| other `process_*` (real MRI, Inter4k, superres, VarNet, spiral prospective) | not ported (stay in `legacy/app/process_data_old.py`) | out of scope by decision |

## Fractals
| Legacy | New | Notes |
|---|---|---|
| `app/fractal_framework/*` | `synthmr/phantoms/fractal/*` | `channels.py` gained `axes=` (per-frame vs whole-volume normalisation) |
| `Slice` (2-D) | `Grid` / `Axis` (`synthmr/phantoms/fractal/grid.py`) | N-D; renders `(T, Z, Y, X)` in one call |
| `julia_generator_new_grasp.{FIELD_CHOICES, random_channel, describe, render_volume}` | `synthmr/phantoms/recipes.py::{FIELD_CHOICES, random_channel, describe, render_recipe}` | same draw order |
| `base_repo/fractal_utils.{julia_escape, mandelbrot_escape, build_grid, test_complexity, quaternion_*}` | `synthmr/phantoms/fractal/legacy.py` | also reproducible via the `escape` field on a `Grid` |
| `base_repo/fractal_utils.{make_complex_fractal, make_sine_wave, blur_and_sharpen, norm}` | `synthmr/phantoms/complexify.py::{sine_wave, make_sine_wave, blur_and_sharpen, norm}` | `rng=` argument; `np.random` = legacy global state |
| `app/process_data_old.make_complex_fractal` | `complexify.sine_wave_smooth` | different (inverted-norm, blurred) variant |
| `app/process_data_old.make_complex_fractal_spiral` | `complexify.sine_wave_spiral` | |
| `np.stack([v.real, v.imag]) * 127.5 + 127.5` | `complexify.complex_to_video` | |

## Simulation (`preprocessing_natural_videos.py`, app version)
| Legacy | New |
|---|---|
| `config_default` | `synthmr.simulation.SimulationConfig` (`from_legacy`) |
| `preprocessing_fn` | `synthmr.simulation.simulate` -> `backend/tf/simulation.py::simulate_2d` (canonical `(C, T, 1, H, W)` k-space) |
| `elliptical_mask` | `synthmr/simulation/masks.py::elliptical_mask_2d` |
| `simulate_coils`, `gauss_kernel` | `backend/tf/simulation.py::simulate_coils_2d`, `gauss_kernel` |
| `_backgroundphase` | `backend/tf/simulation.py::background_phase_2d` |
| `_awgn` (also duplicated in process_data*.py) | `backend/tf/simulation.py::awgn` (noise stream) |
| `rg`, `noise_rg` | `synthmr.rng.RNG().transforms` (seed 1), `.noise` (seed 2) |

## Sampling (`preprocessing_trajectory_gen.py`)
| Legacy | New |
|---|---|
| `config_radial_traj` | `synthmr.sampling.TrajectoryConfig` (`from_legacy`) |
| `create_traj_fn` (radial / spiral search) | `synthmr.sampling.make_trajectory` -> `backend/tf/sampling.py::{radial, spiral}` |
| `gen_spiral_traj` | `backend/tf/sampling.py::gen_spiral_traj` (verbatim) |
| NUFFT calls inside `preprocessing_fn` | `backend/tf/sampling.py::SamplingOperator` (`forward`, `adjoint`, `apply_dcw`) |

## Recon (`preprocessing_multicoil_noselect.py`, app version)
| Legacy | New |
|---|---|
| `config_base_preproc` | `synthmr.recon.ReconConfig` (`from_legacy`) |
| `preprocessing_fn` | `synthmr.recon.reconstruct` -> `backend/tf/recon.py::reconstruct_2d` (coil maps strategy `walsh_xyt_legacy`) |
| `make_fs_rtcine_image`, `gaussian_blur` | `backend/tf/recon.py` |
| `rolling_fn` | `synthmr.recon.augment` -> `backend/tf/recon.py::augment_2d` |
| `rg` | `synthmr.rng.RNG().augment` (seed 1) |

## Helpers from `app/process_data_old.py`
| Legacy | New |
|---|---|
| `display_gifs`, `plot_data.py` | `synthmr/viz/gifs.py::{save_gif, save_panel_gif, save_slice_mosaic_gif, save_ortho_gif, save_sample_gifs}` |
| `convert_to_real`, `convert_to_complex`, `normalise` | `synthmr.utils.mri.{convert_to_real, convert_to_complex, normalise_max_abs}` |
| `compute_trajectory_and_density` (+ `_compute_trajectory_and_density*`) | `synthmr.utils.mri.compute_trajectory_and_density` (ordering is a parameter) |
| `_sample_noncartesian_kspace` | `synthmr.utils.mri.sample_noncartesian_kspace` |
| `LowPassScaler`, `Transformer`, `_scale_kspace_data` | `synthmr.utils.mri.LowPassScaler` (`fit_transform`) |
| `scale_frames`, `circular_pad`, `crop_or_circular_pad`, `float_to_uint8`, `int255_to_float` | `synthmr/utils/array.py` |
| `gaussian_blur_scipy`, `sharpen_image`, `central_crop_and_zoom`, `median_blur_3d_slicewise`, `unsharp_mask_3d_slicewise` | `synthmr/utils/filters.py` (`gaussian_blur`, `sharpen_3d`, ...) |
| `recon_adjoint`, `_make_fs_trigcine_image*`, `CartDims`/`NonCartDims`, `process_kspace*` | not ported (real-MRI only; use `tfmri.recon.adjoint` via the backend if needed) |
