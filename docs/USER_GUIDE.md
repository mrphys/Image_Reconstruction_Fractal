# synthmr user guide

`synthmr` generates paired MRI training data from synthetic quaternion-Julia fractals. This guide covers
everything you can do with it and every option you can set.

**Contents**

1. [Requirements](#1-requirements)
2. [How it works](#2-how-it-works)
3. [Step 1: generating fractals (`generate_fractals.py`)](#3-step-1-generating-fractals)
4. [Step 2: making a dataset (`make_dataset.py`)](#4-step-2-making-a-dataset)
5. [Presets](#5-presets)
6. [Configuration reference](#6-configuration-reference)
7. [Output files and array shapes](#7-output-files-and-array-shapes)
8. [Python API](#8-python-api)
9. [Working with fractals directly](#9-working-with-fractals-directly)
10. [Visualisation](#10-visualisation)
11. [Reproducibility and random seeds](#11-reproducibility-and-random-seeds)
12. [Training a network](#12-training-a-network)
13. [Running the tests](#13-running-the-tests)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Requirements

- **The devcontainer** in `.devcontainer/` (TensorFlow 2.10 GPU image with `tensorflow-mri`,
  `tensorflow-nufft`, `tensorflow-addons`, `numba`, `opencv-python`, `h5py`, `pyyaml`). Open the folder in
  VS Code and choose *Reopen in Container*, or build the Dockerfile yourself.
- **An NVIDIA GPU.** Dataset creation (`make_dataset.py`) needs one. On CPU, TF 2.10 fails inside the Walsh
  coil-sensitivity estimation: it either segfaults or raises `Eigen decomposition was not successful`.
  Fractal generation (`generate_fractals.py`) is pure numpy/numba and runs fine on CPU.
- **Or, without TensorFlow: the sigpy container** in `.devcontainer/sigpy/` (CUDA 12, `sigpy`, `cupy`;
  packages in `requirements-sigpy.txt`). It sets `SYNTHMR_BACKEND=sigpy`, runs on a GPU through cupy and
  also on CPU (numpy). It supports radial and file-loaded trajectories, not generated spirals (see
  [Backend](#backend)).
- Run every command **from the repository root**. The scripts add the repository to `sys.path`, so you
  don't need to install anything.

## 2. How it works

Making data is a two-step process. Each step has its own script:

```
generate_fractals.py                         make_dataset.py (Pipeline)
────────────────────                         ─────────────────────────────────────────────────────────────
random Julia c  ──►  fractal volume (.npy)  ──►  complexify ──► simulate ──► sample ──► reconstruct ──► augment ──► save
                     (T,H,W) or (T,Z,H,W)        to 2-ch      multicoil    radial     zero-filled      rotate/    .npy/.h5
                                                 video        k-space      (+ kz)     input + target   roll
```

| Stage | What happens | Configured by |
|---|---|---|
| **phantom** | A random quaternion Julia set is evaluated over a grid. Two axes are space, one is time, and in 3D one is slice position. Several fractal "fields" are mixed into a real part and an imaginary part. | `generate_fractals.py` options |
| **complexify** | The stored volume becomes a 2-channel (real, imag) "video" with values in [0, 255]. | `input` |
| **simulate** | Contrast adjustment, a random elliptical object mask, smooth background phase, random Gaussian coil sensitivities, complex Gaussian noise at a random SNR, then FFT to multicoil k-space. | `simulation` |
| **sample** | A radial trajectory for each frame (2D), or for each kz partition with Cartesian kz (3D stack-of-stars). | `trajectory` |
| **reconstruct** | Undersample k-space along the trajectory with a NUFFT and density compensation. Optionally SVD coil compression. Build the zero-filled adjoint (input) and the fully sampled image (target), estimate coil maps, and coil-combine. | `recon` |
| **augment** | One random in-plane 90° rotation, an optional z-flip (3D) and a random temporal roll, all shared by input and target. Then crop to the final number of frames. | `augment` |
| **save** | `.npy` pair or `.h5` file, plus metadata JSON and optional GIFs. | `output` |

## 3. Step 1: generating fractals

```bash
python scripts/generate_fractals.py --mode MODE --n N --out DIR [options]
```

### Modes

| `--mode` | Output per fractal | dtype / range | Use with preset |
|---|---|---|---|
| `legacy` | `(T, H, W)` integer escape-time volume. Fractals that fail an entropy-based complexity check (threshold 1.7) are rejected and redrawn. This is the original paper method. | int32 | `fractal_2dt_legacy` |
| `2dt` | One `(T, H, W)` file per slice: `--slices` files per fractal, each a different y-position through the same 4D fractal | complex128 in [-1, 1] | `fractal_new_2dt`, `fractal_new_2dt_nocc` |
| `3dt` | One `(T, Z, H, W)` volume per fractal | complex128 in [-1, 1] | `fractal_3dt_sos` |

In the **recipe** modes, each fractal gets a random *recipe*. A recipe is a Julia parameter `c` plus, for
the real part and for the imaginary part separately, a weighted mix of 1–2 randomly chosen fields (see
[fields](#available-fields)). The recipe is saved in the metadata JSON, so you can re-render any fractal.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--mode` | *(required)* | `legacy`, `2dt` or `3dt` |
| `--n` | *(required)* | Number of fractals to generate. In `2dt` each fractal writes `--slices` files. |
| `--out` | *(required)* | Output directory (created if missing) |
| `--seed` | `0` | Seed for the numpy generator that picks `c` values and recipes. The same seed gives the same fractals. |
| `--size H W` | `256 256` | In-plane grid size |
| `--frames` | `32` | Number of time frames `T` |
| `--slices` | `10` | Recipe modes only: number of slices `Z` |
| `--max-iter` | escape: `50`, recipe: `30` | Maximum Julia iterations. Higher values give finer detail and take longer. |
| `--norm` | `frame` | Recipe modes only. `frame` min-max normalises every 2D frame separately (legacy behaviour; intensity flickers over time and slices). `volume` normalises over the whole block. **Use `volume` for 3D+t.** |
| `--variety` | `none` | Recipe modes only: per-phantom variety steps, comma-separated or `all` (see [below](#phantom-variety)) |
| `--gifs` | `0` | Write preview GIFs for the first N files |

### Output files

| Mode | Files |
|---|---|
| `legacy` | `<c0>_<c1>_<c2>_<c3>_<NNNN>.npy` (named after the Julia `c`) + `.json` |
| `2dt` | `julia_<KK>_slice_<S>.npy` + `.json` (S = 1…Z) |
| `3dt` | `julia_<KKK>.npy` + `.json` |
| GIFs | `<name>.gif` for 3D arrays; `<name>.mosaic.gif` (all slices tiled) and `<name>.ortho.gif` (three orthogonal views) for 4D volumes |

The JSON holds the mode, the domain (grid bounds), the normalisation and the recipe or `c` value.

> `legacy` mode stops after `n × 100` attempts, so if most candidates fail the complexity check it can
> write fewer than `n` files. The script prints how many it wrote.

### The sampling domain

The `2dt` and `3dt` modes sample 4D quaternion space `(w, x, y, z)` as follows (`FractalDomain` in
`synthmr/phantoms/generate.py`):

| Array axis | Quaternion component | Range |
|---|---|---|
| rows (H) | x | −1.2 … 1.2 |
| columns (W) | w | −1.6 … 1.6 |
| time (T) | z | −0.3 … 0.3 |
| slices (Z) | y | centred on 0, step 0.1 |

The bounds, time range and slice step are not CLI options. To change them, call `generate()` from Python
with your own `FractalDomain` (see [§9](#9-working-with-fractals-directly)). `FractalDomain.isotropic(n, extent,
frames)` gives the `n^3` grid used by `scripts/make_cone_pairs.py`.

### Phantom variety

With the fixed domain above, about 60% of voxels escape within two iterations, so they are flat background.
`--variety` (`Variety` in `synthmr/phantoms/variety.py`) changes how each recipe is framed. Every step is off by
default:

| Step | Effect |
|---|---|
| `frame` | Zoom into a random window (0.4–0.75 of the domain) centred on a boundary point of the set. The scale applies to all four components, so the slice spacing and the time sweep shrink with the field of view. |
| `reject` | Reframe (up to 8 tries), then redraw `c` (up to 10 times), until the boundary band `2 < escape < max_iter` covers at least 60% of a coarse preview |
| `background` | Each real/imag channel gets a `distance` or `orbit_trap` field, which varies outside the set. Every field is gamma-balanced so that its median maps to 0.5. |
| `warp` | Smooth fBm domain warp (strength 0.15 of the window half-width) |
| `permute` | Random assignment of the quaternion components to the axes (T, Z, H, W) |
| `rotate` | Random 4D rotation of the sampling grid (oblique sections; includes `permute`) |

The framing is saved in the recipe (`recipe["framing"]`), so `render_recipe` re-renders it from the JSON.
Each step draws from its own child of the seed, so enabling a step does not change `c`, the field mix or
the other steps' draws. The exception is a phantom that `reject` redraws, and that redraw does not affect other
phantoms. `python scripts/review_variety.py` writes GIFs that add the steps one
at a time.

## 4. Step 2: making a dataset

```bash
python scripts/make_dataset.py --preset NAME --input DIR --output DIR [options]
```

The script reads every `*.npy` in `--input` in sorted order and runs it through the pipeline. Each result
is saved under the same file stem in `--output`.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--preset` | none (plain defaults) | Named preset to start from (see [§5](#5-presets)) |
| `--config` | none | A JSON or YAML file of overrides applied on top of the preset |
| `--input` | *(required)* | Directory of phantom volumes (`*.npy`) |
| `--output` | *(required)* | Output directory (created if missing) |
| `--limit` | all | Process only the first N files (in sorted order) |
| `--gifs` | from config (`0`) | Write input/target GIFs for the first N samples (sets `output.gifs`) |
| `--format` | from config | `npy` or `h5` (sets `output.format`) |
| `--device` | backend default | Device to run on, e.g. `gpu:0`, `gpu:1` (sets `device`) |
| `--backend` | `$SYNTHMR_BACKEND`, else `tf` | `tf` or `sigpy` (sets `backend`) |
| `--rng` | `backend` | `backend` or `numpy` random streams (sets `rng`, see [§11](#11-reproducibility-and-random-seeds)) |
| `--list-presets` | | Print the preset names and exit |
| `--print-config` | | Print the fully resolved configuration as JSON and exit. Use it to check what your overrides do. |

Settings are applied in this order, later ones winning: **preset → `--config` file → command-line
flags**.

### Override files

An override file contains only the keys you want to change. Nested sections are merged, so anything you
leave out keeps its preset value. A misspelt key raises `KeyError: ... unknown keys [...]`, so typos are
caught.

```yaml
# more_spokes.yaml
simulation:
  num_coils: [8, 16]        # random number of coils per sample, 8..16
  regsnr: [15, 30]          # SNR between 15 and 30 dB
trajectory:
  spokes: 21
augment:
  z_flip: 1
seeds:
  transforms: 7
  noise: 8
  augment: 7
```

```bash
python scripts/make_dataset.py --preset fractal_3dt_sos --config more_spokes.yaml --print-config
python scripts/make_dataset.py --preset fractal_3dt_sos --config more_spokes.yaml \
    --input data/julia4d --output data/pairs3d_21spokes
```

The resolved configuration is always saved as `pipeline_config.json` in the output directory. You can pass
that file back to `--config` later to rebuild the same dataset.

## 5. Presets

| Preset | Input volumes | Pipeline | Output |
|---|---|---|---|
| `fractal_2dt_legacy` | `legacy` | Sine-wave complexify (paper method), 15 coils, 13 radial spokes/frame (`sorted_half`), SVD compression to 10 coils | `.npy`, input `(24,256,256,2)`, target `(24,256,256,1)` |
| `fractal_new_2dt` | `2dt` | Complex recipe fractal, 10 coils, 13 golden-angle spokes/frame (`golden_half`), compression to 10 coils | `.npy`, same shapes |
| `fractal_new_2dt_nocc` | `2dt` | As `fractal_new_2dt` but **without** coil compression | `.npy`, same shapes |
| `fractal_3dt_sos` | `3dt` | Volume-consistent 3D simulation (10 slices, spacing 10 px), 10 coils, stack-of-stars (13 golden spokes per kz partition, Cartesian kz), no compression, 3D coil maps from time-averaged Tukey-filtered data | `.h5`, input `(24,10,256,256,2)`, target `(24,10,256,256,1)` |

The three 2D presets reproduce the original scripts bit for bit (checked by the regression tests).
`fractal_3dt_sos` is new.

To see a preset's full settings:

```bash
python scripts/make_dataset.py --preset fractal_new_2dt --print-config
```

To add your own preset, see [§8](#adding-a-preset).

## 6. Configuration reference

A configuration (`PipelineConfig`) has these sections. Defaults below are the dataclass defaults, which
apply when no preset is given; presets override many of them.

```
name  input  simulation  trajectory  recon  augment  output  seeds  device
```

> **Keep shared sizes consistent.** `base_resolution` appears in `simulation`, `trajectory` and `recon`,
> and `phases` in all three as well. If you change one, change all three. For 3D, `simulation.slices` must
> match the `Z` of your volumes (`--slices` at generation time). A mismatched volume is centre-cropped or
> zero-padded.

### Top level

| Key | Default | Meaning |
|---|---|---|
| `name` | `"custom"` | Label stored in each sample's metadata |
| `seeds` | `{transforms: 1, noise: 2, augment: 1, numpy: 0}` | Seeds for the random streams (see [§11](#11-reproducibility-and-random-seeds)) |
| `device` | `null` | Device: `"gpu"`, `"gpu:1"`, `"cpu"`. `null`: TF chooses; sigpy uses GPU 0 if cupy finds one, else the CPU. |
| `backend` | `null` | `tf` or `sigpy`. `null` uses the active backend (`$SYNTHMR_BACKEND`, else `tf`). |
| `rng` | `"backend"` | Random streams. `backend`: the backend's generator (TF: philox, reproduces the original code). `numpy`: numpy philox streams, identical on every backend (use it to compare TF and sigpy). |

### `input`: from stored volume to video

| Key | Default | Meaning |
|---|---|---|
| `complexify` | `"complex_to_video"` | How a stored volume becomes the 2-channel [0, 255] video (see below) |
| `legacy_global_np` | `false` | Draw complexify randomness from numpy's global state, as the original code did. `fractal_2dt_legacy` sets this to reproduce the paper exactly. |

| `complexify` value | Expects | Does |
|---|---|---|
| `complex_to_video` | complex volume in [-1, 1] (recipe modes) | real → ch0, imag → ch1, rescaled to [0, 255] |
| `sine_wave` | escape-time volume | Paper method: normalise, map through two random sine waves (one for real, one for imaginary), blur and sharpen each |
| `sine_wave_smooth` | escape-time volume | Variant: inverted normalisation, Gaussian blur, fixed frequency 0.25 |
| `sine_wave_spiral` | escape-time volume | Variant: one sine-mapped channel copied into both real and imaginary |
| `video` | an existing `(…, 2)` video in [0, 255] | Passed through unchanged (use this for your own image data) |

### `simulation`: image to multicoil k-space

| Key | Default | Meaning |
|---|---|---|
| `mode` | `"2d"` | `2d`: per-frame 2D simulation (original). `3d`: one consistent simulation of the whole `(T, Z, H, W)` volume (one 3D mask, one 3D phase map, 3D coils, one SNR). |
| `base_resolution` | `256` | In-plane size; the video is centre-cropped or zero-padded to this |
| `phases` | `32` | Number of frames kept from the video (centre crop or zero-pad in time) |
| `slices` | `1` | `3d` only: number of slices Z |
| `slice_spacing_px` | `10.0` | `3d` only: distance between slices in in-plane pixel units. Sets how far coils, the object mask and the background phase extend along z. |
| `masking` | `true` | Multiply the object by a random filled ellipse (2D) / ellipsoid (3D), also applied to the coil maps |
| `image_adjustment` | `[0, 1]` | Contrast `[offset, gain]`: pixel → clip(offset + gain·pixel, 0, 1) |
| `complex_transform` | `1` | `> 0`: treat the two channels as real/imag (randomly swapped) and multiply the phase by this factor (`2` doubles phase wrapping). `0`: use only the magnitude (real-valued object). |
| `add_phase` | `2` | `> 0`: add smooth random background phase to the object and to every coil map. The value scales the coil phase. `0` disables both. |
| `object_phase_ngrid` | `6` | Coarseness of the object background phase: a random `n × n` grid smoothly upsampled. Smaller means smoother. |
| `coil_phase_ngrid` | `2` | Same, for each coil's phase |
| `num_coils` | `[15]` | `[n]` fixed, or `[lo, hi]`: a random count per sample |
| `sigma_coil` | `[2, 6]` | Coil width as `base_resolution / sigma`. Larger sigma means smaller, more local coils. One value is fixed; two values give a random range per coil and per axis. |
| `coil_size_factor` | `2` | `2d` only: coil kernel support = factor × base_resolution |
| `regsnr` | `[12, 22]` | Noise. `[lo, hi]` draws a random SNR per sample; `[x]` is fixed. Values > 1 are SNR in dB; values ≤ 1 are the noise standard deviation; `0` turns noise off. |
| `extra` | `{}` | Free-form; not used by the pipeline |

### `trajectory`: k-space sampling

Each in-plane encoder has its own config class (`RadialTrajectoryConfig`, `SpiralTrajectoryConfig` in
`synthmr.sampling`), so a config only holds the params of the encoder it uses. `kind` selects the class.
Changing `kind` in an override resets the encoder params to that kind's defaults; the shared keys are kept.

Shared keys:

| Key | Default | Meaning |
|---|---|---|
| `kind` | `"radial"` | In-plane trajectory: `radial`, `spiral` (see note) or `file` (load from `.npy`) |
| `ordering` | `"tiny_half"` | Spoke angle ordering, passed to `tensorflow_mri.sampling.radial_trajectory`. Common values: `golden_half` (golden angle, half circle), `tiny_half` (tiny golden angle), `sorted_half` (evenly spaced, rotated per frame). Spiral uses e.g. `golden`, `tiny`. |
| `phases` | `32` | Number of distinct trajectory frames (repeated in time if the recon needs more) |
| `base_resolution` | `256` | Readout resolution |
| `reverse` | `true` | Swap the two trajectory coordinates (kept for compatibility with the original code) |
| `through_plane` | `"none"` | `none`: 2D (needs Z = 1). `cartesian_kz`: stack-of-stars. The z-direction is fully sampled by FFT and every kz partition uses the in-plane trajectory. Required for 3D volumes. |
| `kz_rotation` | `"aligned"` | `cartesian_kz` only. `aligned`: every partition uses the same spokes (classic stack-of-stars). `golden`: partition k is rotated by k × golden angle, which spreads spokes through kz. |

`kind: radial` keys:

| Key | Default | Meaning |
|---|---|---|
| `spokes` | `13` | Spokes per frame. This sets the acceleration: full radial sampling at 256 needs ~400 spokes, so 13 is very undersampled. |

`kind: spiral` keys:

| Key | Default | Meaning |
|---|---|---|
| `field_of_view` | `400` | FOV in mm |
| `vd_spiral_arms`, `min_max_arm_time`, `max_tempres`, `vd_*`, `max_grad_ampl`, `min_rise_time`, `dwell_time`, … | *(see `SpiralTrajectoryConfig`)* | Spiral design parameters: arms, variable-density cutoffs, gradient limits, dwell time, arm-duration window, … |

`kind: file` keys (trajectory and density from your own `.npy` files; works with every backend):

| Key | Default | Meaning |
|---|---|---|
| `traj_path` | `""` | Trajectory, last axis = 2 coordinates in image-axis order `(y, x)`. Shapes `(R, 2)`, `(T, R, 2)`, `(T, V, S, 2)`, or `(V, S, 2)` with `views_axis: true`. |
| `density_path` | `""` | Density (or weights) with the trajectory's shape minus the last axis |
| `traj_units` | `"radians"` | `radians`: [-π, π] (tfmri). `pixels`: [-N/2, N/2] with N = `base_resolution` (sigpy / extreme_mri `coord`). |
| `density_kind` | `"density"` | `density`: weights = 1/density (as for generated radial trajectories). `dcw`: the file already holds density-compensation weights. |
| `views_axis` | `false` | Read a 3-D array as one frame of `(V, S, 2)` instead of `(T, R, 2)` |
| `reverse` | `false` | Swap the two coordinates (generated trajectories default to `true`) |

If the file has fewer frames than `phases`, they are repeated in time.

> **Spiral** trajectories (`kind: spiral`) need the `spiral-waveform` library, which the TF devcontainer
> installs. The trajectory builder supports them, but the pipeline is only tested end to end with radial
> trajectories. Treat spiral as experimental. The sigpy backend can't generate spirals; save one (e.g. from
> the TF container) and load it with `kind: file`.

### `recon`: undersampling and building the pair

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `base_resolution` | `256` | both | Image size of the reconstruction |
| `phases` | `32` | both | Frames reconstructed; the simulated frames are repeated if fewer |
| `input_format` | `"coil_compressed"` | both | `coil_compressed`: SVD coil compression (fitted on the undersampled data) to `input_coils` virtual coils, input as 2 channels (real, imag). `abs`: no compression, input as 1-channel magnitude. Any other value (e.g. `complex`): no compression, 2 channels. |
| `input_coils` | `10` | both | Number of virtual coils after compression |
| `output_format` | `"abs"` | both | `abs`: target is 1-channel magnitude. `null`: target as 2 channels (real, imag). |
| `normalize_input` | `true` | both | Min-max scale the zero-filled coil images before combining |
| `coil_maps` | `"walsh_xyt_legacy"` | | How coil sensitivities for combining the input are estimated. `walsh_xyt_legacy` (**2D only**): Walsh on `(H, W, T, C)`, i.e. time is treated as a third spatial axis (original behaviour). `time_avg_tukey_3d` (**3D only**): sum all frames, low-pass with a 3D Tukey window in k-space, then 3D Walsh. `jsense` (**sigpy backend**, 2D and 3D): JSENSE (`sigpy.mri.app.JsenseRecon`) on the non-Cartesian data of all frames; one time-independent set of maps. |
| `jsense_ker_width`, `jsense_calib_width`, `jsense_lamda`, `jsense_max_iter`, `jsense_max_inner_iter` | `16`, `24`, `0`, `10`, `10` | `jsense` | JSENSE map kernel width, k-space calibration width (pixels), regularisation and iterations |
| `tukey_width` | `[1.0, 0.25, 0.25]` | 3D | Fraction of k-space kept along (Z, Y, X) for the coil-map low-pass |
| `tukey_alpha` | `0.5` | 3D | Taper of the Tukey window (0 = box, 1 = Hann) |
| `anchor_phase` | `true` | 3D | Reference the coil-map phase to coil 0. This makes the input phase reproducible. The 2D path is unanchored, as in the original. |
| `input_combine` | `"maps"` | both | `maps`: the zero-filled input is coil-combined with the estimated maps (1 complex channel). `none`: coils are kept and folded into the channel axis as `[re_1 … re_C, im_1 … im_C]` (2C channels; C channels for `input_format: abs`). C is `input_coils` with compression, else the simulated coil count. Maps are still estimated and returned in `rec["maps"]`. |
| `target_combine` | `"sos"` | 3D | `sos`: target is the root-sum-of-squares magnitude. `maps`: target combined with the same coil maps as the input, which keeps its phase (pair with `output_format: null` for a complex target). In 2D the target is always sum-of-squares. |
| `gfilt` | `null` | 2D | If set, Gaussian-blur sigma applied to the target (legacy option) |
| `roll` | `0` | | Must stay `0` (temporal rolling is done in `augment`) |

Scaling: the target is divided by its maximum, so its peak is 1. The input is scaled with the same factor
before `normalize_input`.

### `augment`: shared random transforms

| Key | Default | Meaning |
|---|---|---|
| `phases` | `24` | Frames in the final sample. The first `phases` frames are kept after rolling. |
| `rotation` | `0` | `> 0`: rotate input and target by the same random multiple of 90° in-plane |
| `roll` | `1` | `> 0`: circularly shift input and target by the same random number of frames. Needs `recon.phases > augment.phases`. |
| `z_flip` | `0` | 3D only. `> 0`: flip the slice order with probability ½. |

In 3D, one rotation, flip and roll are drawn per sample and applied identically to every slice and frame.
NaNs are replaced by 0.

### `output`

| Key | Default | Meaning |
|---|---|---|
| `format` | `"npy"` | `npy`: `<stem>_input.npy` and `<stem>_output.npy`. `h5`: one `<stem>.h5` with datasets `input` and `output`. |
| `gifs` | `0` | Write GIFs for the first N samples |
| `save_meta` | `true` | Write `<stem>.json` with the preset name, seeds and source file |

## 7. Output files and array shapes

For an output directory `data/pairs`:

```
data/pairs/
  pipeline_config.json          the resolved configuration used for the whole run
  <stem>_input.npy              (npy format) network input
  <stem>_output.npy             (npy format) target
  <stem>.h5                     (h5 format) datasets "input" and "output"
  <stem>.json                   metadata: {"preset", "seeds", "source"}
  gifs/<stem>/pair.gif          2D+t: input | target, animated over time
  gifs/<stem>/input_mosaic.gif  3D+t: all slices tiled, animated over time
  gifs/<stem>/target_mosaic.gif
  gifs/<stem>/target_ortho.gif  3D+t: axial / XZ / YZ views through the centre
```

| | 2D+t | 3D+t |
|---|---|---|
| input | `(T, H, W, ch)` float32 | `(T, Z, H, W, ch)` float32 |
| target | `(T, H, W, 1)` or `(T, H, W, 2)` | `(T, Z, H, W, 1)` or `(…, 2)` |
| defaults | T = 24, H = W = 256, ch = 2 (real, imag) | T = 24, Z = 10, H = W = 256, ch = 2 |

Channel order is always `[real, imag]` for 2-channel arrays. By default the input is the **coil-combined
zero-filled reconstruction**. With `recon.input_combine: none` it keeps the coils instead: ch = 2C, ordered
`[re_1 … re_C, im_1 … im_C]` (all real parts, then all imaginary parts), or ch = C for `input_format: abs`.
The target is always the **fully sampled coil-combined image** (magnitude by default).

Loading:

```python
from synthmr.pipeline import load_sample
x, y = load_sample("data/pairs", "julia_00_slice_1", fmt="npy")
x, y = load_sample("data/pairs3d", "julia_000", fmt="h5")
```

## 8. Python API

### Running the pipeline

```python
import numpy as np
from synthmr.pipeline import Pipeline, load_config, save_sample

# preset <- optional override file <- optional dict of overrides
cfg = load_config("fractal_new_2dt", path=None, overrides={"simulation": {"regsnr": [20]}})

pipe = Pipeline(cfg)
sample = pipe.run(np.load("data/julia/julia_00_slice_1.npy"))
sample.input, sample.target, sample.meta

save_sample(sample, "data/pairs", "my_sample", fmt="npy", save_meta=True)
```

`Pipeline.run(volume, return_stages=True)` also fills `sample.stages`, a dict of the intermediate arrays.
Use it for debugging or figures:

| Key | Content |
|---|---|
| `video`, `video_uint8` | the complexified 2-channel video |
| `sim_kspace` | fully sampled multicoil k-space `(C, T, Z, H, W)` |
| `sim_smaps` | true coil sensitivities `(C, Z, H, W)` |
| `sim_image` | object with background phase `(T, Z, H, W)` |
| `sim_mask` | object mask |
| `uskspace` | undersampled k-space |
| `traj`, `dcw` | trajectory (radians, in [-π, π]) and density compensation weights |
| `maps` | estimated coil maps |
| `zfill`, `image` | input and target before augmentation |

Create one `Pipeline` for many volumes, not one per volume. It keeps its random streams running from sample
to sample, so every sample gets different coils, noise and augmentation.

### Configs in code

```python
from synthmr.pipeline import PipelineConfig, get_preset, PRESETS

cfg = get_preset("fractal_3dt_sos")
cfg2 = cfg.updated({"trajectory": {"kz_rotation": "golden"}})   # returns a new config
cfg.to_dict()                                                    # plain dict (JSON-serialisable)
PipelineConfig.from_dict(d)                                      # back again; rejects unknown keys
sorted(PRESETS)                                                  # preset names
```

You can also edit the dataclasses directly (`cfg.recon.input_coils = 8`).

### Adding a preset

```python
from synthmr.pipeline import register_preset, get_preset

@register_preset
def my_3dt_fast():
    cfg = get_preset("fractal_3dt_sos")
    cfg.name = "my_3dt_fast"
    cfg.trajectory.spokes = 34
    cfg.simulation.num_coils = (8,)
    return cfg
```

The preset name is the function name. To make it available to `make_dataset.py`, define it in
`synthmr/pipeline/presets.py` and add it to the `PRESETS` dictionary there.

### Lower-level stages

The stages can be called on their own if you want to swap one out:

```python
from synthmr.rng import RNG
from synthmr.simulation import simulate
from synthmr.sampling import make_trajectory
from synthmr.recon import reconstruct, augment

rng = RNG({"transforms": 1, "noise": 2, "augment": 1})
sim = simulate(video_uint8, cfg.simulation, rng)                  # dict: kspace, smaps, image, mask
traj, dcw = make_trajectory(cfg.trajectory)
rec = reconstruct(sim["kspace"], traj, dcw, cfg.recon,
                  through_plane=cfg.trajectory.through_plane, kz_rotation=cfg.trajectory.kz_rotation)
zfill, target = augment(rec["zfill"], rec["image"], cfg.augment, rng)
```

### Backend

All MRI operations (FFT, NUFFT, coil compression, Walsh maps, …) go through a backend:

- `tf` (default): TensorFlow / tensorflow-mri, reproduces the original code exactly.
- `sigpy`: numpy (CPU) or cupy (GPU) with sigpy NUFFTs; needs no TensorFlow. Same stages and same results up
  to float rounding (see [BACKEND.md](BACKEND.md) for measured differences). It adds `recon.coil_maps: jsense`
  and lacks generated spirals and `LowPassScaler`.

Select one per pipeline with `backend` in the config (`--backend`), or globally with `$SYNTHMR_BACKEND` /
`synthmr.backend.set_backend("sigpy")`. `register_backend(name, module_path)` adds another backend. See
[BACKEND.md](BACKEND.md) for what a backend must provide.

## 9. Working with fractals directly

Everything in `synthmr.phantoms` is numpy/numba and needs no GPU.

### Batch generation with a custom domain

```python
from synthmr.phantoms.generate import generate, FractalDomain, LegacyDomain

dom = FractalDomain(shape=(192, 192), frames=40, slices=16, slice_step=0.05,
                    time_range=(-0.5, 0.5), bounds=((-1.2, 1.2), (-1.6, 1.6)), max_iter=40)
paths = generate("3dt", n=20, out_dir="data/custom", seed=3, domain=dom, norm="volume")
```

`LegacyDomain(shape, extent, time_extent, frames, max_iter, complexity_threshold)` configures the `legacy`
mode the same way.

### Rendering a single fractal

```python
from synthmr.phantoms.fractal import QuaternionJulia, JuliaConfig, Slice

julia = QuaternionJulia(JuliaConfig(c=(-0.11, 0.83, -0.11, 0.32), max_iter=50))
img = julia.render(shape=(256, 256), slice_=Slice(axes=(1, 2), fixed=(0, 0, 0, 0.1)), field="distance")
stack = julia.render_stack(["smooth_escape", "orbit_trap"], shape=(256, 256))   # (2, H, W)
```

`Slice` picks two quaternion components `(w, x, y, z) = (0, 1, 2, 3)` as image axes, with bounds, and holds
the other two at `fixed`.

### N-D grids

`Grid` generalises `Slice` to any number of axes. Each output axis maps to one quaternion component:

```python
from synthmr.phantoms.fractal import Grid, Axis, JuliaConfig, evaluate_field

grid = Grid(axes=[Axis.linspace("t", 3, -0.3, 0.3, 32),     # time   <- z
                  Axis.centred("z", 2, 10, 0.1),            # slices <- y
                  Axis.linspace("y", 1, -1.2, 1.2, 256),    # rows   <- x
                  Axis.linspace("x", 0, -1.6, 1.6, 256)])   # cols   <- w
field = evaluate_field(grid.coordinates(), "smooth_escape", {"config": JuliaConfig(c=(-0.2, 0.6, 0.2, 0.2))})
field.shape   # (32, 10, 256, 256)
```

### Available fields

| Field | Description | Extra params |
|---|---|---|
| `escape` | Integer escape iteration count | |
| `smooth_escape` | Continuous (smoothed) escape count | |
| `distance` | Distance estimate to the set | |
| `orbit_trap` | Minimum distance of the orbit to a point, line or plane | `trap` (`point`/`line`/`plane`), `axes` |
| `power` | Escape count for z → z^p + c | `power` (int) |
| `perturbed` | Escape count with an ε·sin(z) perturbation | `epsilon` |
| `internal_detail` | Smooth escape outside the set, internal structure inside | |

Field parameters that apply everywhere live in `JuliaConfig`: `c`, `max_iter` (50), `bailout` (4.0),
`power` (2), `epsilon` (0.01), `trap` (`plane`), `trap_axes` (`(0,)`).

### Custom fields

```python
from synthmr.phantoms.fractal import register_field

@register_field("my_field")
def my_field(q, config, scale=1.0):
    """q: one quaternion (w, x, y, z); return a float."""
    ...
```

Custom fields are evaluated point by point in Python, which is much slower than the built-in compiled
fields. To include a field in random recipes, add a parameter sampler to
`synthmr.phantoms.recipes.FIELD_CHOICES`. This changes which recipes a given seed produces.

### Recipes, mixing and helpers

- `recipes.random_recipe(rng, c)`, `recipes.render_recipe(recipe, grid, max_iter, norm_axes)`: draw and
  render recipes. `render_recipe` re-renders a fractal from its saved JSON recipe.
- `fractal.compose_complex(real_fields, imag_fields, real_weights, imag_weights, axes)`: mix rendered fields
  into a complex image.
- `fractal.candidate_c(n, resolution, extent, min_iter, max_escape)`: Julia `c` values near the quaternion
  Mandelbrot boundary, where Julia sets are most detailed.
- `fractal.domain_warp(coords, strength, scale, octaves, …)`: warp grid coordinates with smooth noise
  before evaluating, for more organic shapes.
- `complexify.complex_to_video`, `complexify.sine_wave`, …: the complexify methods from [§6](#input-from-stored-volume-to-video).

## 10. Visualisation

`synthmr.viz.gifs` writes animated GIFs (time on axis 0, complex arrays shown as magnitude):

| Function | Shows |
|---|---|
| `save_gif(frames, path, title)` | one `(T, H, W)` series |
| `save_panel_gif([a, b, …], titles, path)` | several series side by side |
| `save_comparison_gif(old, new, path)` | `old \| new \| \|diff\|` |
| `save_slice_mosaic_gif(volume, path)` | `(T, Z, H, W)` with all slices tiled |
| `save_ortho_gif(volume, path)` | `(T, Z, H, W)` as axial / XZ / YZ views |
| `save_sample_gifs(sample, out_dir)` | the input/target GIFs that `make_dataset.py --gifs` writes |

## 11. Reproducibility and random seeds

Randomness comes from four independent, seeded streams:

| Stream | Default seed | Controls |
|---|---|---|
| `transforms` | 1 | Mask, coils, background phase, channel swap, SNR level |
| `noise` | 2 | The noise values |
| `augment` | 1 | Rotation, z-flip, temporal roll |
| `numpy` | 0 | Random draws in the `sine_wave*` complexify methods |

- A `make_dataset.py` run is reproducible: the same inputs, config and seeds give the same outputs.
- The streams run on from file to file, so a sample depends on how many files came **before** it in that run
  (sorted order, `--limit`). To change the random draws for a whole dataset, change `seeds`.
- Because the noise stream is separate, re-seeding `transforms` and `augment` (for example to give two
  datasets the same coils and augmentations) leaves the noise independent. In Python:
  `pipe.rng.reset(transforms=5, augment=5)`.
- Known exception: in the 2D path the input's coil-map phase is not anchored. On GPU it can flip sign at
  isolated pixels from run to run; the magnitude is stable. The 3D path anchors the phase
  (`recon.anchor_phase`).
- Fractal generation is separate and reproducible through `--seed`.
- With `rng: backend` (default) the TF and sigpy backends draw different random numbers. Set `rng: numpy`
  (`--rng numpy`) to make every backend draw the same ones: the same config and seeds then give the same
  dataset on TF and sigpy, up to float rounding and the arbitrary coil-map phase.

## 12. Training a network

`train_network.ipynb` is a complete example:

1. Builds a 2D+t dataset with the `fractal_2dt_legacy` preset (escape-time fractals → sine-wave complexify →
   pipeline → `.npy` pairs).
2. Splits the files into train, validation and test sets and wraps them in a `tf.data` generator.
3. Trains a `tensorflow_mri` 3D UNet with an SSIM loss and PSNR/SSIM metrics.

Set `pretrained_weights` in the notebook to start from the weights in `trained_models/` (`F-DL.h5`,
`CMR-DL.h5`).

## 13. Running the tests

```bash
python -m pytest tests -q -p no:warnings        # ~1 min on GPU
```

- `tests/unit/`: backend contract, grids, sampling, generator and 3D properties.
- `tests/regression/`: compares the pipeline with stored "golden" outputs of the original code. These tests
  are skipped until you capture the golden outputs once on a GPU:
  `python tests/regression/capture_golden.py --device gpu`. See [REGRESSION.md](REGRESSION.md).
- In the sigpy container the same command runs only the backend-agnostic tests (TF-only modules aren't
  collected). `tests/regression/test_xbackend.py` compares sigpy with TF outputs captured by
  `python tests/regression/capture_xbackend.py` in the TF container. Copy `tests/artifacts/xbackend/`
  across.

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Segfault, or `Eigen decomposition was not successful` during `make_dataset.py` | Running on CPU. The Walsh coil-map estimation needs a GPU with this TF version. |
| `KeyError: PipelineConfig: unknown keys [...]` (or `SimulationConfig`, …) | Typo or wrong section in an override file or dict |
| `through_plane='none' needs Z=1; use 'cartesian_kz' for 3D+t` | A `(T, Z, H, W)` volume with a 2D preset. Use `fractal_3dt_sos`, or generate with `2dt`. |
| `3D path supports coil_maps='time_avg_tukey_3d'` / `2D path supports coil_maps='walsh_xyt_legacy'` | `recon.coil_maps` doesn't match the data's dimensionality |
| `NotImplementedError: roll>0 in recon ...` | Set `recon.roll: 0`; use `augment.roll` instead |
| 2D: error while drawing the temporal roll | `augment.roll > 0` needs `recon.phases` greater than `augment.phases` (3D skips the roll instead) |
| Zero-filled or missing slices in 3D | `simulation.slices` doesn't match the volume's Z (it is centre-cropped or zero-padded); regenerate with matching `--slices` or set `simulation.slices` |
| Intensity flickers between frames/slices | Volumes generated with `--norm frame`; use `--norm volume` for 3D+t |
| Fewer escape fractals than `--n` | Too many candidates failed the complexity check; run again with another `--seed` or a larger `--n` |
| `NotImplementedError: spiral trajectories need the TF backend` | sigpy can't generate spirals; use `trajectory.kind: file` with a saved trajectory |
| `coil_maps='jsense' needs the sigpy backend` | JSENSE is sigpy-only; use `--backend sigpy` |
| `trajectory exceeds [-pi, pi]` | A `kind: file` trajectory in pixel units read as radians (set `traj_units: pixels`), or the wrong `base_resolution` |
| sigpy 3D `coil_maps: jsense` is very slow | On CPU, sigpy's 3D gridding is single-threaded; run on a GPU (cupy) |
| GPU memory taken by another process | `make_dataset.py` turns on TF memory growth; pick a free GPU with `--device gpu:1` |
