# Data conventions

## Canonical internal layout
- Complex image: `(C, T, Z, Y, X)`; 2D+t uses `Z = 1`; single-coil uses `C = 1`.
- Cartesian k-space: same layout. Non-Cartesian k-space: in-plane `(Y, X)` replaced by samples `R`.
- Trajectory: `(T, R, 2)` in radians `[-pi, pi]`; dcw `(T, R)`.

## Legacy on-disk formats (reproduced by I/O adapters)
| Array | Shape | Notes |
|---|---|---|
| video into simulation | `(T, H, W, 2)` uint8 | ch0 = real, ch1 = imag |
| simulated k-space | `(1, T, C, H, W)` complex64 | leading "slice" dim |
| `*_input.npy` (zfill) | `(T, H, W, 2)` float32 | Walsh-combined, real/imag; after roll/rotation, 24 frames |
| `*_output.npy` (target) | `(T, H, W, 1)` float32 | magnitude (`output_format='abs'`) |
| 3D+t (new) | `(T, Z, H, W, ch)` in h5 `input`/`output` | + meta.json |
| uncombined input (`recon.input_combine='none'`) | `(T, H, W, 2C)` / `(T, Z, H, W, 2C)` | `[re_1..re_C, im_1..im_C]`; `C` channels if `input_format='abs'` |

## 3D+t specifics
- Slices are 2D planes stacked along Z. `SimulationConfig.slice_spacing_px` gives the slice spacing in in-plane
  pixel units (default 10, close to the fractal generator's y-step / pixel size). It sets the z-extent of the coil
  Gaussians, the ellipsoid mask and the background-phase grid.
- Fractal intensity normalisation: legacy normalises each 2D frame separately (`norm_axes=(-2, -1)`),
  which makes intensities flicker across time and slices. 3D+t uses `norm="volume"` (whole block).
- Coil maps (3D): `(Z, H, W, C)`, phase referenced to coil 0, so input/target phase is reproducible.
- Recon outputs before augmentation: zfill `(T, Z, H, W, ch)`, target `(T, Z, H, W)` (abs) or `(T, Z, H, W, 2)`.

## Fractal quaternion conventions
- New framework: `(w, x, y, z)`, Hamilton product in `_kernels._qmul`.
- Legacy `julia_escape`: grid dim 0 is the real part; same product. Grid `build_grid([1,1,0,0.2],[256,256,1,32])`
  = in-plane on quaternion axes 0/1, axis 2 fixed at 0, axis 3 swept over 32 frames (time).
- `julia_generator_new_grasp.py`: rows = x, cols = w, quaternion z sweep (32) = time, y (10) = slices.

## Normalisation (legacy behaviour)
- Video scaled min-max to [0, 1] before complex transform.
- Coil images divided by max SoS; target divided by max |coil-combined fully-sampled image|.
- zfill min-max scaled (`normalize_input=True`) before Walsh combination.

## Gotchas found
- tfmri `estimate_sensitivities` is called on zfill shaped `(H, W, T, coils)`, so Walsh treats **time as a
  third spatial axis** (3D uniform filter over x, y, t). It is not per-frame 2D.
- Walsh eigenvector phase is not anchored: on GPU, run-to-run sign/phase flips at isolated pixels of the
  combined zfill (magnitude is stable to ~4e-6).
- Three different `make_complex_fractal` variants exist (base repo, app/process_data.py, app/process_data_old.py).
  The base-repo one (sine waves + blur/sharpen) is the paper version and fixture A's reference.
- `input_format='blah'` in `process_fractals_new_grasp_grouped` silently disables coil compression.
