<!-- Hand-off doc. Written 2026-10-19 in the CPU-only container. After rebuilding into the sigpy (GPU) devcontainer,
point an agent at this file and ask it to execute "Implementation (after the rebuild)" through "Verification".
Step 0 below is already DONE. Source: mikgroup/extreme_mri colab-dce-demo.ipynb (demo params: num_ro crop optional,
JSENSE maps, T=100, lamda=1e-8, blk_widths=[8,16,32] — only the gridding step and T=100 binning are used here). -->

# Cone-trajectory paired data from Julia phantoms (one sample for review)

## Context
We want paired training data that uses the real 3D cone trajectory and DCF from the Extreme MRI DCE demo
(`mikgroup/extreme_mri` `colab-dce-demo.ipynb`). The data is at `app/resources/4d_extreme_mri/3D_cone_data/{coord,dcf}.npy`.
The existing phantoms in `app/resources/new_julia_grasp` are `(32, 10, 256, 256)`, which is too anisotropic. User
decisions:
1. Run the **random 3dt generation protocol again** on a new isotropic **64×64×64** grid. Don't re-render the old
   recipes.
2. **Rescale the trajectory** to the 64³ grid.
3. Bin the interleaves into **100 frames** (demo T=100) and use the **first 32** bins, one per phantom frame.
4. Use simulated 3D coils (synthmr).
5. The input is a zero-filled gridding recon only. No MSLR.
6. Save everything as **float16**. Write nothing to local disk: all outputs go to `app/resources/` (NAS).
7. A GPU becomes available once the container is rebuilt. So **step 0 writes a self-contained hand-off doc** for the
   agent that runs after the rebuild.

First milestone: one phantom → one paired sample + review plots. Batch conversion only after the user reviews it.

## Step 0: hand-off doc — DONE (this file)
Write `Image_Reconstruction_Fractal/docs/plans/CONE_PAIRS_PLAN.md`. It lives in the workspace, so it survives the
rebuild. It contains this whole plan plus:
- **Container**: rebuild with `Image_Reconstruction_Fractal/.devcontainer/sigpy` ("Reopen in Container" → *sigpy*).
  That image has `--gpus=all`, sigpy 0.1.27 and cupy-cuda12x (`requirements-sigpy.txt`), and sets
  `SYNTHMR_BACKEND=sigpy`. Check with `nvidia-smi` and
  `python -c "import cupy, sigpy; print(cupy.cuda.runtime.getDeviceCount())"`.
  At 64³ the CPU would also suffice, so the script falls back to the CPU when there is no GPU.
- **NAS**: must be mounted at `/workspaces/mr_recon/app/resources` (docker-compose maps `/media/aviary-nas/anirudh`).
  Check that `app/resources/4d_extreme_mri/3D_cone_data/coord.npy` exists before running.
- Save a memory pointer to this doc.

## Key facts
- `coord (41861, 711, 3)` float32 in sigpy pixel units, with per-axis extents ≈ ±63.2, ±81.8, ±185.5 (native grid
  126×163×370; `coord[..., i]` ↔ image axis `i`). `dcf (41861, 711)`. `ksp` is real data and is not used.
- **Trajectory rescale**: `coord_64 = coord * (64 / N_i)` per axis, with `N = sp.estimate_shape(coord)` =
  (126, 163, 370). Every axis then spans ±32 in 64³ pixel units. Per-axis linear scaling multiplies the sampling
  density by a constant, so `dcf` keeps its shape. The overall scale cancels in the normalisation.
- **Binning**: `41861 // 100 = 418` contiguous interleaves per bin (61 trailing dropped). Bins 0..31 → phantom frames
  0..31, ~297k samples/frame.
- Demo gridding = `sp.nufft_adjoint(ksp * dcf, coord)` with sigpy's default NUFFT (oversamp 1.25, width 4). We use
  the same.

## Implementation (after the rebuild)

### A. Phantom generation: reuse the existing random protocol
`synthmr/phantoms/generate.py::generate(mode="3dt", ...)` draws `random_recipe(rng, candidates[...])` and renders it
with `render_recipe`.
- Small refactor: pull the draw-and-render loop into `iter_recipes_3dt(domain, n, seed, norm)`, a generator that yields
  `(k, recipe, block)`. `generate()` then uses it, with identical draw order and output. The cone script streams
  phantoms from it.
- Domain (isotropic, voxel `v = extent/64`, default `extent = 2.4`, i.e. the old y extent; `--extent` overrides it):
  `FractalDomain(shape=(64, 64), slices=64, slice_step=v, bounds=((-31.5*v, 31.5*v),)*2, frames=32,
  time_range=(-0.3, 0.3), max_iter=30)` with `norm="volume"`. Setting the bounds to ±31.5·v makes the linspace step
  equal `slice_step`, so the voxels are exactly isotropic. Axes are `(T, Z, Y, X)` = quaternion comps (3, 2, 1, 0),
  as before.
- Save each phantom to `app/resources/julia_cone_3dt/julia_000.npy` as float16 `(32, 64, 64, 64, 2)` (re, im), plus
  the recipe/domain JSON via `_write_meta`.

### B. Simulation (numpy, following the `synthmr/simulation/sim3d.py` defaults)
- `rng = synthmr.rng.RNG(kind="numpy")`, seeded per phantom.
- `ellipsoid_mask_3d((64,64,64), gen, slice_spacing=1)` from `synthmr/simulation/masks.py`.
- Object background phase: `background_phase_3d(shape, gen, ngrid=6, slice_spacing=1)` from
  `synthmr/simulation/phase.py`.
- `simulate_coils_3d(gen, shape, sigma_coil=(2,6), num_coils=(12,), add_phase=2, ngrid=2, slice_spacing=1)` from
  `synthmr/simulation/coils.py`, then masked like in sim3d. 12 coils matches the real data.
- Normalise the coil images by their max SoS, as in sim3d. Add complex AWGN to the **k-space samples** at an SNR drawn
  from `regsnr=(12,22)` dB relative to mean signal power.

### C. Sampling and zero-fill: new `Image_Reconstruction_Fractal/scripts/make_cone_pairs.py`
- Device: `sp.Device(0)` (cupy) when available, otherwise CPU. Load `coord`/`dcf` with mmap, rescale them, and slice
  out bins 0..31.
- Per frame `t`, all 12 coils batched: `k = sp.nufft(S * x_t, coord_t)` → add noise →
  `zf = sp.nufft_adjoint(k * dcf_t, coord_t, oshape=(12, 64, 64, 64))`.
- Coil-combine with the true maps: `sum conj(S) zf / sum |S|^2`. This is the formula in
  `synthmr/backend/sigpy/recon3d.py::combine_with_maps`, inlined for the `(C, Z, Y, X)` layout.
- Target = `|x_t · mask|` divided by its global max. Input = the complex combined zero-fill divided by its global
  max |·|.
- CLI: `--n 1 --seed 0 --extent 2.4 --frames-total 100 --frames 32 --out app/resources/new_julia_grasp_cone_paired
  --phantom-out app/resources/julia_cone_3dt --plots --device 0`.

### D. Output (existing 3D+t h5 convention, `docs/DATA_CONVENTIONS.md`, but float16)
`app/resources/new_julia_grasp_cone_paired/julia_000.h5`:
- `input (32, 64, 64, 64, 2)` float16 (re, im).
- `output (32, 64, 64, 64, 1)` float16.
- `julia_000.json` with the meta: recipe, extent/voxel, domain, trajectory scale factors, bins (100, first 32,
  418/frame), coil seed, SNR and normalisation scales.

About 33 MB + 17 MB per sample, so 500 samples is ~25 GB. No disk concern.

### E. Review plots, under `app/resources/new_julia_grasp_cone_paired/review/`
- `phantom_ortho.png`: |·| and phase on the central axial/coronal/sagittal planes, frames 0/16/31.
- `pair_ortho.png`: target, |input|, ∠input and |error| on the three planes, frames 0 and 16.
- `coils.png`: |S| and ∠S for the 12 coils on the central plane, plus SoS.
- `trajectory.png`: a subset of one frame's rescaled cone interleaves in 3D, plus k-space coverage projections for one
  frame vs all 32.
- `pair_time.gif`: central slices over time, target | |input| (`synthmr/viz/gifs.save_panel_gif`).

## Verification
1. Check the env (GPU, sigpy/cupy, NAS mount).
2. Confirm the `generate()` refactor is unchanged: `generate("3dt", 2, <scratch>, seed=0)` must give the same bytes
   before and after. Also run `python -m pytest tests -q -p no:warnings`.
3. Run `make_cone_pairs.py --n 1 --plots`.
4. Check the h5 shapes, float16 dtype, that there are no NaNs/infs, and that the ranges lie in [0, 1].
5. Sanity check: the zero-fill averaged over all 32 frames should be closer to the target than a single frame. Print
   the NRMSE of both, plus the per-frame sample count vs 64³.
6. The user reviews the PNGs/GIF before the batch.

## Status (2026-10-19): steps A–E done, awaiting review
- The container in use is the TF 2.10 image (CUDA 11.2, RTX 4090), not the sigpy one. sigpy 0.1.27 and
  cupy-cuda11x 12.3 were pip-installed with `--no-deps`, which leaves numpy at 1.23.2. The cupy NUFFT runs on the GPU.
  cupy cuSOLVER fails (`CUSOLVER_STATUS_INTERNAL_ERROR`, CUDA 11.2 libs vs sm_89), so the sigpy GPU tests fail
  there. With cupy hidden, all 114 tests pass.
- `generate("3dt", 2, seed=0)`: byte-identical before and after the `iter_recipes_3dt` refactor.
- julia_000: native shape (126, 163, 370) → 64³, 297,198 samples/frame (1.13× 64³), SNR 18.2 dB.
  NRMSE (|input| LS-scaled to target): single frame 0.455 (0.374 noise-free), 32-frame average 0.208. ~7 s/sample
  on the GPU.
- Fixes after review (numbers above/below are superseded): (1) noise SNR is now relative to mean coil-*image* power
  (as sim3d), not k-space sample power, which the dense cone centre inflated to ~−6 dB image SNR; (2) the object is
  built as in `prepare_video` (joint min-max of re/im to [0, 1] + random channel roll); the raw block's ±π phase
  speckle put ~48% of energy at |k| > N/8 (13% after). Now: 64³ NRMSE single 0.392 / 32-avg 0.073; 128³ 0.488 / 0.091.
- (3) The input is no longer combined with the true (ellipsoid-masked) maps, which zeroed it outside the object.
  As in the stack-of-stars 3D+t pairs (`new_julia_grasp_paired`), it uses `time_avg_tukey_3d` Walsh maps estimated
  from the time-summed zero-fill (Tukey width 0.25 on every axis, since the grid is isotropic; the old data used
  (1, .25, .25) with Z = 10). Target = SoS of the noise-free coil images (`target_combine='sos'`). Walsh `eigh`
  falls back to the CPU here because cuSOLVER is broken: 64³ ≈ 17 s, 128³ ≈ 112 s per sample. NRMSE (now over the
  full FOV): 64³ 0.461 / 0.088, 128³ 0.560 / 0.112.
- `--size 128` (same extent 2.4, voxel halves) → `new_julia_grasp_cone_paired_128/`, `julia_cone_3dt_128/`.
  0.14 samples/voxel/frame; NRMSE single frame 0.708, 32-frame average 0.420. ~23 s/sample, 385 MB h5 + 257 MB
  phantom per sample (500 samples ≈ 320 GB).
