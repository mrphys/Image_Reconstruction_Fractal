# Regression testing against the old code

## Golden capture (Phase 1, done)
```
python tests/regression/capture_golden.py --device gpu          # -> tests/artifacts/golden/gpu/<case>/
python tests/regression/compare.py tests/artifacts/golden/gpu <other>/gpu
```
Fixtures (`tests/regression/fixtures.py`):
- **A**: base-repo escape-time fractal, `c=(-0.02,0.38,-0.53,-0.47)` -> `make_complex_fractal` (np seed 1234)
  -> 15 coils, 13 spokes `sorted_half`, coil compression to 10.
- **B_cc**: new fractal framework recipe (seed 0) -> `process_fractals_new_grasp` config.
- **B_grouped_slice_{1,2}**: two y-slices -> `process_fractals_new_grasp_grouped` (per-group RNG reset, no compression).

Stages saved per case: `fractal_*`, `video_uint8`, `kspace`, `traj`, `dcw`, `uskspace`, `zfill`, `image`,
`zfill_rolled`, `image_rolled`, plus `manifest.json` (sha256) and GIFs in `gifs/`.

## Determinism (measured)
- CPU is unusable: TF 2.10 CPU `conv3d` (inside tfmri Walsh) segfaults. Golden is GPU-only.
- Six GPU runs of the unmodified legacy code: all stages bit-exact, **except** Walsh-combined `zfill*`
  (unanchored eigenvector phase; worst |mag diff| 1.7e-5, worst sign-invariant diff 7.6e-3).
  Rules in `tests/regression/tolerances.json`: exact everywhere, `coilmap_phase` for zfill
  (magnitude atol 1e-4, sign-invariant complex atol 2e-2).

## Tests
`python -m pytest tests -q -p no:warnings`. `tests/regression/legacy_runner.py` runs the legacy code (optionally
with injected `synthmr.rng.RNG` streams); `tests/regression/harness.py::assert_matches_golden` compares in memory
and writes `old | new | |diff|` GIFs.

## Test map
| File | What it proves |
|---|---|
| `tests/regression/test_phase2a_rng.py` | legacy code driven by `synthmr.rng` streams == golden |
| `tests/regression/test_phase2b_fractal.py` | R1-R3: escape field, 4-D recipe render, complexify == golden (exact) |
| `tests/regression/test_phase2c_simulation.py` | R4: simulated k-space == golden (exact) |
| `tests/regression/test_phase2d_end_to_end.py` | R5-R7: whole refactored chain (via `Pipeline` + presets) == golden at every stage |
| `tests/regression/test_phase2e_pipeline.py` | presets == legacy configs; config round trip; `make_dataset.py` == golden files |
| `tests/regression/test_phase3_smoke.py` | full-size `fractal_3dt_sos` run, maps vs true coils, GIFs in `tests/artifacts/phase3/smoke` |
| `tests/unit/*` | backend contract, grids, sampling, utils, generator, 3D properties (adjoint, fully-sampled recon, Z=1 == 2D, map accuracy, slice-consistent augmentation) |

Environment note: golden was captured in the `mr_recon` devcontainer (TF 2.10, tensorflow-mri built from the
`wip1200` branch, tensorflow-nufft 0.10.1). A different tfmri/nufft build may change low-order bits; recapture
golden with the frozen `legacy/` code if the environment changes, then rerun the suite.

## Per-step workflow (Phases 2 and 3)
1. Run the new code on the same fixture inputs with RNG streams seeded as in `fixtures.py`, and write stages
   into the same directory structure.
2. `compare.py golden new` must PASS for every stage.
3. Write `old | new | |diff|` GIFs (`synthmr.viz.gifs.save_comparison_gif`) to `tests/artifacts/phase<N>/<step>/`.
4. 3D+t has no golden: property tests + slice-mosaic / ortho-view GIFs.
