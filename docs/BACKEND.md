# Backend contract

A backend is a package selected with `synthmr.backend.set_backend(name)` / `get_backend()`:

```
synthmr/backend/tf/
  core.py         # the ops in the table below (thin wrappers over the exact legacy TF calls)
  <stage>.py      # TF implementation of stage logic that is tensor math (simulation, recon, ...)
```
Only this package imports TensorFlow-family libraries. A torch backend (`synthmr/backend/torch/`, registered with
`register_backend`) must implement the same functions with the same semantics and pass `tests/unit/test_backend.py`
(parametrised over backends) plus the golden regression within tolerance.

Current TF stage modules: `simulation.py` (2D legacy simulation), `sampling.py` (trajectories,
`SamplingOperator` with `none` / `cartesian_kz` through-plane encoders), `recon.py` (2D recon + augmentation),
`recon3d.py` (3D recon, coil maps), `utils.py` (real-MRI helpers). The 3D simulation and 3D augmentation are
numpy and need only the core ops (`fft`, `Generator`, `to_numpy`/`from_numpy`) plus `simulation.prepare_video`.

Elementwise tensor math inside stage implementations is deliberately *not* abstracted: wrapping every op
would add indirection without making a port easier. Porting a stage means rewriting its `<stage>.py`.

| Op | Current TF call | Semantics to preserve | Torch equivalent |
|---|---|---|---|
| `fft` / `ifft` | `tfmri.signal.fft/ifft(axes, norm='ortho', shift=True)` | centred, orthonormal | TBD |
| `nufft_forward` | `tfft.nufft(type_2, forward) / sqrt(npix)` | traj in [-pi, pi], `(..., R, 2)` | TBD |
| `nufft_adjoint` | `tfft.nufft(type_1, backward, grid_shape) / sqrt(npix)` | after dcw multiplication | TBD |
| `radial_trajectory` | `tfmri.sampling.radial_trajectory` + `flatten_trajectory` | orderings: sorted_half, golden_half, tiny_half | TBD |
| `radial_density` | `tfmri.sampling.estimate_radial_density`, dcw = 1/dens | | TBD |
| `coil_compress_svd` | `tfmri.coils.CoilCompressorSVD(coil_axis=-3).fit/transform` | fit on undersampled k-space | TBD |
| `estimate_sensitivities` | `tfmri.coils.estimate_sensitivities(method='walsh')` | spatial rank 2 or 3, coil axis last | TBD |
| `combine_coils` | `tfmri.coils.combine_coils(maps=None -> SoS)` | | TBD |
| `crop_or_pad` | `tfmri.resize_with_crop_or_pad` | centred | TBD |
| `scale_by_min_max` | `tfmri.scale_by_min_max` | complex: on magnitude | TBD |
| `resize` | `tf.image.resize(bicubic)` | background phase upsampling | TBD |
| `translate` | `tfa.image.translate(bilinear)` | coil placement | TBD |
| `Generator(seed)` | `tf.random.Generator.from_seed(seed, "philox")`: `uniform`, `normal`, `reset` | draw order is part of the contract for exact regression; `synthmr.rng.RNG` holds streams transforms (seed 1), noise (2), augment (1) | TBD |
| `device(spec)` | `tf.device` | legacy hard-codes `/gpu:0`; now chosen by config | TBD |
