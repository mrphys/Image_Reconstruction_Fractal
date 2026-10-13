---
name: add-trajectory
description: Add a new sampling pattern (in-plane encoder such as radial/spiral/cartesian, or through-plane encoder such as cartesian kz) to synthmr.sampling.
---
1. A `Trajectory` = in-plane encoder (produces `(T, R, 2)` coords + dcw) x through-plane encoder (none | cartesian_kz | ...).
2. In-plane encoders: add a branch in `synthmr/backend/tf/sampling.py::make_trajectory` (and a `TrajectoryConfig.kind`).
   Through-plane encoders: add to `synthmr/sampling/encoders.py` (`THROUGH_PLANE`) and handle it in
   `SamplingOperator.forward/adjoint/apply_dcw`; route it in `backend/tf/recon.py::reconstruct`.
3. Only call MRI ops through `synthmr.backend.get_backend()` (TF code lives in `synthmr/backend/tf/`); random choices through the injected `rng`.
4. Tests (`tests/unit/test_sampling_<name>.py`): adjoint test `<Ax,y> ~= <x,A^H y>`; fully-sampled reconstruction
   NRMSE below threshold; `Z=1` matches the 2D operator.
5. Add a GIF of zero-filled vs target to `tests/artifacts/` and update docs/BACKEND.md if a new backend op was needed.
