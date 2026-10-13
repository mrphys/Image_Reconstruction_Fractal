---
name: regression-check
description: Verify refactored code still reproduces the legacy pipeline outputs (golden stages + GIFs). Use after any change to synthmr/ that can affect 2D+t outputs.
---
1. Golden must exist in `tests/artifacts/golden/gpu/` (if not: `python tests/regression/capture_golden.py --device gpu`). GPU only; CPU segfaults in tfmri Walsh.
2. Run the regression tests: `pytest tests/regression -x` (or write new-code stages and run
   `python tests/regression/compare.py tests/artifacts/golden/gpu <new_dir>`).
3. Rules are in `tests/regression/tolerances.json`. Do not loosen a tolerance to make a test pass; find the first
   diverging stage instead (stages are ordered: fractal -> video -> kspace -> traj/dcw -> uskspace -> zfill/image -> rolled).
4. RNG draw order is part of the contract: a new random draw inserted anywhere in the 2D path breaks every later stage.
5. Open the comparison GIFs in `tests/artifacts/phase*/` and describe what they show when reporting.
